from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from transformers import TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType

import os
import time
import sys
import argparse
from typing import Optional
import psutil

from utils import select_optimal_device, get_t5_model


def get_memory_usage(process: psutil.Process):
    """Get memory usage as shown in Activity Monitor"""
    mem_info = process.memory_info()
    # Use rss (Resident Set Size) which matches Activity Monitor's "Memory" column
    return mem_info.rss / (1024 ** 3)  # Convert to GB


# Create callback to track memory
class MemoryTrackingCallback(TrainerCallback):
    def __init__(self, process):
        self.max_memory = 0
        self.process = process

    def on_step_end(self, args, state, control, **kwargs):
        mem_info = self.process.memory_info()
        current_memory = mem_info.rss / (1024 ** 3)
        self.max_memory = max(self.max_memory, current_memory)
        return control


def use_peft() -> bool:
    parser = argparse.ArgumentParser(
        prog="Training with PEFT", description="Training with PEFT")

    parser.add_argument("--peft", action='store_true', required=False,
                        default=False, help="Use PEFT (LORA) for fine-tuning")
    args = parser.parse_args()

    return args.peft


def preprocess_fn(examples, tokenizer: T5Tokenizer, max_input_length: int, max_output_length: int, input_prefix: str, padding_strategy: str = 'max_length'):
    inputs = [f'{input_prefix}{q}' for q in examples['question']]
    targets = examples['answer']

    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding=padding_strategy)
    labels = tokenizer(targets, max_length=max_output_length,
                       truncation=True, padding=padding_strategy)
    model_inputs['labels'] = labels['input_ids']
    model_inputs['original_inputs'] = inputs

    return model_inputs


def get_tokenized_dataset(tokenizer: T5Tokenizer, max_input_length: int, max_output_length: int, automatic_split: bool = True):
    ds = load_dataset("basavyr/qa_tocqueville")

    tokenized_dataset = ds.map(
        lambda examples: preprocess_fn(
            examples,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            input_prefix="Question: "
        ),
        batched=True,
        batch_size=1,
    )

    tokenized_dataset.set_format(type="torch",
                                 columns=['id', 'question', 'answer', 'input_ids', 'attention_mask', 'labels', 'original_inputs'])

    if automatic_split:
        # create train and test data (automatic split from the entire dataset)
        tokenized_dataset = tokenized_dataset["train"].train_test_split(
            test_size=0.1)
        return tokenized_dataset["train"], tokenized_dataset["test"]
    else:
        raise ValueError("Manual dataset splitting not supported")


def train_model(model_name: str, device: str, peft_enabled: bool, num_epochs: int, train_batch_size: int, eval_batch_size: int, lr: float, use_collator: bool = False):
    process = psutil.Process(os.getpid())

    initial_memory = get_memory_usage(process)
    max_memory = initial_memory
    memory_callback = MemoryTrackingCallback(process)

    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map=device)

    train_dataset, eval_dataset = get_tokenized_dataset(tokenizer, 128, 512)
    # # get the checkpoint prefix
    # train_checkpoint_number = num_epochs*(len(train_dataset)//train_batch_size)
    # print(f'checkpoint-{train_checkpoint_number}')
    if use_collator:
        collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model)
    else:
        collator_fn = None

    output_dir = "results/base"
    if peft_enabled:
        # hint for target modules: https://huggingface.co/geektech/t5-large-lora/blob/main/adapter_config.json
        # similar config: https://medium.com/nerd-for-tech/optimizing-flan-t5-a-practical-guide-to-peft-with-lora-soft-prompts-3bab39e4a137
        rank = 4
        alpha = 8
        targets_l = ["q", "v", "o", "k", "wi", "wo"]
        targets_s = ["q", "v"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=targets_s,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["classifier"],
            task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
        )
        model = get_peft_model(model, lora_config)
        print(f'Using LORA: {lora_config}')
        model.print_trainable_parameters()
        output_dir = f"results/peft-lora_r{rank}_a{alpha}"

    # Comment on evaluation strategy (former argument eval_strategy)
    # https://naman1011.medium.com/llm-deployment-with-vllm-62e9d912a638
    # https://github.com/huggingface/setfit/issues/512
    # https://github.com/huggingface/transformers/issues/7974
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs-tensorboard',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator_fn,  # Add this for efficient batching
        processing_class=tokenizer,
        callbacks=[memory_callback],
    )

    # try:
    #     # Train the model
    #     # source: https://stackoverflow.com/questions/76217781/how-to-continue-training-with-huggingface-trainer
    #     # source: https://discuss.huggingface.co/t/how-to-resume-training-from-a-checkpoint-using-huggingface-trainer/153879
    #     trainer.train(resume_from_checkpoint=True)
    # except ValueError:
    #     print(f'No checkpoint available. Fine-tuning from scratch...')
    trainer.train()

    # Get final max memory
    final_memory = process.memory_info().rss / (1024 ** 3)
    max_memory_used = max(memory_callback.max_memory, final_memory)
    print(f"\n{'='*50}")
    print(f"Memory Statistics:")
    print(f"Initial Memory: {initial_memory:.2f} GB")
    print(f"Max Memory Used: {max_memory_used:.2f} GB")
    print(f"Memory Increase: {max_memory_used - initial_memory:.2f} GB")
    print(f"{'='*50}\n")

    trainer.save_model(output_dir=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


if __name__ == "__main__":
    device = select_optimal_device()
    model_name = get_t5_model("s")
    peft_enabled: bool = use_peft()

    # Training arguments
    train_model(model_name=model_name,
                device=device,
                peft_enabled=peft_enabled,
                num_epochs=1,
                train_batch_size=32,
                eval_batch_size=32,
                lr=5e-4,
                use_collator=False)
