from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import torch

import time
import sys

from utils import select_optimal_device, get_t5_model


def preprocess_fn(examples, tokenizer: T5Tokenizer, max_input_length: int, max_output_length: int, input_prefix: str):
    inputs = [f'{input_prefix}{q}' for q in examples['question']]
    targets = examples['answer']

    model_inputs = tokenizer(inputs, max_length=max_input_length,
                             truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=max_output_length,
                       truncation=True, padding='max_length')
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


def train_model(model_name: str, device: str, output_dir: str, num_epochs: int, train_batch_size: int, eval_batch_size: int, lr: float, use_collator: bool = False):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map=device)

    train_dataset, eval_dataset = get_tokenized_dataset(tokenizer, 128, 512)

    if use_collator:
        collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model)
    else:
        collator_fn = None

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        learning_rate=lr,
        lr_scheduler_type="linear",
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
    )

    try:
        # Train the model
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        print(f'No checkpoint available. Fine-tuning from scratch...')
        trainer.train()

    # # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    device = select_optimal_device()
    output_dir = "results"
    model_name = get_t5_model("s")

    # Training arguments
    train_model(model_name=model_name,
                device=device,
                output_dir=output_dir,
                num_epochs=5,
                train_batch_size=4,
                eval_batch_size=4,
                lr=5e-4,
                use_collator=False)
