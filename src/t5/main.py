import time
import sys

from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import torch

from utils import select_optimal_device


def preprocess_fn(examples, tokenizer: T5Tokenizer, max_input_length: int, max_output_length: int, input_prefix: str):
    inputs = [f'{input_prefix}{q}' for q in examples['question']]
    targets = examples['answer']

    model_inputs = tokenizer(inputs, max_length=max_input_length,
                             truncation=True, padding=False)
    labels = tokenizer(targets, max_length=max_output_length,
                       truncation=True, padding=False)
    model_inputs['labels'] = labels['input_ids']
    model_inputs['original_inputs'] = inputs

    return model_inputs


def test_model_load(model_path: str):
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path, device_map='mps')

    model.eval()
    test_question = "What is discussed in passage 10?"
    input_text = f"Question: {test_question}"

    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(inputs=input_ids['input_ids'],
                                max_length=128,
                                temperature=0.2,
                                do_sample=True)

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nTest Question: {test_question}")
    print(f"Generated Answer: {answer}")


if __name__ == "__main__":
    device = select_optimal_device()

    tokenizer = T5Tokenizer.from_pretrained(
        "google/flan-t5-small", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-small", device_map=device)

    ds = load_dataset("basavyr/qa_tocqueville")

    max_input_length = 256
    max_output_length = 512
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

    # create train and test data
    tokenized_dataset = tokenized_dataset["train"].train_test_split(
        test_size=0.1)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model)

    # Training arguments
    num_epochs = 20
    train_bs = 8
    eval_bs = 8
    lr = 5e-5
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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

    # Train the model
    trainer.train()

    # # Save the model
    trainer.save_model('./t5-fine_tune')
    tokenizer.save_pretrained('./t5-fine_tune')

    test_model_load('./t5-fine_tune')
