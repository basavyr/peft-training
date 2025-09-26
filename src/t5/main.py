import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import select_optimal_device

import time

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)


def preprocess_fn(examples, tokenizer: T5Tokenizer, max_input_length: int, max_output_length: int, input_prefix: str):
    inputs = [f'{input_prefix}{q}' for q in examples['question']]
    targets = examples['answer']

    model_inputs = tokenizer(inputs, max_length=max_input_length,
                             truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_output_length,
                       truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    model_inputs['original_inputs'] = inputs

    return model_inputs


if __name__ == "__main__":
    device = select_optimal_device()

    tokenizer = T5Tokenizer.from_pretrained(
        "google/flan-t5-small", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-small", device_map=device)

    ds = load_dataset("basavyr/qa_tocqueville")

    tokenized_dataset = ds.map(
        lambda examples: preprocess_fn(
            examples,
            tokenizer=tokenizer,
            max_input_length=512,
            max_output_length=512,
            input_prefix="Question: "
        ),
        batched=True,
        batch_size=2
    )

    tokenized_dataset.set_format(type="torch",
                                 columns=['id', 'question', 'answer', 'input_ids', 'attention_mask', 'labels', 'original_inputs'])

    # create train and test data
    tokenized_dataset = tokenized_dataset["train"].train_test_split(
        test_size=0.1)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]
