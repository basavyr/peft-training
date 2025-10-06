import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# how to get last checkpoint: https://github.com/huggingface/trl/issues/674
from transformers.trainer_utils import get_last_checkpoint

from peft import PeftConfig, PeftModel

import time
import sys
import argparse

from utils import select_optimal_device, DEFAULT_SEED, RESULTS_DIR, get_t5_model, validate_checkpoint_dir, print_trainable_parameters


def get_model_path() -> str:
    parser = argparse.ArgumentParser(
        prog="Test model", description='Testing model')
    parser.add_argument(
        "--path", '-p', type=str, required=True, help=f'Path to the finetuned model. (Default one = {RESULTS_DIR})')
    return str(parser.parse_args().path)


def run_inference(input_ids: torch.Tensor, model: T5ForConditionalGeneration):
    inf_start = time.perf_counter()

    model.eval()
    with torch.no_grad():
        model_output = model.generate(
            inputs=input_ids,
            max_length=128,
            temperature=0.15,
            do_sample=True)  # return IDs
    inf_finish = time.perf_counter() - inf_start

    print(f'Inference finished: {inf_finish:.3f} s')
    return model_output[0]


def eval_model(path: str, deterministic: bool, device: str):
    if deterministic:
        torch.manual_seed(DEFAULT_SEED)
        passage_id = 10
    else:
        passage_id = torch.randint(0, 100, (1,)).item()

    tokenizer = T5Tokenizer.from_pretrained(path, legacy=False)
    original_model = T5ForConditionalGeneration.from_pretrained(
        path,
        device_map=device)

    try:
        peft_model = PeftModel.from_pretrained(original_model,
                                               path,
                                               is_trainable=False)
    except ValueError:
        print(f'Could not find peft model')
        peft_model = original_model

    print(
        f'Loaded {peft_model._get_name()} on < device = {peft_model.device} >')

    question = f"Question: What is discussed in passage {passage_id} ?"
    model_input = tokenizer(
        question, return_tensors='pt').to(peft_model.device)

    model_output = run_inference(input_ids=model_input['input_ids'],
                                 model=peft_model)
    answer = tokenizer.decode(model_output, skip_special_tokens=True)

    # print_trainable_parameters(peft_model)
    print(question)
    print(f'Answer: {answer}')


if __name__ == "__main__":
    model_path = get_model_path()
    eval_model(path=model_path,
               deterministic=True,
               device=select_optimal_device())
