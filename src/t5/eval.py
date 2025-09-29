import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import time
import sys
import argparse

from utils import select_optimal_device


def get_model_path() -> str:
    parser = argparse.ArgumentParser(
        prog="Test model", description='Testing model')
    parser.add_argument(
        "--path", '-p', default="t5-fine_tune", type=str, required=True, help='Path to the finetuned model.')
    return str(parser.parse_args().path)


def run_inference(input_ids: None, model: T5ForConditionalGeneration, deterministic: bool):
    if deterministic:
        torch.manual_seed(1137)
    inf_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        model_output = model.generate(
            inputs=input_ids,
            max_length=128,
            temperature=0.15,
            do_sample=True)  # return IDs
    inf_finish = time.perf_counter() - inf_start
    print(f'Inference finished: {inf_finish:3f} s')
    return model_output[0]


def main(deterministic: bool, device):
    if deterministic:
        torch.manual_seed(1137)
        passage_id = 10
    else:
        passage_id = torch.randint(0, 100, (1,)).item()

    model_path: str = get_model_path()
    try:
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map=device)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        print(f'Loaded {model._get_name()} on < device = {model.device} >')
    except Exception:
        raise FileNotFoundError(f"Incorrect model path: {model_path}")

    question = f"Question: What is discussed in passage {passage_id} ?"
    model_input = tokenizer(question, return_tensors='pt').to(model.device)

    model_output = run_inference(input_ids=model_input['input_ids'],
                                 model=model,
                                 deterministic=deterministic)
    answer = tokenizer.decode(model_output, skip_special_tokens=True)

    print(question)
    print(f'Answer: {answer}')


if __name__ == "__main__":
    device = select_optimal_device()
    main(deterministic=True, device=device)
