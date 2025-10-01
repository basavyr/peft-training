import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

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


def eval_model(model_name: str, deterministic: bool, device: str):
    if deterministic:
        torch.manual_seed(DEFAULT_SEED)
        passage_id = 10
    else:
        passage_id = torch.randint(0, 100, (1,)).item()

    model_path: str = get_model_path()
    if validate_checkpoint_dir(model_path):
        model_name = model_path
        print(f'Found valid model checkpoint: {model_name}')
    else:
        print(
            f'Invalid checkpoint. Using the original model name: {model_name}')
    try:
        config = PeftConfig.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            device_map=device)
        model = PeftModel.from_pretrained(model, model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        print(f'Loaded {model._get_name()} on < device = {model.device} >')
    except Exception:
        raise FileNotFoundError(
            f"Incorrect model path or model name: {model_name}")

    question = f"Question: What is discussed in passage {passage_id} ?"
    model_input = tokenizer(question, return_tensors='pt').to(model.device)

    model_output = run_inference(input_ids=model_input['input_ids'],
                                 model=model)
    answer = tokenizer.decode(model_output, skip_special_tokens=True)

    print_trainable_parameters(model)
    print(question)
    print(f'Answer: {answer}')


if __name__ == "__main__":
    device = select_optimal_device()
    model_name = get_t5_model(size="s")
    eval_model(model_name=model_name, deterministic=True, device=device)
