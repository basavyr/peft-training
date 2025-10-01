import os
import torch


RESULTS_DIR = "./results"
DEFAULT_SEED = 1137


def print_trainable_parameters(model: torch.nn.Module):
    for k, v in model.named_parameters():
        print(k, v.shape)


def select_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_t5_model(size: str = "s"):
    if size == "s":
        # 77M
        return "google/flan-t5-small"
    elif size == "b":
        # 248M
        return "google/flan-t5-base"
    elif size == "l":
        # 783M
        return "google/flan-t5-large"
    elif size == "xl":
        # 2.85b
        return "google/flan-t5-xl"
    else:
        return "google/flan-t5-small"


def validate_checkpoint_dir(results_dir: str):
    static_files = [
        "added_tokens.json", "config.json", "generation_config.json", "model.safetensors",
        "special_tokens_map.json", "spiece.model", "tokenizer_config.json", "training_args.bin"]

    try:
        checkpoint_files = list(
            filter(lambda x: 'checkpoint-' not in x, os.listdir(results_dir)))

        for s in static_files:
            if s not in checkpoint_files:
                return False
        return True
    except FileNotFoundError:
        print(f'Could not find results dir: [{results_dir}]')
        return False
