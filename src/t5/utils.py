import os
import torch


RESULTS_DIR = "./results"
DEFAULT_SEED = 1137


def print_trainable_parameters(model: torch.nn.Module):
    """
    Source: https://github.com/huggingface/peft/issues/41

    Docs: 
    How to Estimate the Number of Parameters in Transformer models: https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0/
    """
    all_params = 0
    trainable_params = 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    print(f'All params: {all_params} (trainable: {trainable_params})')
    print(f'Trainable {(trainable_params/all_params*100):.3f} %')


def select_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_t5_model(size: str = "s"):
    """
    - Get the original Text-to-Text Transfer Transformer (T5)
    - Original paper: https://arxiv.org/pdf/1910.10683
    """
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
