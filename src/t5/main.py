# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import select_optimal_device


if __name__ == "__main__":
    device = select_optimal_device()
    tokenizer = T5Tokenizer.from_pretrained(
        "google/flan-t5-small", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-small", device_map=device)

    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))
