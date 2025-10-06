# PEFT Training with PyTorch

A project for implementing PEFT with LORA and much more.

## Fine-tuning

Python commands:

```
python3 eval.py -p t5-fine_tune
python3 eval.py -p results/checkpoint-882
```


## Resources

1. Parameter-Efficient Transfer Learning for NLP: https://arxiv.org/pdf/1902.00751
2. INTRINSIC DIMENSIONALITY EXPLAINS THE EFFEC-TIVENESS OF LANGUAGE MODEL FINE-TUNING: https://arxiv.org/pdf/2012.13255
3. Finding model size (pytorch discussion): https://discuss.pytorch.org/t/finding-model-size/130275
4. For icons: https://icons8.com/icons/set/link--style-ios
5. Discussion on the value of alpha and r within LORA [here](https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal_question_what_rank_r_and_alpha_to_use_in/)
6. Tutorial on fine-tuning with LORA and quantized LORA [here](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
7. Although not discussed in the presentation, we also can use VLLM to test models. A medium post is available [here](https://medium.com/@yevhen.herasimov/serving-llama3-8b-on-cpu-using-vllm-d41e3f1731f7). Another medium article is also available [here](https://naman1011.medium.com/llm-deployment-with-vllm-62e9d912a638), which focuses in deploying LLMs with VLLM.
8. Generate data using Meta-llama `synthetic-data-kit`. Github [here](https://github.com/meta-llama/synthetic-data-kit?tab=readme-ov-file).
9. Add more sample datasets from HF [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/tree/main).
