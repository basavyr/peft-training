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
