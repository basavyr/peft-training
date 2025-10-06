# PEFT Training with PyTorch

A project for implementing PEFT with LORA and much more.

## Fine-tuning

Python commands:

```
python3 eval.py -p <path_to_model_checkpoint>
```

## Measurements

LORA with multiple ranks.
We set `alpha = 2* rank`. Target modules: `targets_s = ["q", "v"]`.
$r=4$.
```
Using LORA: LoraConfig(task_type=<TaskType.SEQ_2_SEQ_LM: 'SEQ_2_SEQ_LM'>, peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='google/flan-t5-small', revision=None, inference_mode=False, r=4, target_modules={'v', 'q'}, exclude_modules=None, lora_alpha=8, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=['classifier'], init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', trainable_token_indices=None, loftq_config={}, eva_config=None, corda_config=None, use_dora=False, use_qalora=False, qalora_group_size=16, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lora_bias=False, target_parameters=None)
trainable params: 172,032 || all params: 77,133,184 || trainable%: 0.2230
```
$r=8$.
```
Using LORA: LoraConfig(task_type=<TaskType.SEQ_2_SEQ_LM: 'SEQ_2_SEQ_LM'>, peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='google/flan-t5-small', revision=None, inference_mode=False, r=8, target_modules={'q', 'v'}, exclude_modules=None, lora_alpha=16, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=['classifier'], init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', trainable_token_indices=None, loftq_config={}, eva_config=None, corda_config=None, use_dora=False, use_qalora=False, qalora_group_size=16, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lora_bias=False, target_parameters=None)
trainable params: 344,064 || all params: 77,305,216 || trainable%: 0.4451
```
$r=16$.
```
Using LORA: LoraConfig(task_type=<TaskType.SEQ_2_SEQ_LM: 'SEQ_2_SEQ_LM'>, peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='google/flan-t5-small', revision=None, inference_mode=False, r=16, target_modules={'q', 'v'}, exclude_modules=None, lora_alpha=32, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=['classifier'], init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', trainable_token_indices=None, loftq_config={}, eva_config=None, corda_config=None, use_dora=False, use_qalora=False, qalora_group_size=16, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lora_bias=False, target_parameters=None)
trainable params: 688,128 || all params: 77,649,280 || trainable%: 0.8862
```
$r=32$.
```
Using LORA: LoraConfig(task_type=<TaskType.SEQ_2_SEQ_LM: 'SEQ_2_SEQ_LM'>, peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='google/flan-t5-small', revision=None, inference_mode=False, r=32, target_modules={'q', 'v'}, exclude_modules=None, lora_alpha=64, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=['classifier'], init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', trainable_token_indices=None, loftq_config={}, eva_config=None, corda_config=None, use_dora=False, use_qalora=False, qalora_group_size=16, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lora_bias=False, target_parameters=None)
trainable params: 1,376,256 || all params: 78,337,408 || trainable%: 1.7568
```

## Tracking the gradient change.

The evolution of gradients during training. We also set the target modules `targets_s = ["q", "v"]` (same as above experiments).
Experiment with $r=8$ and $alpha=8$.
```
{'loss': 31.0025, 'grad_norm': 1.3949666023254395, 'learning_rate': 4.4999999999999996e-05, 'epoch': 0.31}                                                                               
{'loss': 30.9866, 'grad_norm': 1.6945232152938843, 'learning_rate': 9.5e-05, 'epoch': 0.62}                                                                                              
{'loss': 30.5885, 'grad_norm': 2.20668625831604, 'learning_rate': 0.000145, 'epoch': 0.94}
```

Experiment with $r=8$ and $alpha=16$.
```
{'loss': 31.1164, 'grad_norm': 2.7735562324523926, 'learning_rate': 4.4999999999999996e-05, 'epoch': 0.31}                                                                               
{'loss': 30.9172, 'grad_norm': 3.5233919620513916, 'learning_rate': 9.5e-05, 'epoch': 0.62}                                                                                              
{'loss': 30.4293, 'grad_norm': 4.484832763671875, 'learning_rate': 0.000145, 'epoch': 0.94}
```

Experiment with $r=8$ and $alpha=32$.
```
{'loss': 31.0987, 'grad_norm': 5.783611297607422, 'learning_rate': 4.4999999999999996e-05, 'epoch': 0.31}                                                                                
{'loss': 30.6659, 'grad_norm': 6.987475872039795, 'learning_rate': 9.5e-05, 'epoch': 0.62}                                                                                               
{'loss': 29.5352, 'grad_norm': 9.144457817077637, 'learning_rate': 0.000145, 'epoch': 0.94}
```

Final (optional) experiment for $r=8$ and $alpha=4$.
```
Using LORA: LoraConfig(task_type=<TaskType.SEQ_2_SEQ_LM: 'SEQ_2_SEQ_LM'>, peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='google/flan-t5-small', revision=None, inference_mode=False, r=8, target_modules={'v', 'q'}, exclude_modules=None, lora_alpha=4, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=['classifier'], init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', trainable_token_indices=None, loftq_config={}, eva_config=None, corda_config=None, use_dora=False, use_qalora=False, qalora_group_size=16, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lora_bias=False, target_parameters=None)
trainable params: 344,064 || all params: 77,305,216 || trainable%: 0.4451
{'loss': 31.0306, 'grad_norm': 0.6093101501464844, 'learning_rate': 4.4999999999999996e-05, 'epoch': 0.31}                                                                               
{'loss': 30.9857, 'grad_norm': 0.9116100072860718, 'learning_rate': 9.5e-05, 'epoch': 0.62}                                                                                              
{'loss': 30.847, 'grad_norm': 1.1311554908752441, 'learning_rate': 0.000145, 'epoch': 0.94}
```

## Memory Efficiency

For a `T5-small` model, with `per_device_train_batch_size=32`
- FT: ~37.60 GB
- LORA: ~28.0 GB

> *Tested on M4 MacBook Pro*

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
10. Hugging-face Fine-Tuning guide [here](https://huggingface.co/docs/transformers/main/en/training).
11. Several videos on this Topic:
    -  LoRA explained (and a bit about precision and quantization) - [link](https://www.youtube.com/watch?v=t509sv5MT0w&t=438s)
    -   LoRA & QLoRA Fine-tuning Explained In-Depth - [link](https://www.youtube.com/watch?v=t1caDsMzWBk)
    -  What is Low-Rank Adaptation (LoRA) | explained by the inventor - [link](https://www.youtube.com/watch?v=DhRoTONcyZE)
    -  LoRA - Explained! - [link](https://www.youtube.com/watch?v=Bq9zqTJDsjg)
12. Finetuning of LLM (model: T5–3b) using Lora and QLORA (medium article). Uses `bitsandbytes` for quantization. Source [here](https://medium.com/@alishafique3/finetuning-of-llm-model-t5-3b-on-single-gpu-using-qlora-for-summarization-task-ac40ae7ae2ca).
    ```python
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )
    ```
    Finetuning of LLM (model: T5–3b) on single GPU using QLoRA for summarization task.
    Provides full code with the implementation.
13. Layer Normalization is used to avoid **Internal Covariate Shift**. A useful article on this topic is available [here](https://www.geeksforgeeks.org/deep-learning/internal-covariant-shift-problem-in-deep-learning/). Main idea is available on this arxiv paper [link](https://arxiv.org/pdf/1502.03167). "We define Internal Covariate Shift as the change in the distribution of network activations due to the change in network parameters during training.
14. Great article with code implementation and **production tips** (such as using batched input and different checkpoint paths for adapters when loading them). See post: [LoRA Fine-Tuning Tutorial: Reduce GPU Memory Usage by 90% in 2025](https://markaicode.com/lora-fine-tuning-tutorial-reduce-gpu-memory/).