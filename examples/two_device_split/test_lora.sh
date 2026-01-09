#!/bin/bash

# Test LoRA adapters by comparing base model vs fine-tuned model

python infer_with_lora.py \
  --lora_adapters ./outputs/lora_adapters_final.pt \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --model_card gpt2.sm \
  --client_layers 6 \
  --client_device cuda:0 \
  --server_device cuda:0 \
  --max_length 50 \
  --temperature 0.8 \
  --top_k 50

# You can also test with a custom prompt:
# python infer_with_lora.py \
#   --lora_adapters ./outputs/lora_adapters_final.pt \
#   --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
#   --model_card gpt2.sm \
#   --client_layers 6 \
#   --custom_prompt "Your custom prompt here"
