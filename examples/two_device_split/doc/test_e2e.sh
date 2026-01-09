#!/bin/bash
# Test script for E2E NLG generation with LoRA adapters

cd "$(dirname "$0")"

echo "Testing E2E NLG generation with fine-tuned LoRA model..."
echo "=========================================="

python infer_e2e.py \
  --lora_adapters ./outputs/lora_adapters_final.pt \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --vocab_path ../vocab \
  --model_card gpt2.sm \
  --client_layers 6 \
  --client_device cuda:0 \
  --server_device cuda:0 \
  --max_length 80 \
  --temperature 0.7

echo ""
echo "=========================================="
echo "Test complete!"
echo ""
echo "To test with your own context, run:"
echo "python infer_e2e.py \\"
echo "  --lora_adapters ./outputs/lora_adapters_final.pt \\"
echo "  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \\"
echo "  --vocab_path ../vocab \\"
echo "  --context 'name : Your Restaurant | Type : restaurant | food : Italian | price : moderate'"
