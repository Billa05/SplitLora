#!/bin/bash
# Test inference with the CORRECT separator format

echo "Testing inference with <|endoftext|> separator (matches training data)..."
echo ""

python infer_2device.py \
  --prompt "name : Blue Spice | Type : restaurant | food : French | price : more than £ 30 | area : riverside<|endoftext|>" \
  --max_length 50 \
  --temperature 0.7 \
  --top_k 40 \
  --top_p 0.9 \
  --repetition_penalty 1.2

echo ""
echo "=========================================="
echo ""
echo "More test examples:"
echo ""

echo "Example 2: Italian restaurant"
python infer_2device.py \
  --prompt "name : The Olive Garden | Type : restaurant | food : Italian | price : £ 20 - 25 | area : city centre<|endoftext|>" \
  --max_length 40 \
  --temperature 0.7 \
  --top_k 40 \
  --top_p 0.9 \
  --repetition_penalty 1.2

echo ""
echo "Example 3: Coffee shop"
python infer_2device.py \
  --prompt "name : Café Mocha | Type : coffee shop | food : Fast food | price : cheap | customer rating : high | family friendly : yes<|endoftext|>" \
  --max_length 40 \
  --temperature 0.7 \
  --top_k 40 \
  --top_p 0.9 \
  --repetition_penalty 1.2
