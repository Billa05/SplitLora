# E2E NLG Inference - Problem and Solution

## The Problem You Encountered

When you tested your fine-tuned model, you got incoherent outputs like:
```
[BASE MODEL OUTPUT]
The price range is moderate and $500-530,000 USD. When you get here near the main airport...

[FINE-TUNED MODEL OUTPUT]
name : The Golden Dragon | Type : restaurant | food : Chinese | price : moderate - moderate |
```

## Why This Happened

The issue was with **how the model processes E2E NLG data**:

### Training Data Format
Your E2E dataset has this structure:
```json
{
  "context": "name : The Vaults | Type : pub | price : more than £30 | ...",
  "completion": "The Vaults pub near Café Adriatic has a 5 star rating..."
}
```

During training, the model sees:
```
[context_tokens] + [BOS_token] + [completion_tokens] + [EOS_token]
```

### Your Original Inference Problem
The old `infer_with_lora.py` script was using a generic GPT-2 tokenizer that:
1. Didn't use the same tokenization as your training data
2. Didn't add the BOS token between context and completion
3. Just treated the input as generic text to continue

This is why the model either:
- Generated random text (base model)
- Just repeated the input (fine-tuned model confused)

## The Solution

I created a new script `infer_e2e.py` that properly handles E2E NLG format:

### What It Does Differently

1. **Uses the correct encoder** (from your `../vocab` directory)
   ```python
   encoder = get_encoder(args.vocab_path)
   ```

2. **Formats input correctly** by adding BOS token after context:
   ```python
   context_tokens, _ = encoder.encode(context)
   bos_token = 50256  # GPT-2 BOS/EOS token
   input_ids = context_tokens + [bos_token]
   ```

3. **Generates only the completion** (not the context)
   ```python
   # Model generates tokens after the BOS
   # Then extracts just the completion part
   ```

## How to Use It

### Option 1: Quick Test (Default Examples)
```bash
cd /home/biresh/Downloads/coding/prism/SplitFM/SplitLoRA/examples/two_device_split
./test_e2e.sh
```

### Option 2: Test Your Own Context
```bash
python infer_e2e.py \
  --lora_adapters ./outputs/lora_adapters_final.pt \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --vocab_path ../vocab \
  --context "name : The Golden Dragon | Type : restaurant | food : Chinese | price : moderate"
```

## What to Expect

### Base Model (No LoRA)
Should produce **incoherent/random** text because it wasn't trained on E2E format.

### Fine-Tuned Model (With LoRA)
Should produce **natural restaurant descriptions** like:
- *"The Golden Dragon is a moderately priced Chinese restaurant located in the city centre."*
- *"Cafe Mocha serves fast food near the riverside with a customer rating of 4 out of 5."*

## Troubleshooting

### If you still get bad outputs:

1. **Check training completed properly:**
   ```bash
   # Look for checkpoint files
   ls -lh ./outputs/
   # Should see: lora_adapters_epoch1.pt, lora_adapters_final.pt
   ```

2. **Check training loss decreased:**
   - Training loss should decrease from initial value (~3-4) to lower values
   - Validation perplexity should decrease
   - If loss didn't decrease much, you may need to train longer

3. **Try different generation parameters:**
   ```bash
   # Lower temperature = more focused/deterministic
   python infer_e2e.py ... --temperature 0.5
   
   # More tokens
   python infer_e2e.py ... --max_length 100
   ```

4. **Ensure you trained on the RIGHT data:**
   - If you used `train.jsonl` (pre-tokenized), that's correct
   - If you used `train_formatted.jsonl`, make sure you passed `--vocab_path ../vocab`

## Key Takeaways

1. **Task-specific inference matters**: E2E NLG needs special handling because it's a structured-to-text generation task, not free-form text generation.

2. **Tokenization consistency**: Must use the same tokenizer/encoder during training AND inference.

3. **Format matters**: The BOS token acts as a separator between context and completion.

4. **Fine-tuning needs enough data/steps**: If your model didn't see enough examples or train long enough, it won't learn the task properly.
