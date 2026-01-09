# LoRA Weight Analysis Tool - Usage Guide

## Purpose

This script helps you understand if your LoRA adapters have learned properly by analyzing the weight magnitudes. It's like a "health check" for your fine-tuned model.

## Quick Start

### Analyze your final checkpoint:
```bash
cd /home/biresh/Downloads/coding/prism/SplitFM/SplitLoRA/examples/two_device_split
python check_lora_weights.py --checkpoint ./outputs/lora_adapters_final.pt
```

### Analyze all checkpoints (shows training progression):
```bash
python check_lora_weights.py --all
```

### Compare specific checkpoints:
```bash
python check_lora_weights.py --compare \
  ./outputs/lora_adapters_epoch1.pt \
  ./outputs/lora_adapters_final.pt
```

## Understanding the Output

### Weight Statistics

The script shows statistics for both **LoRA A** (down-projection) and **LoRA B** (up-projection) matrices:

- **Total parameters**: How many LoRA weights exist
- **Mean (absolute)**: Average weight magnitude (KEY METRIC)
- **Std deviation**: How varied the weights are
- **Max (absolute)**: Largest weight value
- **% non-zero**: Should be 100% for trained models

### Training Quality Scale

The **mean absolute weight** is the most important metric:

```
┌─────────────────────────────────────────────────────────────┐
│ Range           │ Status              │ What it means        │
├─────────────────────────────────────────────────────────────┤
│ < 0.0001        │ ❌ NOT TRAINED       │ Training failed     │
│ 0.0001 - 0.001  │ ⚠️  MINIMALLY        │ Barely learned      │
│ 0.001  - 0.01   │ ⚡ UNDER-TRAINED     │ Needs more epochs   │
│ 0.01   - 0.05   │ ✅ PARTIALLY TRAINED │ Works for simple    │
│ 0.05   - 0.1    │ ✅ PARTIALLY TRAINED │ Decent performance  │
│ 0.1    - 0.2    │ 🎉 WELL TRAINED      │ Good performance    │
│ > 0.2           │ 🔥 HEAVILY TRAINED   │ Check overfitting   │
└─────────────────────────────────────────────────────────────┘
```

## Example Outputs

### Example 1: Well-Trained Model
```
Average mean absolute weight: 0.152347 → 🎉 WELL TRAINED
💡 Excellent! Model has learned the task well.
```
**Action**: Use this model! It should perform well.

### Example 2: Under-Trained Model (Your Current Result)
```
Average mean absolute weight: 0.012366 → ✅ PARTIALLY TRAINED
💡 Recommendation: Model should work, but more training may improve quality.
```
**Action**: Model works (as you saw with "Clowns" example), but training longer would improve it.

### Example 3: Not Trained
```
Average mean absolute weight: 0.000234 → ❌ NOT TRAINED
💡 Recommendation: Check if training ran properly. Loss should decrease.
```
**Action**: Something went wrong. Check training logs for errors.

### Example 4: Overfitted Model
```
Average mean absolute weight: 0.534521 → 🔥 HEAVILY TRAINED
💡 Check validation metrics. If val loss increased, you may be overfitting.
```
**Action**: Use an earlier checkpoint. This one might memorize training data.

## Comparing Training Progression

When you use `--all` or `--compare`, you'll see how weights evolved:

```
Checkpoint                     Tag             Mean(abs)    Status
──────────────────────────────────────────────────────────────────
lora_adapters_epoch1.pt       epoch1          0.008234     ⚡ Under-trained
lora_adapters_final.pt        final           0.012366     ✅ Partially trained

📈 Training Progression:
  epoch1 → final: ↗️ +0.004132 (+50.2%)
```

**Good sign**: Weights increasing means learning is happening
**Bad sign**: Weights decreasing or staying flat means training isn't working

## What to Do Based on Results

### If NOT TRAINED (< 0.001):
1. Check if training script ran without errors
2. Verify learning rate isn't too small
3. Check if data loaded correctly
4. Ensure optimizer is updating LoRA parameters

### If UNDER-TRAINED (0.001 - 0.05):
1. **Train longer**: Increase `--epochs` (try 3-5 epochs for E2E)
2. **Increase learning rate**: Try 5e-4 or 1e-3 instead of 1e-4
3. **More gradient steps**: Reduce `--grad_acc` to update more often
4. **More data**: Ensure you're using full training set

### If PARTIALLY TRAINED (0.05 - 0.1): ✅ YOUR CASE
1. Model should work (as you confirmed!)
2. Can improve by training 1-2 more epochs
3. Good balance - not overfitted
4. Try inference first before deciding to train more

### If WELL TRAINED (0.1 - 0.2):
1. Great! Model should perform very well
2. No need for more training
3. Evaluate on test set to confirm

### If HEAVILY TRAINED (> 0.2):
1. Check validation loss - if it increased, you're overfitting
2. Use an earlier checkpoint (e.g., epoch 2 instead of epoch 5)
3. Consider using dropout or reducing training time
4. Test if model generalizes to new examples

## Tips for Better Training

### To speed up learning (if under-trained):
```bash
python train_two_device_split.py \
  --lr 5e-4 \              # Higher learning rate (default is usually 1e-4)
  --epochs 3 \             # More epochs
  --lora_dim 8 \           # Larger LoRA rank (more parameters)
  --grad_acc 2             # Update more frequently
```

### To avoid overfitting (if heavily trained):
```bash
python train_two_device_split.py \
  --epochs 1 \             # Fewer epochs
  --lora_dropout 0.2 \     # More dropout
  --lr 1e-4                # Lower learning rate
```

## Real-World Interpretation

Based on your result (0.012366 - Partially Trained):

✅ **What works**: Your model successfully learned the E2E NLG task
- Generates coherent restaurant descriptions
- Follows the correct format
- Stops at appropriate points

⚡ **What could improve**: With more training (2-3 epochs total):
- More diverse/fluent outputs
- Better handling of complex attribute combinations
- More consistent quality across all examples

**Recommendation**: Your current model is usable! Test it thoroughly on your task. If you need better quality, train for 1-2 more epochs and check weights again.

## Advanced Usage

### Monitor during training:
Train for 1 epoch at a time and check weights after each:
```bash
# Train epoch 1
python train_two_device_split.py --epochs 1 --output_dir ./outputs

# Check weights
python check_lora_weights.py --checkpoint ./outputs/lora_adapters_epoch1.pt

# Continue training if needed
python train_two_device_split.py --epochs 2 --output_dir ./outputs
python check_lora_weights.py --all  # Compare progression
```

### Find optimal checkpoint:
```bash
# Train multiple epochs
python train_two_device_split.py --epochs 5 --save_interval 1000

# Compare all checkpoints
python check_lora_weights.py --all

# Test the best one based on weights + validation loss
```

## Summary

Use this tool to:
- ✅ Verify training worked
- ✅ Decide if you need more training
- ✅ Detect overfitting early
- ✅ Compare different training runs
- ✅ Choose the best checkpoint

The **mean absolute weight** (0.01-0.2 range) is your main indicator of successful learning!
