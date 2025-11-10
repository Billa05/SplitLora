# Root Cause Analysis: Poor Generation Quality

### **CRITICAL: Training/Inference Separator Mismatch**
**Issue**: The training data uses `<|endoftext|>` (token 50256) as separator:
```
context: "name : Blue Spice | ... | area : riverside<|endoftext|>"
completion: " Blue Spice is a French restaurant..."
```

But your inference prompt uses `||` (token 15886):
```
"name : Blue Spice | ... | area : riverside||"
```
---
**Impact**: The model learned to generate text after seeing token 50256, but during inference you're giving it token 15886. This causes the model to be confused and generate poor quality text.

**Fix**: Use `<|endoftext|>` in inference prompts instead of `||`, OR retrain with `||` separator.

## Current Situation
After training for 2000 steps with the correct data format, the model generates:
```
Input:  "name : Blue Spice | food : Italian | price : less than £ 20 | area : riverside<|endoftext|>"
Output: "name : Blue Spice | food : Italian | price : less than £ 20 | area : riverside Spice Italian riverside Blue"
```

Expected:
```
"The Blue Spice offers Italian food in riverside for less than £ 20."
```

## Issues Identified

### 1. ✅ Data Format - CORRECT
- Training data uses `<|endoftext|>` separator (token 50256) ✓
- Inference uses `<|endoftext|>` separator ✓
- Data is properly tokenized ✓

### 2. ⚠️ LoRA Parameters Barely Updated
**Critical Issue**: LoRA matrices have very small values after 2000 steps:
- Standard deviation: ~0.0008
- Max value: ~0.004
- This means the model learned **almost nothing**

**Why?**
- Learning rate too low: 1e-5
- Only 2000 steps (with batch_size=4 = 8000 samples total)
- LoRA scaling: alpha/r = 16/16 = 1.0 (reasonable)

### 3. ⚠️ Training Not Long Enough
- E2E dataset has 42,062 training examples
- Batch size = 4
- 2000 steps = only **0.19 epochs** (seeing 8000 out of 42,062 examples)
- Model barely saw the data!

### 4. ⚠️ Learning Rate Too Conservative
- Current: 1e-5
- For LoRA fine-tuning, typical range is **1e-4 to 5e-4**
- Our LR is 10-50x too small

### 5. ⚠️ No Loss Monitoring
- Can't verify if training is working without seeing loss curves
- Need to log loss every few steps to Device 0

## Recommended Fixes (Priority Order)

### Priority 1: Increase Learning Rate
**Change**: `lr: float = 0.0001` (10x increase from 1e-5 to 1e-4)

**Why**: LoRA adapters need stronger updates to actually learn patterns.

### Priority 2: Train for More Steps
**Options**:
- **Option A**: 10,000 steps (~1 epoch) - Moderate training
- **Option B**: 20,000 steps (~2 epochs) - Better training
- **Option C**: Until convergence - Best quality

For E2E dataset, **at least 1 full epoch** (10,500 steps with batch_size=4).

### Priority 3: Add Loss Logging to Device 0
Currently only Device 1 sees the loss. Device 0 should also receive and log it.

### Priority 4: Verify Gradient Flow
The gradients might be too small. Check if:
- Gradients are properly flowing back from Device 1 to Device 0
- Gradient norms are reasonable (should be ~0.1 to 10)

## Quick Test: Train with Better Settings

### Minimal Fix (Fast test - 30 min):
```python
lr: float = 0.0001  # 10x increase
max_step: int = 2000  # Keep for quick test
```

### Recommended Fix (2-3 hours):
```python
lr: float = 0.0001  # 10x increase  
max_step: int = 10000  # ~1 epoch
```

### Best Quality (6-8 hours):
```python
lr: float = 0.0001  # 10x increase
max_step: int = 21000  # ~2 epochs
```

## What to Expect After Fixes

### With Current Settings (lr=1e-5, 2000 steps):
- LoRA std: ~0.0008
- Result: Random/repetitive generation ❌

### With Better Settings (lr=1e-4, 10000 steps):
- LoRA std: ~0.05-0.1 (60-125x larger)
- Result: Coherent sentences, some errors ✓

### With Best Settings (lr=1e-4, 20000 steps):
- LoRA std: ~0.1-0.2 (125-250x larger)
- Result: High quality generation ✓✓

## Additional Issues (Lower Priority)

### Inference Temperature
Current default: 0.8 - this is fine

Could try: 0.5-0.7 for more deterministic output

### Batch Size
Current: 4 - reasonable for memory
Could increase to 8 if you have GPU memory

### Warmup
Current: 0 steps
Could add: warmup_step = 500 for more stable training

## Summary

**The main issues are:**
1. 🔴 Learning rate is **10-50x too small** (1e-5 vs 1e-4 to 5e-4)
2. 🔴 Only trained for **0.19 epochs** (2000 steps / 10,531 steps per epoch)
3. 🟡 No loss visibility on Device 0

**The model architecture and data are correct**, but the model **hasn't actually learned anything yet** because:
- Learning rate too conservative
- Not enough training steps
- LoRA parameters barely moved from initialization

**Next steps:**
1. Increase LR to 1e-4
2. Train for at least 10,000 steps (1 epoch)
3. Add loss logging to see progress
4. Retrain and test

The generation should improve dramatically with these changes.
