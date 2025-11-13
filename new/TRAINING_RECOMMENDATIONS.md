# Training Recommendations for E2E Dataset

## Dataset Analysis

### Dataset Statistics
- **Training examples**: 42,060
- **Batch size**: 4
- **Sequence length**: 128
- **Steps per epoch**: ~10,515 (42,060 / 4)

### Task Type
The E2E (End-to-End) dataset is a **data-to-text generation** task where:
- **Input (context)**: Structured data with attributes (name, type, price, rating, location, etc.)
- **Output (completion)**: Natural language descriptions of restaurants/establishments
- **Complexity**: Medium - requires learning to convert structured data into fluent English

### Data Characteristics
```
Example:
Context: "name : The Vaults | Type : pub | price : more than £30 | customer rating : 5 out of 5 | near : Café Adriatic"
Completion: "The Vaults pub near Café Adriatic has a 5 star rating. Prices start at £30."
```

## Recommended Number of Epochs

### For LoRA Fine-tuning on GPT-2:

#### **RECOMMENDED: 3-5 Epochs**

Here's why:

1. **Minimum Effective Training**: 
   - **2 epochs** (~21,030 steps) - Basic learning, model starts to understand the task
   - **3 epochs** (~31,545 steps) - Good learning, generates coherent outputs ✅ **RECOMMENDED START**
   - **4 epochs** (~42,060 steps) - Better quality, more consistent outputs ✅ **OPTIMAL**
   - **5 epochs** (~52,575 steps) - High quality, near-optimal performance ✅ **BEST QUALITY**

2. **Why Not More?**
   - **6+ epochs** - Diminishing returns, risk of overfitting
   - LoRA is efficient and learns quickly
   - E2E dataset has repetitive patterns (same restaurants, similar descriptions)
   - Too many epochs may cause the model to memorize rather than generalize

3. **Why Not Less?**
   - **1 epoch** (~10,515 steps) - Insufficient for good quality (as you experienced: UNDER_TRAINED)
   - Model needs multiple exposures to learn the structured-to-text mapping well

### Epoch Recommendations by Goal:

| Goal | Epochs | Steps | Expected Quality |
|------|--------|-------|------------------|
| **Quick Test** | 1 | ~10,515 | Poor - under-trained |
| **Minimum Viable** | 2 | ~21,030 | Basic - some coherence |
| **Good Quality** | 3 | ~31,545 | Good - mostly correct ⭐ |
| **High Quality** | 4 | ~42,060 | Very Good - reliable ⭐⭐ |
| **Best Quality** | 5 | ~52,575 | Excellent - near-optimal ⭐⭐⭐ |
| **Experimental** | 6-7 | ~63,090+ | May overfit ⚠️ |

## Expected LoRA Parameter Changes

Based on your current training setup (lr=0.0001, rank=16, alpha=16):

### After 3 Epochs:
- **LoRA_B Std**: 0.03 - 0.08 (PARTIALLY_TRAINED to WELL_TRAINED)
- **LoRA_B |Max|**: 0.08 - 0.15
- **Status**: Good for most use cases

### After 4-5 Epochs:
- **LoRA_B Std**: 0.06 - 0.12 (WELL_TRAINED)
- **LoRA_B |Max|**: 0.12 - 0.20
- **Status**: High quality, production-ready

## Training Time Estimates

With your current setup (~1.5-2 steps/second):

| Epochs | Total Steps | Estimated Time |
|--------|-------------|----------------|
| 2 | ~21,030 | ~3-4 hours |
| 3 | ~31,545 | ~4.5-6 hours |
| 4 | ~42,060 | ~6-8 hours |
| 5 | ~52,575 | ~7.5-10 hours |

## Recommended Configuration

Update `ws_client.py`:

```python
class ClientArgs:
    # ... other settings ...
    num_epochs: int = 4  # Optimal balance of quality and training time
    lr: float = 0.0001   # Good for LoRA
    lora_dim: int = 16   # Good rank
    lora_alpha: int = 16 # Good scaling
```

## Training Strategy

### Option A: Conservative (Recommended for first run)
```python
num_epochs: int = 3  # Safe, good quality
```
- Check results after training
- If quality is insufficient, train 1-2 more epochs

### Option B: Optimal (Best for production)
```python
num_epochs: int = 4  # Best balance
```
- Reliable quality
- Not too much training time
- Low risk of overfitting

### Option C: Maximum Quality
```python
num_epochs: int = 5  # Push for best results
```
- Highest quality without overfitting
- Longer training time
- Best for final model

## Monitoring During Training

Watch for these signs:

### Good Training:
- Loss decreases steadily across epochs
- LoRA_B parameters grow from ~0.01 to 0.05-0.12 std
- Generated text becomes more coherent

### Overfitting (stop if you see):
- Loss stops decreasing or increases
- LoRA_B parameters become very large (std > 0.2)
- Model outputs become too similar to training examples

## Validation Strategy

After training, check quality:
1. Run inference on test set
2. Compare generated descriptions with reference
3. Check for:
   - Factual accuracy (correct attributes mentioned)
   - Fluency (natural English)
   - Completeness (all important attributes covered)

## Summary

**For your E2E dataset with 42,060 examples:**

🎯 **Best Recommendation**: **4 epochs** (~42,060 steps)
- Optimal quality-time trade-off
- Should achieve "WELL_TRAINED" status
- Approximately 6-8 hours of training

🚀 **Quick Start**: **3 epochs** if you want faster results
⭐ **Maximum Quality**: **5 epochs** if you want the absolute best

**Do NOT train for more than 6 epochs** - diminishing returns and overfitting risk!
