# Split GPT-2 LoRA Fine-Tuning Across Two Devices

This directory contains a stand-alone demo that splits a GPT-2 transformer into a **client partition** (early layers) and a **server partition** (late layers). The partitions communicate using the activations at the split boundary, so you can fine-tune GPT-2 while sharing the compute load across two machines. For quick validation, the training script can also keep both halves on a single device.

## What’s Included

- `src/splitmodel.py` – GPT-2 definitions that expose `client_layers`, allowing you to choose where the model is split.
- `train_two_device_split.py` – end-to-end training loop that:
  - runs the client forward pass,
  - ships the intermediate activations to the server partition,
  - backpropagates gradients across the boundary,
  - stores LoRA adapter weights for later inference.
- Copies of the helper utilities (`data_utils.py`, `optimizer.py`, `exp_utils.py`, `gpu.py`) that the script depends on.

## Requirements

1. Python ≥ 3.8 with PyTorch ≥ 1.13 (CUDA recommended).
2. The E2E NLG dataset JSONL files from `SplitLoRA/examples/data/e2e/`.
3. A base GPT-2 checkpoint from Hugging Face, e.g. `gpt2-pytorch_model.bin` (small), `gpt2-medium-pytorch_model.bin` (medium), or `gpt2-large-pytorch_model.bin` (large).
4. The LoRA package (`pip install loralib`).

## Running the Demo on a Single Device (for smoke testing)

### Option 1: Using Pre-tokenized Data (Recommended)

If you have pre-tokenized data files (e.g., `train.jsonl`, `valid.jsonl`), you can use them directly:

```bash
cd /home/biresh/Downloads/coding/prism/SplitFM/SplitLoRA/examples/two_device_split

python train_two_device_split.py \
  --train_data ../data/e2e/train.jsonl \
  --valid_data ../data/e2e/valid.jsonl \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --model_card gpt2.sm \
  --client_layers 6 \
  --client_device cuda:0 \
  --server_device cuda:0 \
  --train_batch_size 1 \
  --seq_len 128 \
  --grad_acc 4 \
  --epochs 1 \
  --output_dir ./outputs
```

**Note for small GPUs (< 4GB):** If you encounter out-of-memory errors, try:
- Reducing `--train_batch_size` to 1
- Reducing `--seq_len` to 128 or 256
- Increasing `--grad_acc` to 4 or 8 to maintain effective batch size
- Using CPU for one partition: `--client_device cpu --server_device cuda:0`

### Option 2: Using Raw Text Data (with Auto-tokenization)

If your data contains raw text strings (e.g., `train_formatted.jsonl`), the script will automatically tokenize them if you provide the vocab path:

```bash
cd /home/biresh/Downloads/coding/prism/SplitFM/SplitLoRA/examples/two_device_split

python train_two_device_split.py \
  --train_data ../data/e2e/train_formatted.jsonl \
  --valid_data ../data/e2e/valid_formatted.jsonl \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --vocab_path ../vocab \
  --model_card gpt2.sm \
  --client_layers 6 \
  --client_device cuda:0 \
  --server_device cuda:0 \
  --train_batch_size 4 \
  --epochs 1 \
  --output_dir ./outputs
```

The script automatically:

- Loads the base GPT-2 weights into both partitions.
- Marks only the LoRA parameters as trainable.
- Logs training and validation perplexities.
- Saves LoRA adapters under `outputs/` (one file per checkpoint plus a final snapshot).

## Adapting to Two Physical Machines

The training loop is written so that only the tensor at the split point needs to travel between client and server. To distribute across machines:

1. Launch the script on the **client machine**, exporting the activation tensor (`server_input`) through your preferred transport (PyTorch RPC, gRPC, ZeroMQ, etc.).
2. Run the server partition on the **server machine**, receive the tensor, carry out the remaining layers, and return the gradient of that tensor.
3. Replace the in-process transfer in `train_two_device_split.py` with your messaging layer (see where `move_hidden_for_server` and `hidden_states.backward(...)` are called).

Because only LoRA weights are trainable, you can ship the resulting adapters and reuse the frozen GPT-2 base for inference.

## Checking Training Quality

Before testing inference, you can verify that your LoRA adapters actually learned something:

```bash
# Quick check of your final checkpoint
python check_lora_weights.py --checkpoint ./outputs/lora_adapters_final.pt

# Or analyze all checkpoints to see training progression
python check_lora_weights.py --all
```

This will show you:
- **Weight statistics** for client and server LoRA adapters
- **Training quality assessment** (not trained / under-trained / well-trained / over-trained)
- **Recommendations** on whether you need more training

**Interpreting results:**
- `< 0.001`: ❌ Training failed - check your setup
- `0.001 - 0.01`: ⚡ Under-trained - needs more epochs
- `0.01 - 0.1`: ✅ Partially trained - should work (your model is here!)
- `0.1 - 0.2`: 🎉 Well trained - excellent performance
- `> 0.2`: 🔥 Heavily trained - check for overfitting

See `WEIGHT_ANALYSIS_GUIDE.md` for detailed explanation.

## Using the Adapters for Inference

The checkpoint files saved in `output_dir` contain:

- LoRA state dict for the client partition,
- LoRA state dict for the server partition,
- Configuration metadata (`n_layer`, `client_layers`, etc.).

### Testing Your Fine-Tuned Model on E2E NLG Task

**Important:** The E2E NLG dataset trains models to generate natural language descriptions from structured restaurant data. The model expects:
- **Input (context):** Structured attributes like `name : X | Type : Y | food : Z | ...`
- **Output (completion):** Natural language description of the restaurant

We provide a specialized inference script `infer_e2e.py` for the E2E NLG task:

#### Quick Test (Using Default Examples):

```bash
# Make the test script executable
chmod +x test_e2e.sh

# Run inference comparison on default E2E examples
./test_e2e.sh
```

This will test three different restaurant contexts and show you the difference between the base model and your fine-tuned model.

#### Test With Your Own Context:

```bash
python infer_e2e.py \
  --lora_adapters ./outputs/lora_adapters_final.pt \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --vocab_path ../vocab \
  --model_card gpt2.sm \
  --client_layers 6 \
  --context "name : The Golden Dragon | Type : restaurant | food : Chinese | price : moderate"
```

#### More Examples:

```bash
# Example 1: Coffee shop with details
python infer_e2e.py \
  --lora_adapters ./outputs/lora_adapters_final.pt \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --vocab_path ../vocab \
  --context "name : Cafe Mocha | Type : coffee shop | food : Fast food | price : cheap | customer rating : 4 out of 5 | area : riverside"

# Example 2: High-end pub
python infer_e2e.py \
  --lora_adapters ./outputs/lora_adapters_final.pt \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --vocab_path ../vocab \
  --context "name : The Kings Head | Type : pub | price : more than £30 | customer rating : high | near : Train Station"

# Example 3: Family-friendly restaurant
python infer_e2e.py \
  --lora_adapters ./outputs/lora_adapters_final.pt \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --vocab_path ../vocab \
  --context "name : Pizza Paradise | Type : restaurant | food : Italian | price : £20-25 | customer rating : 5 out of 5 | area : city centre | family friendly : yes"
```

#### Input Format Guide:

Use the **same format** as the E2E training data for best results:

**Attributes you can include:**
- `name : [Restaurant Name]` (required)
- `Type : [restaurant|pub|coffee shop]` (recommended)
- `food : [Chinese|Italian|French|English|Japanese|Fast food|etc.]`
- `price : [cheap|moderate|high|£20-25|more than £30|less than £20]`
- `customer rating : [low|average|high|3 out of 5|5 out of 5]`
- `area : [riverside|city centre]`
- `family friendly : [yes|no]`
- `near : [Landmark name]`

**Format rules:**
- Separate attributes with ` | ` (space + pipe + space)
- Use ` : ` (space + colon + space) between attribute and value
- You can include any combination of attributes

The script will:
1. Generate completion with the **base model** (no LoRA) - likely to be incoherent
2. Generate completion with your **fine-tuned model** (with LoRA) - should produce proper E2E-style descriptions
3. Display both outputs for comparison

**How to know if fine-tuning worked:**
- The **base model** will likely generate random text unrelated to restaurants
- The **fine-tuned model** should generate coherent restaurant descriptions like:
  - *"The Golden Dragon is a moderately priced Chinese restaurant..."*
  - *"Located near the riverside, Cafe Mocha serves fast food..."*
- The fine-tuned output should be similar in style to the E2E training examples

## Next Steps

- Swap in `client_layers` values other than half-and-half to explore different memory/compute splits.
- Integrate PyTorch RPC or another transport to prototype true multi-node execution.
- Extend the script with resume logic or additional evaluation metrics as needed.

