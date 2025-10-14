# Distributed GPT-2 Training with LoRA Adapters

This project implements distributed training of GPT-2 models using Low-Rank Adaptation (LoRA) across multiple devices connected via WebSockets. The model is split into parts, with each device handling a subset of layers, enabling efficient pipeline parallelism.

## Project Structure

### Core Files

#### `splitmodel.py`
Main implementation for splitting GPT-2 into distributed parts.
- `GPT2SplitPart`: A PyTorch module that represents a portion of the GPT-2 model for a specific device
- Handles embedding layers, transformer layers, and language model head distribution
- Integrates LoRA adapters for parameter-efficient fine-tuning
- `get_lora_config()`: Creates LoRA configuration with specified rank, alpha, and dropout

#### `ws_client.py`
WebSocket-based client for distributed training communication.
- Implements pipeline parallelism across devices
- Handles forward/backward passes through device-specific model parts
- Manages gradient communication between devices
- Supports training loop for the first device (with data) and inference for subsequent devices
- Saves trained LoRA adapters to disk

#### `infer_2device.py`
Inference script for the trained 2-device pipeline.
- Loads trained LoRA adapters for both devices
- Performs text generation using the distributed model
- Supports temperature sampling and top-k filtering
- Command-line interface with configurable generation parameters

#### `run_pipeline.sh`
Shell script to orchestrate the training pipeline.
- Starts devices in reverse order (last device first)
- Manages process lifecycle and logging
- Monitors training progress and handles cleanup
- Saves LoRA adapters after training completion

### Data Files
- `data/e2e/`: End-to-end training data directory
  - `train.jsonl`, `valid.jsonl`: Training and validation datasets
  - `train_formatted.jsonl`, `valid_formatted.jsonl`: Preprocessed datasets
  - `train.txt`, `valid.txt`: Raw text files
  - Additional split files (`train0.jsonl`, `train1.jsonl`, etc.)

### Model and Log Files
- `lora_device_0.pth`, `lora_device_1.pth`: Trained LoRA adapters for each device
- `device_0.log`, `device_1.log`: Training logs for each device

## Usage

### Training
1. Ensure training data is in `data/e2e/train.jsonl`
2. Run the training pipeline:
   ```bash
   ./run_pipeline.sh
   ```
   This will start both devices and begin distributed training.

### Inference
Generate text using the trained model:
```bash
python infer_2device.py --prompt "Your text here" --max_length 50 --temperature 1.0 --top_k 50
```

## Architecture

The system uses pipeline parallelism to distribute GPT-2 across devices:
- **Device 0**: Handles embeddings + layers 0-5
- **Device 1**: Handles layers 6-11 + language model head

Communication happens via WebSockets:
- Hidden states flow forward from Device 0 to Device 1
- Gradients flow backward from Device 1 to Device 0
- Each device updates only its LoRA parameters

## Dependencies

- PyTorch
- Transformers
- PEFT (for LoRA)
- WebSockets
- NumPy
- JSON

## Configuration

Key parameters can be modified in `ws_client.py`:
- `lora_dim`: LoRA rank (default: 8)
- `lr`: Learning rate (default: 1e-5)
- `train_batch_size`: Batch size (default: 1)
- `seq_len`: Sequence length (default: 128)
- `max_step`: Maximum training steps (default: 2000)
