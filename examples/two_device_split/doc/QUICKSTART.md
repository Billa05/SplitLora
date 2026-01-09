# Quick Start Guide: Distributed Split Learning

## Prerequisites

```bash
# Install websockets
pip install websockets

# Verify environment
python check_environment.py
```

## Running Training

### Step 1: Start Server (Terminal 1)

```bash
python ws_client.py --device_id 1
```

This will:
- Start as **Server (Device 1)**
- Listen on port 8765
- Wait for client connection

### Step 2: Start Client (Terminal 2)

```bash
python ws_client.py --device_id 0 \
  --train_data ../data/e2e/train.jsonl \
  --valid_data ../data/e2e/valid.jsonl
```

This will:
- Start as **Client (Device 0)**
- Connect to server
- Load data and begin training

## Expected Output

### Terminal 1 (Server):
```
================================================================================
Starting as SERVER (Device 1)
================================================================================
[Server] Initializing on device: cuda:0
[Server] Loading pretrained weights from ../pretrained_checkpoints/gpt2-pytorch_model.bin
[Server] Ready and waiting for client connection on port 8765
[Server] Listening on ws://localhost:8765
[Server] Client connected!
```

### Terminal 2 (Client):
```
================================================================================
Starting as CLIENT (Device 0)
================================================================================
[Client] Initializing on device: cuda:0
[Client] Loading pretrained weights from ../pretrained_checkpoints/gpt2-pytorch_model.bin
[Client] Loading datasets...
[Client] Calculated max_step: 1000
[Client] Attempting to connect to server...
[Client] Connected to server!

[Client] Starting epoch 1/1
[epoch 1] step 50 | loss 3.2541 | ppl 25.89 | time 45.2s
[epoch 1] step 100 | loss 2.9812 | ppl 19.72 | time 43.8s
...
[epoch 1] train_loss=2.8765 ppl=17.76
[epoch 1] val_loss=2.9123 val_ppl=18.40
[Client] Saved LoRA adapters to ./outputs/lora_adapters_epoch1.pt
[Client] Saved LoRA adapters to ./outputs/lora_adapters_final.pt
[Client] Training completed!
```

## Common Configurations

### Minimal (for testing):
```bash
# Server
python ws_client.py --device_id 1

# Client
python ws_client.py --device_id 0 --train_batch_size 1 --seq_len 64 --epochs 1
```

### Recommended (small GPU):
```bash
# Server
python ws_client.py --device_id 1 --server_device cuda:0

# Client
python ws_client.py --device_id 0 --client_device cuda:0 \
  --train_batch_size 1 --seq_len 128 --grad_acc 4 --epochs 1
```

### Multiple GPUs:
```bash
# Server on GPU 1
python ws_client.py --device_id 1 --server_device cuda:1

# Client on GPU 0
python ws_client.py --device_id 0 --client_device cuda:0 \
  --train_batch_size 2 --seq_len 256 --grad_acc 2 --epochs 3
```

### Remote Training:
```bash
# Server (Machine 1 with IP 192.168.1.100)
python ws_client.py --device_id 1 --server_host 0.0.0.0

# Client (Machine 2)
python ws_client.py --device_id 0 --server_host 192.168.1.100
```

## Troubleshooting

### Problem: "Connection refused"
**Solution:** Start server (device_id=1) first, then client (device_id=0)

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size and sequence length:
```bash
--train_batch_size 1 --seq_len 64 --grad_acc 8
```

### Problem: "Port already in use"
**Solution:** Use a different port:
```bash
--server_port 8766
```

### Problem: Missing data files
**Solution:** Provide explicit paths:
```bash
--train_data /path/to/train.jsonl --valid_data /path/to/valid.jsonl
```

## Next Steps

- See `DISTRIBUTED_TRAINING.md` for detailed documentation
- Test trained model with `infer_with_lora.py`
- Adjust `--client_layers` to balance compute load

## Help

```bash
python ws_client.py --help
```
