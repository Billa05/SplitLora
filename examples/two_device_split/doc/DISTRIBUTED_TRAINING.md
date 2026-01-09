# Distributed Split Learning with WebSockets

This setup allows you to train a split GPT-2 model across two devices using WebSocket communication.

## Quick Start

### Step 1: Install Dependencies

```bash
pip install websockets
```

### Step 2: Start Server (Terminal 1)

```bash
python ws_client.py --device_id 1 \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --model_card gpt2.sm \
  --client_layers 6 \
  --server_device cuda:0
```

**What this does:**
- Runs as **Server (Device 1)**
- Loads the server partition of GPT-2 (layers 6-11)
- Waits for client connection on port 8765
- Processes forward/backward passes when requested

### Step 3: Start Client (Terminal 2)

```bash
python ws_client.py --device_id 0 \
  --train_data ../data/e2e/train.jsonl \
  --valid_data ../data/e2e/valid.jsonl \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --model_card gpt2.sm \
  --client_layers 6 \
  --client_device cuda:0 \
  --train_batch_size 1 \
  --seq_len 128 \
  --grad_acc 4 \
  --epochs 4 \
  --output_dir ./outputs
```

**What this does:**
- Runs as **Client (Device 0)**
- Loads training data
- Loads the client partition of GPT-2 (layers 0-5 + embeddings)
- Connects to server
- Orchestrates the training loop

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
├──────────────────────┬──────────────────────────────────────┤
│   Device 0 (Client)  │         Device 1 (Server)            │
├──────────────────────┼──────────────────────────────────────┤
│  1. Load Data        │  1. Wait for connection              │
│  2. Embeddings       │                                       │
│  3. Layers 0-5       │                                       │
│  4. Send hidden ───> │  5. Receive hidden                   │
│     states          │  6. Layers 6-11                       │
│                      │  7. LM Head                           │
│                      │  8. Compute Loss                      │
│                      │  9. Backward Pass                     │
│ 10. Receive grad <─── 10. Send gradients                    │
│ 11. Backward Pass    │                                       │
│ 12. Update Params    │ 13. Update Params                    │
└──────────────────────┴──────────────────────────────────────┘
```

### Communication Flow

1. **Forward Pass:**
   - Client: Input → Embeddings → Layers 0-5 → Hidden States
   - Client sends: `hidden_states`, `presents`, `labels`, `mask`
   - Server: Hidden States → Layers 6-11 → LM Head → Loss
   - Server sends back: `loss`

2. **Backward Pass:**
   - Client requests backward pass
   - Server: Computes gradients, backpropagates
   - Server sends: `gradients` w.r.t. hidden states
   - Client: Backpropagates through layers 0-5

3. **Optimizer Step:**
   - Client sends optimizer step signal
   - Both devices update their LoRA parameters
   - Schedulers step (if configured)

## Configuration Options

### Device Selection

- `--device_id 0`: Client (has data, orchestrates training)
- `--device_id 1`: Server (processes later layers)

### Network Configuration

- `--server_host localhost`: Server hostname (use IP for remote)
- `--server_port 8765`: WebSocket port

### Model Configuration

- `--model_card`: `gpt2.sm` (12 layers), `gpt2.md` (24 layers), `gpt2.lg` (36 layers)
- `--client_layers`: Number of layers on client (e.g., 6 means client gets 0-5, server gets 6-11)

### Training Configuration

- `--train_batch_size`: Batch size (set to 1 for small GPUs)
- `--seq_len`: Sequence length (128 or 256 for small GPUs)
- `--grad_acc`: Gradient accumulation steps (4 or 8 recommended)
- `--epochs`: Number of training epochs

### LoRA Configuration

- `--lora_dim`: LoRA rank (default: 4)
- `--lora_alpha`: LoRA scaling (default: 32)
- `--lora_dropout`: Dropout for LoRA layers (default: 0.1)

## Different Device Configurations

### 1. Two GPUs on Same Machine

**Terminal 1:**
```bash
python ws_client.py --device_id 1 --server_device cuda:1
```

**Terminal 2:**
```bash
python ws_client.py --device_id 0 --client_device cuda:0
```

### 2. GPU + CPU

**Terminal 1 (Server on GPU):**
```bash
python ws_client.py --device_id 1 --server_device cuda:0
```

**Terminal 2 (Client on CPU):**
```bash
python ws_client.py --device_id 0 --client_device cpu
```

### 3. Two Different Machines

**Machine 1 (Server):**
```bash
python ws_client.py --device_id 1 \
  --server_host 0.0.0.0 \
  --server_device cuda:0
```

**Machine 2 (Client):**
```bash
python ws_client.py --device_id 0 \
  --server_host <machine1_ip> \
  --client_device cuda:0 \
  --train_data ./data/e2e/train.jsonl \
  --valid_data ./data/e2e/valid.jsonl
```

## Outputs

The client will save LoRA adapters to `--output_dir` (default: `./outputs/`):

- `lora_adapters_epoch1.pt`: Best model from epoch 1
- `lora_adapters_final.pt`: Final model after all epochs

Each checkpoint contains:
- Client LoRA parameters
- Server LoRA parameters
- Model configuration

## Troubleshooting

### Connection Issues

**Problem:** Client can't connect to server

**Solutions:**
1. Ensure server is started first
2. Check firewall settings (open port 8765)
3. Verify `--server_host` and `--server_port` match
4. For remote: use actual IP, not localhost

### Out of Memory

**Problem:** CUDA out of memory

**Solutions:**
1. Reduce `--train_batch_size` to 1
2. Reduce `--seq_len` to 128 or 64
3. Increase `--grad_acc` to maintain effective batch size
4. Use CPU for one partition: `--client_device cpu`

### Port Already in Use

**Problem:** Port 8765 already in use

**Solution:**
```bash
# Use a different port
python ws_client.py --device_id 1 --server_port 8766
python ws_client.py --device_id 0 --server_port 8766
```

### Slow Training

**Problem:** Training is very slow

**Reasons:**
1. Network latency (if using remote machines)
2. Small batch size with gradient accumulation
3. CPU bottleneck

**Solutions:**
1. Use machines on same network/data center
2. Increase batch size if memory allows
3. Use GPU for both partitions

## Monitoring

### Server Terminal
- Shows connection status
- Processes forward/backward requests
- No progress bars (stateless)

### Client Terminal
- Shows epoch progress
- Displays loss and perplexity
- Shows training/validation metrics
- Saves checkpoints

## Advanced Usage

### Custom Split Points

For GPT-2 Small (12 layers):
- `--client_layers 3`: Client (0-2), Server (3-11)
- `--client_layers 6`: Client (0-5), Server (6-11) [default]
- `--client_layers 9`: Client (0-8), Server (9-11)

### Multiple Epochs with Checkpointing

```bash
python ws_client.py --device_id 0 \
  --epochs 3 \
  --output_dir ./outputs_3epochs
```

The best model from each epoch will be saved automatically.

## Testing the Trained Model

After training, test your model with the inference script:

```bash
python infer_with_lora.py \
  --lora_adapters ./outputs/lora_adapters_final.pt \
  --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin \
  --model_card gpt2.sm \
  --client_layers 6
```

## Performance Tips

1. **Network:** Use 10 Gbps Ethernet or faster for remote training
2. **Memory:** Client needs more memory (embeddings + data)
3. **Compute:** Server does more compute (most layers + loss)
4. **Balance:** Adjust `--client_layers` to balance compute load

## Comparison with Single-Device Training

### Single Device (Original)
```bash
python train_two_device_split.py \
  --client_device cuda:0 \
  --server_device cuda:0
```

### Distributed (New)
```bash
# Terminal 1
python ws_client.py --device_id 1 --server_device cuda:0

# Terminal 2  
python ws_client.py --device_id 0 --client_device cuda:0
```

**Benefits of Distributed:**
- Can use multiple GPUs/machines
- Better memory distribution
- Enables privacy-preserving learning
- More flexible deployment

**Trade-offs:**
- Network communication overhead
- Slightly more complex setup
- Requires both devices to be available
