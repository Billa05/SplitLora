# 🚀 Distributed Split Learning - Ready to Use!

You now have a complete distributed split learning implementation that allows training GPT-2 models across two devices using WebSocket communication.

## 📁 New Files Added

| File | Purpose |
|------|---------|
| `ws_client.py` | **Main script** - runs as client or server |
| `QUICKSTART.md` | Quick reference guide |
| `DISTRIBUTED_TRAINING.md` | Comprehensive documentation |
| `IMPLEMENTATION_SUMMARY.md` | Technical implementation details |
| `check_environment.py` | Environment validation script |
| `example_configs.sh` | Example configurations for different scenarios |
| `run_distributed_training.sh` | Helper script with commands |

## ⚡ Quick Start (2 Commands)

### 1️⃣ Terminal 1 (Start Server)
```bash
python ws_client.py --device_id 1
```

### 2️⃣ Terminal 2 (Start Client)
```bash
python ws_client.py --device_id 0 \
  --train_data ../data/e2e/train.jsonl \
  --valid_data ../data/e2e/valid.jsonl
```

**That's it!** Training will begin automatically once both are running.

## 🔍 Before You Start

**Check your environment:**
```bash
python check_environment.py
```

**Install dependencies if needed:**
```bash
pip install websockets
```

## 📚 Documentation

- **New to this?** → Read `QUICKSTART.md`
- **Need details?** → Read `DISTRIBUTED_TRAINING.md`
- **Technical info?** → Read `IMPLEMENTATION_SUMMARY.md`
- **Examples?** → Run `./example_configs.sh`

## 🎯 Usage Patterns

### Same Machine, Two GPUs
```bash
# Terminal 1
python ws_client.py --device_id 1 --server_device cuda:1

# Terminal 2
python ws_client.py --device_id 0 --client_device cuda:0
```

### Same Machine, Single GPU (Testing)
```bash
# Terminal 1
python ws_client.py --device_id 1 --server_device cuda:0

# Terminal 2
python ws_client.py --device_id 0 --client_device cuda:0
```

### Two Different Machines
```bash
# Machine 1 (Server at 192.168.1.100)
python ws_client.py --device_id 1 --server_host 0.0.0.0

# Machine 2 (Client)
python ws_client.py --device_id 0 --server_host 192.168.1.100
```

### Memory Constrained (Small GPU)
```bash
# Terminal 1
python ws_client.py --device_id 1 --server_device cuda:0

# Terminal 2
python ws_client.py --device_id 0 --client_device cuda:0 \
  --train_batch_size 1 --seq_len 128 --grad_acc 4
```

## 💡 Key Parameters

| Parameter | Description | Default | Recommendations |
|-----------|-------------|---------|-----------------|
| `--device_id` | 0=client, 1=server | **Required** | Start server first |
| `--client_layers` | Split point | 6 | Balance compute load |
| `--train_batch_size` | Batch size | 1 | Keep at 1 for small GPU |
| `--seq_len` | Sequence length | 128 | 64-512 depending on memory |
| `--grad_acc` | Gradient accumulation | 4 | 4-8 for small batches |
| `--epochs` | Training epochs | 1 | 1-5 typical |
| `--server_host` | Server address | localhost | Use 0.0.0.0 for remote |
| `--server_port` | WebSocket port | 8765 | Change if port busy |

## 🎬 What Happens During Training

1. **Server starts** → Waits for connection on port 8765
2. **Client starts** → Connects to server
3. **For each batch:**
   - Client processes embeddings + early layers
   - Client sends hidden states to server
   - Server processes late layers + computes loss
   - Server sends loss back to client
   - Server computes and sends gradients
   - Both update their LoRA parameters
4. **After epoch:**
   - Validation on validation set
   - Save checkpoints if improved
5. **Training completes** → Save final checkpoint

## 📊 Expected Output

### Server Terminal:
```
[Server] Initializing on device: cuda:0
[Server] Loading pretrained weights...
[Server] Ready and waiting for client connection on port 8765
[Server] Listening on ws://localhost:8765
[Server] Client connected!
```

### Client Terminal:
```
[Client] Initializing on device: cuda:0
[Client] Loading datasets...
[Client] Connected to server!
[epoch 1] step 50 | loss 3.25 | ppl 25.79 | time 45.2s
[epoch 1] train_loss=2.88 ppl=17.76
[epoch 1] val_loss=2.91 val_ppl=18.40
[Client] Saved LoRA adapters to ./outputs/lora_adapters_final.pt
[Client] Training completed!
```

## 🐛 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| "Connection refused" | Start server (device_id=1) first |
| "CUDA out of memory" | Reduce `--train_batch_size` and `--seq_len` |
| "Port already in use" | Use different port: `--server_port 8766` |
| Missing data files | Provide explicit paths: `--train_data /path/to/data` |
| Slow training | Use faster network or same machine |

## 🎓 Understanding the Architecture

```
┌─────────────────────────┬─────────────────────────┐
│   Device 0 (Client)     │   Device 1 (Server)     │
├─────────────────────────┼─────────────────────────┤
│ • Has training data     │ • Waits for connection  │
│ • Token embeddings      │                         │
│ • Layers 0-5            │ • Layers 6-11           │
│ • Sends hidden states ──→ Receives hidden states │
│                         │ • LM Head               │
│                         │ • Computes loss         │
│                         │ • Backpropagates        │
│ ← Receives gradients ─── Sends gradients         │
│ • Updates LoRA params   │ • Updates LoRA params   │
│ • Saves checkpoints     │                         │
└─────────────────────────┴─────────────────────────┘
```

## 🔬 Testing Your Setup

1. **Check environment:**
   ```bash
   python check_environment.py
   ```

2. **Quick test (100 steps):**
   ```bash
   # Terminal 1
   python ws_client.py --device_id 1
   
   # Terminal 2
   python ws_client.py --device_id 0 --max_train_steps 100
   ```

3. **Verify output:**
   ```bash
   ls -lh outputs/
   # Should see: lora_adapters_final.pt
   ```

## 📦 Output Files

After training, you'll find in `./outputs/`:

- `lora_adapters_epoch{N}.pt` - Best model from each epoch
- `lora_adapters_final.pt` - Final trained model
- `log.txt` - Training logs

Each checkpoint contains:
- Client LoRA parameters
- Server LoRA parameters  
- Model configuration

## 🚀 Next Steps

1. **Train your model:**
   ```bash
   # See QUICKSTART.md for commands
   ```

2. **Test trained model:**
   ```bash
   python infer_with_lora.py \
     --lora_adapters ./outputs/lora_adapters_final.pt \
     --init_checkpoint ../pretrained_checkpoints/gpt2-pytorch_model.bin
   ```

3. **Experiment:**
   - Try different `--client_layers` (3, 6, 9)
   - Adjust `--lora_dim` (4, 8, 16)
   - Train longer with `--epochs 5`

## 🤝 Comparison with Original

| Feature | Original Script | New Distributed Script |
|---------|----------------|------------------------|
| File | `train_two_device_split.py` | `ws_client.py` |
| Devices | Same machine only | Same or different machines |
| Setup | Single command | Two terminals |
| Communication | In-memory | WebSocket |
| Use Case | Testing | Production |
| Privacy | Low | High (only activations shared) |

## 📖 Help & Documentation

- **Quick help:** `python ws_client.py --help`
- **Examples:** `./example_configs.sh`
- **Full docs:** See `DISTRIBUTED_TRAINING.md`
- **Troubleshooting:** Check `QUICKSTART.md`

## ✅ Checklist

Before starting training:
- [ ] `websockets` installed (`pip install websockets`)
- [ ] Environment checked (`python check_environment.py`)
- [ ] Data files available (or paths specified)
- [ ] GPT-2 checkpoint available (or path specified)
- [ ] Ports 8765 available (or different port specified)
- [ ] Two terminals ready

---

**Ready to train?** Open two terminals and let's go! 🎉

```bash
# Terminal 1
python ws_client.py --device_id 1

# Terminal 2  
python ws_client.py --device_id 0
```

For questions, see the documentation files or check the troubleshooting sections.

Happy distributed training! 🚂🔥
