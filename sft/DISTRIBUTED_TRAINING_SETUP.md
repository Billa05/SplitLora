# Distributed Training Setup Guide (2 Laptops)

This guide explains how to run FSDP training across 2 laptops on the same network.

## Prerequisites

1. Both laptops must be on the same network
2. Both laptops need the same Python environment with all dependencies installed
3. Both laptops need the same code/dataset
4. Firewall must allow communication between the machines

## Setup Steps

### Step 1: Find the Main Laptop's IP Address

On **Laptop 1** (main/master node), find its IP address:

```bash
# On Linux
hostname -I | awk '{print $1}'

# Or
ip addr show | grep "inet " | grep -v 127.0.0.1
```

Let's say the IP is `192.168.1.100`

### Step 2: Update Config Files

**On Laptop 1 (main node):**

Edit `configs/fsdp_config.yaml`:
```yaml
machine_rank: 0
main_process_ip: 192.168.1.100  # Your actual IP
main_process_port: 29500
num_machines: 2
num_processes: 2
```

**On Laptop 2 (worker node):**

Edit `configs/fsdp_config_worker.yaml`:
```yaml
machine_rank: 1
main_process_ip: 192.168.1.100  # Same as Laptop 1's IP
main_process_port: 29500
num_machines: 2
num_processes: 2
```

### Step 3: Open Firewall Port

On **Laptop 1**, allow incoming connections on port 29500:

```bash
# Ubuntu/Debian
sudo ufw allow 29500

# Or if using firewalld (Fedora/RHEL)
sudo firewall-cmd --zone=public --add-port=29500/tcp --permanent
sudo firewall-cmd --reload
```

### Step 4: Test Network Connectivity

From **Laptop 2**, verify you can reach Laptop 1:

```bash
ping 192.168.1.100
nc -zv 192.168.1.100 29500  # After training starts
```

### Step 5: Start Training

**IMPORTANT:** Start in this order:

1. **First, on Laptop 1 (main node):**
```bash
cd /home/biresh/Downloads/coding/prism/peft/examples/sft
bash run_peft_fsdp.sh
```

2. **Then, on Laptop 2 (worker node):**
```bash
cd <path-to-same-directory>
accelerate launch --config_file "configs/fsdp_config_worker.yaml" train.py \
--seed 100 \
--model_name_or_path "gpt2" \
--dataset_name "timdettmers/openassistant-guanaco" \
--chat_template_format "none" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_length 256 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub False \
--bf16 False \
--packing False \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "gpt2-sft-lora-fsdp" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "text" \
--use_flash_attn False \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization False \
--use_8bit_quantization True
```

### Step 6: Monitor Training

Both laptops will show training progress. The main node (Laptop 1) will coordinate and save checkpoints.

## Troubleshooting

### Connection Timeout
- Verify both laptops can ping each other
- Check firewall settings on both machines
- Ensure port 29500 is not being used by another process

### Training Hangs
- Make sure you start Laptop 1 first
- Wait for Laptop 1 to initialize before starting Laptop 2
- Check that both machines have the same code and dataset

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_length` to 128 or 192
- Enable `fsdp_offload_params: true` in the config (but this slows training)

### Different Training Speeds
- This is normal if laptops have different GPU speeds
- The slower GPU will be the bottleneck

## Benefits of This Setup

✅ Combine GPU memory from both laptops (can train larger models)
✅ Faster training than single laptop
✅ Model sharding across devices reduces per-device memory usage

## Notes

- Both laptops must stay connected to the same network during training
- If connection drops, training will fail and need to restart
- Checkpoints are saved on the main node (Laptop 1)
- For best results, use laptops with similar GPU performance
