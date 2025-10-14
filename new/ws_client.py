import asyncio
import io
import json
import logging
import argparse
from typing import Dict

import torch
import loralib as lora
import websockets
from torch.utils.data import DataLoader
from transformers import GPT2Config

from data_utils import FT_Dataset
from splitmodel import GPT2SplitPart, get_lora_config
from optimizer import create_adam_optimizer_from_args
from ws_utils import send_json, recv_json, send_bytes, recv_bytes, state_dict_to_bytes


class ClientArgs:
    random_seed: int = 42
    fp16: bool = False
    lora_dim: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    grad_acc: int = 1
    train_batch_size: int = 1
    seq_len: int = 128
    model_card: str = "gpt2"
    max_step: int = 2000
    device_id: int = 0
    num_devices: int = 2  # Changed to 2 devices
    
    # Optimizer attributes
    lr: float = 0.00001
    weight_decay: float = 0.01
    correct_bias: bool = False
    adam_epislon: float = 1e-6
    no_decay_bias: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    scheduler: str = "linear"
    warmup_step: int = 0


CONFIG = {
    "valid_data_path": "./data/e2e/train.jsonl",
}


def build_config(model_card: str, args: ClientArgs) -> GPT2Config:
    if model_card == "gpt2.sm":
        return GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    if model_card == "gpt2.md":
        return GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    return GPT2Config(
        n_embd=1280,
        n_layer=36,
        n_head=20,
        lora_attn_dim=args.lora_dim,
        lora_attn_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


async def client_main(device_id):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger = logging.getLogger(f"ws_pipeline_{device_id}")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ClientArgs()
    args.device_id = device_id
    args.num_devices = 2  # 2 devices total

    base_port = 8765
    listen_port = base_port + device_id if device_id > 0 else None
    next_uri = f"ws://127.0.0.1:{base_port + device_id + 1}" if device_id < args.num_devices - 1 else None

    # Split 12 layers across 2 devices: 0-5 and 6-11
    layers_per_part = 12 // args.num_devices
    start_layer = device_id * layers_per_part
    end_layer = (device_id + 1) * layers_per_part
    has_embeddings = device_id == 0
    has_lm_head = device_id == args.num_devices - 1

    config = GPT2Config()
    lora_config = get_lora_config(r=args.lora_dim, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model_part = GPT2SplitPart(config, start_layer, end_layer, has_embeddings, has_lm_head, lora_config).to(device)
    optimizer = create_adam_optimizer_from_args(model_part, args)

    if device_id == 0:
        train_data = FT_Dataset(CONFIG["valid_data_path"], args.train_batch_size, args.seq_len)
        train_loader = DataLoader(train_data, batch_size=args.train_batch_size, drop_last=True)
    else:
        train_loader = None

    # Start server if listening
    prev_queue = None
    server = None
    next_ws = None
    training_done = asyncio.Event()  # Signal for when training completes
    
    if listen_port:
        prev_queue = asyncio.Queue()
        async def handle_prev(ws):
            nonlocal next_ws
            logger.info(f"Accepted connection from previous device")
            try:
                while True:
                    try:
                        # Receive hidden states, labels, and mask from previous device
                        hidden_bytes = await recv_bytes(ws)
                        hidden_states = torch.load(io.BytesIO(hidden_bytes), map_location=device).to(device).requires_grad_(True)
                        labels_bytes = await recv_bytes(ws)
                        lm_labels = torch.load(io.BytesIO(labels_bytes), map_location=device)
                        mask_bytes = await recv_bytes(ws)
                        lm_mask = torch.load(io.BytesIO(mask_bytes), map_location=device)
                        
                        # Process through this device's layers
                        if has_lm_head:
                            # Last device: compute loss
                            logits, loss = model_part(hidden_states=hidden_states, labels=lm_labels)
                            loss = loss.mean()
                            loss.backward()
                            logger.info(f"Device {device_id}: Loss = {loss.item():.4f}")
                            
                            # Send gradient back to previous device
                            buf = io.BytesIO()
                            torch.save(hidden_states.grad.detach().cpu(), buf)
                            await send_bytes(ws, buf.getvalue())
                            
                            # Update local LoRA parameters
                            optimizer.step()
                            optimizer.zero_grad()
                        else:
                            # Middle device: forward to next device
                            hidden_states, _ = model_part(hidden_states=hidden_states)
                            hidden_states.retain_grad()
                            
                            # Send to next device
                            buf = io.BytesIO(); torch.save(hidden_states.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                            buf = io.BytesIO(); torch.save(lm_labels.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                            buf = io.BytesIO(); torch.save(lm_mask.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                            
                            # Receive gradient from next device
                            grad_bytes = await recv_bytes(next_ws)
                            grad = torch.load(io.BytesIO(grad_bytes), map_location=device)
                            hidden_states.backward(grad)
                            
                            # Update local LoRA parameters
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # Send gradient back to previous device
                            buf = io.BytesIO()
                            torch.save(hidden_states.grad.detach().cpu(), buf)
                            await send_bytes(ws, buf.getvalue())
                            
                            logger.info(f"Device {device_id}: Processed and forwarded")
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("Connection closed, ending handle_prev")
                        break
                    except Exception as e:
                        logger.error(f"Error in handle_prev: {e}")
                        break
            finally:
                # Signal that training is done when connection closes
                training_done.set()
                
        server = await websockets.serve(handle_prev, "0.0.0.0", listen_port, max_size=None, max_queue=None)
        logger.info(f"Listening on port {listen_port}")

    # Connect to next device (must happen after server starts for middle devices)
    if next_uri:
        logger.info(f"Attempting to connect to {next_uri}")
        next_ws = await websockets.connect(next_uri, max_size=None, max_queue=None)
        logger.info(f"Connected to {next_uri}")

    # Training loop (only for Device 0 - the first device with data)
    train_step = 0
    if train_loader:
        for batch in train_loader:
            if train_step >= args.max_step:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _input = batch["input"]
            _target = batch["target"]
            _mask = batch["mask"]
            
            # Forward pass through Device 0's layers
            hidden_states, _ = model_part(input_ids=_input)
            hidden_states.retain_grad()
            
            # Send hidden states, target, mask to next device
            buf = io.BytesIO(); torch.save(hidden_states.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
            buf = io.BytesIO(); torch.save(_target.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
            buf = io.BytesIO(); torch.save(_mask.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
            
            # Receive gradient from next device
            grad_bytes = await recv_bytes(next_ws)
            grad = torch.load(io.BytesIO(grad_bytes), map_location=device)
            hidden_states.backward(grad)
            
            # Update local LoRA parameters
            optimizer.step()
            optimizer.zero_grad()
            
            train_step += 1
            logger.info(f"Device 0: Step {train_step} processed")
        
        # Training complete - close connection to signal downstream devices
        logger.info(f"Device 0: Training complete, closing connections")
        if next_ws:
            await next_ws.close()
    else:
        # For other devices, wait until training is done (connection closes)
        logger.info(f"Device {device_id}: Waiting for incoming connections...")
        await training_done.wait()
        logger.info(f"Device {device_id}: Training complete signal received")

    # Save LoRA adapter
    logger.info(f"Device {device_id}: Saving LoRA adapter to ./lora_device_{device_id}.pth")
    torch.save(model_part.state_dict(), f"./lora_device_{device_id}.pth")

    # Cleanup
    if server:
        server.close()
        await server.wait_closed()
    if next_ws:
        try:
            await next_ws.close()
        except Exception:
            pass  # Already closed
    
    logger.info(f"Device {device_id}: Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, required=True, help="Device ID (0 or 1)")
    args_parsed = parser.parse_args()
    if args_parsed.device_id not in [0, 1]:
        raise ValueError("Invalid device_id, must be 0 or 1")
    asyncio.run(client_main(args_parsed.device_id))


