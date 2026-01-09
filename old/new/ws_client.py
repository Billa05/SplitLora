import asyncio
import io
import json
import logging
import argparse
from typing import Dict

import torch
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
    lora_dim: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    grad_acc: int = 1
    train_batch_size: int = 4
    seq_len: int = 128
    model_card: str = "gpt2"
    num_epochs: int = 4  # Increased from 6 to 10 epochs for better convergence
    device_id: int = 0
    num_devices: int = 2 
    
    # Optimizer attributes
    lr: float = 0.0003  # Increased to 3e-4 for faster convergence with LoRA
    weight_decay: float = 0.01
    correct_bias: bool = False
    adam_epislon: float = 1e-6
    no_decay_bias: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999  # Changed from 0.98 to 0.999 (more standard)
    scheduler: str = "linear"
    warmup_step: int = 500  # Warmup for stability


CONFIG = {
    "train_data_path": "./data/e2e/train.jsonl",
}

async def client_main(device_id):
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(f"Device_{device_id}")
    
    logger.info("="*70)
    logger.info(f"INITIALIZING DEVICE {device_id}")
    logger.info("="*70)
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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

    logger.info(f"Configuration:")
    logger.info(f"  - Layers: {start_layer} to {end_layer-1} (total: {end_layer-start_layer})")
    logger.info(f"  - Has embeddings: {has_embeddings}")
    logger.info(f"  - Has LM head: {has_lm_head}")
    logger.info(f"  - LoRA rank: {args.lora_dim}, alpha: {args.lora_alpha}")
    logger.info(f"  - Batch size: {args.train_batch_size}, seq_len: {args.seq_len}")
    logger.info(f"  - Number of epochs: {args.num_epochs}")
    logger.info(f"  - Learning rate: {args.lr}")
    if listen_port:
        logger.info(f"  - Listen port: {listen_port}")
    if next_uri:
        logger.info(f"  - Next device URI: {next_uri}")

    logger.info("Loading model components...")
    config = GPT2Config()
    lora_config = get_lora_config(r=args.lora_dim, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model_part = GPT2SplitPart(config, start_layer, end_layer, has_embeddings, has_lm_head, lora_config).to(device)
    
    logger.info("Creating optimizer...")
    optimizer = create_adam_optimizer_from_args(model_part, args)
    
    # Count parameters
    total_params = sum(p.numel() for p in model_part.parameters())
    trainable_params = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create learning rate scheduler
    from optimizer import create_optimizer_scheduler
    if device_id == 0:
        train_data = FT_Dataset(CONFIG["train_data_path"], args.train_batch_size, args.seq_len)
        train_loader = DataLoader(train_data, batch_size=args.train_batch_size, drop_last=True)
        num_batches = len(train_loader)
        args.max_step = args.num_epochs * num_batches
        logger.info(f"Total training steps: {args.max_step}")
    else:
        args.max_step = 10000  # Dummy value for non-first devices
    
    scheduler = create_optimizer_scheduler(optimizer, args)
    if scheduler:
        logger.info(f"✓ Learning rate scheduler created: {args.scheduler}")
    else:
        logger.info("No learning rate scheduler (using constant LR)")

    if device_id != 0:
        train_loader = None
        logger.info("No training data (waiting for upstream device)")

    # Start server if listening
    prev_queue = None
    server = None
    next_ws = None
    training_done = asyncio.Event()  # Signal for when training completes
    
    if listen_port:
        logger.info(f"Starting WebSocket server on port {listen_port}...")
        prev_queue = asyncio.Queue()
        async def handle_prev(ws):
            nonlocal next_ws
            logger.info(f"✓ Connection accepted from previous device (Device {device_id-1})")
            step_count = 0
            try:
                while True:
                    try:
                        step_count += 1
                        # Receive hidden states, labels, and mask from previous device
                        logger.debug(f"[Step {step_count}] Receiving hidden states from Device {device_id-1}...")
                        hidden_bytes = await recv_bytes(ws)
                        hidden_states = torch.load(io.BytesIO(hidden_bytes), map_location=device).to(device).requires_grad_(True)
                        logger.debug(f"[Step {step_count}] Hidden states shape: {hidden_states.shape}")
                        
                        labels_bytes = await recv_bytes(ws)
                        lm_labels = torch.load(io.BytesIO(labels_bytes), map_location=device)
                        
                        mask_bytes = await recv_bytes(ws)
                        lm_mask = torch.load(io.BytesIO(mask_bytes), map_location=device)
                        
                        # Process through this device's layers
                        if has_lm_head:
                            # Last device: compute loss with mask
                            logger.debug(f"[Step {step_count}] Forward pass through layers {start_layer}-{end_layer-1}...")
                            logits, loss = model_part(hidden_states=hidden_states, labels=lm_labels, mask=lm_mask)
                            logger.info(f"[Step {step_count}] Loss = {loss.item():.4f}")
                            
                            # Backward pass
                            logger.debug(f"[Step {step_count}] Computing gradients...")
                            loss.backward()
                            
                            # Send gradient back to previous device
                            logger.debug(f"[Step {step_count}] Sending gradients to Device {device_id-1}...")
                            buf = io.BytesIO()
                            torch.save(hidden_states.grad.detach().cpu(), buf)
                            await send_bytes(ws, buf.getvalue())
                            
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(model_part.parameters(), max_norm=1.0)
                            
                            # Update local LoRA parameters
                            logger.debug(f"[Step {step_count}] Updating LoRA parameters...")
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # Step scheduler if available
                            if scheduler:
                                scheduler.step()
                            
                            if step_count % 100 == 0:
                                logger.info(f"Processed {step_count} steps so far")
                        else:
                            # Middle device: forward to next device
                            logger.debug(f"[Step {step_count}] Forward pass through layers {start_layer}-{end_layer-1}...")
                            hidden_states, _ = model_part(hidden_states=hidden_states)
                            hidden_states.retain_grad()
                            
                            # Send to next device
                            logger.debug(f"[Step {step_count}] Sending to Device {device_id+1}...")
                            buf = io.BytesIO(); torch.save(hidden_states.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                            buf = io.BytesIO(); torch.save(lm_labels.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                            buf = io.BytesIO(); torch.save(lm_mask.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                            
                            # Receive gradient from next device
                            logger.debug(f"[Step {step_count}] Receiving gradients from Device {device_id+1}...")
                            grad_bytes = await recv_bytes(next_ws)
                            grad = torch.load(io.BytesIO(grad_bytes), map_location=device)
                            hidden_states.backward(grad)
                            
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(model_part.parameters(), max_norm=1.0)
                            
                            # Update local LoRA parameters
                            logger.debug(f"[Step {step_count}] Updating LoRA parameters...")
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # Step scheduler if available
                            if scheduler:
                                scheduler.step()
                            
                            # Send gradient back to previous device
                            logger.debug(f"[Step {step_count}] Sending gradients to Device {device_id-1}...")
                            buf = io.BytesIO()
                            torch.save(hidden_states.grad.detach().cpu(), buf)
                            await send_bytes(ws, buf.getvalue())
                            
                            if step_count % 100 == 0:
                                logger.info(f"Processed {step_count} steps so far")
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"Connection closed by Device {device_id-1}, training complete")
                        break
                    except Exception as e:
                        logger.error(f"Error in handle_prev at step {step_count}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        break
            finally:
                # Signal that training is done when connection closes
                logger.info(f"Total steps processed: {step_count}")
                training_done.set()
                
        server = await websockets.serve(handle_prev, "0.0.0.0", listen_port, max_size=None, max_queue=None)
        logger.info(f"✓ WebSocket server listening on 0.0.0.0:{listen_port}")

    # Connect to next device (must happen after server starts for middle devices)
    if next_uri:
        logger.info(f"Connecting to next device at {next_uri}...")
        next_ws = await websockets.connect(next_uri, max_size=None, max_queue=None)
        logger.info(f"✓ Connected to Device {device_id+1}")

    # Training loop (only for Device 0 - the first device with data)
    train_step = 0
    if train_loader:
        logger.info("="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70)
        
        import time
        start_time = time.time()
        loss_history = []
        
        num_batches_per_epoch = len(train_loader)
        total_steps = args.num_epochs * num_batches_per_epoch
        logger.info(f"Training for {args.num_epochs} epochs × {num_batches_per_epoch} batches = {total_steps} total steps")
        
        # Loop through dataset for specified number of epochs
        for epoch in range(1, args.num_epochs + 1):
            epoch_start_time = time.time()
            epoch_start_step = train_step
            logger.info(f"{'='*70}")
            logger.info(f"Starting Epoch {epoch}/{args.num_epochs} (Step {train_step}/{total_steps})")
            logger.info(f"{'='*70}")
            
            for batch_idx, batch in enumerate(train_loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                _input = batch["input"]
                _target = batch["target"]
                _mask = batch["mask"]
                
                step_start = time.time()
                
                # Forward pass through Device 0's layers
                logger.debug(f"[Step {train_step+1}] Forward pass through layers {start_layer}-{end_layer-1}...")
                hidden_states, _ = model_part(input_ids=_input)
                hidden_states.retain_grad()
                
                # Send hidden states, target, mask to next device
                logger.debug(f"[Step {train_step+1}] Sending to Device {device_id+1}...")
                logger.debug(f"[Step {train_step+1}] Hidden states shape: {hidden_states.shape}, Input shape: {_input.shape}")
                buf = io.BytesIO(); torch.save(hidden_states.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                buf = io.BytesIO(); torch.save(_target.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                buf = io.BytesIO(); torch.save(_mask.detach().cpu(), buf); await send_bytes(next_ws, buf.getvalue())
                
                # Receive gradient from next device
                logger.debug(f"[Step {train_step+1}] Receiving gradients from Device {device_id+1}...")
                grad_bytes = await recv_bytes(next_ws)
                grad = torch.load(io.BytesIO(grad_bytes), map_location=device)
                hidden_states.backward(grad)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model_part.parameters(), max_norm=1.0)
                
                # Update local LoRA parameters
                logger.debug(f"[Step {train_step+1}] Updating LoRA parameters...")
                optimizer.step()
                optimizer.zero_grad()
                
                # Step scheduler
                if scheduler:
                    scheduler.step()
                
                train_step += 1
                step_time = time.time() - step_start
                
                # Periodic detailed logging
                if train_step % 100 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = train_step / elapsed
                    remaining_steps = total_steps - train_step
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    eta_mins = eta_seconds / 60
                    
                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    logger.info(f"Epoch {epoch}/{args.num_epochs} | "
                              f"Batch {batch_idx}/{num_batches_per_epoch} | "
                              f"Step {train_step}/{total_steps} | "
                              f"LR: {current_lr:.2e} | "
                              f"Time: {step_time:.3f}s | "
                              f"Speed: {steps_per_sec:.2f} steps/s | "
                              f"ETA: {eta_mins:.1f}min")
                elif train_step % 500 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Progress: Epoch {epoch}/{args.num_epochs}, Step {train_step}/{total_steps}, LR: {current_lr:.2e}")
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            epoch_steps = train_step - epoch_start_step
            logger.info(f"{'='*70}")
            logger.info(f"Epoch {epoch}/{args.num_epochs} Complete | "
                      f"Steps: {epoch_steps} | "
                      f"Time: {epoch_time/60:.2f}min | "
                      f"Speed: {epoch_steps/epoch_time:.2f} steps/s")
            logger.info(f"{'='*70}")
        
        elapsed_total = time.time() - start_time
        logger.info("="*70)
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"Total epochs: {args.num_epochs}")
        logger.info(f"Total steps: {train_step}")
        logger.info(f"Total time: {elapsed_total/60:.2f} minutes")
        logger.info(f"Average speed: {train_step/(elapsed_total):.2f} steps/sec")
        logger.info("="*70)
        
        # Training complete - close connection to signal downstream devices
        logger.info(f"Closing connection to Device {device_id+1}...")
        if next_ws:
            await next_ws.close()
    else:
        # For other devices, wait until training is done (connection closes)
        logger.info("Waiting for training to complete...")
        await training_done.wait()
        logger.info("Training complete signal received")

    # Save LoRA adapter
    logger.info("="*70)
    logger.info(f"Saving LoRA adapter to ./lora_device_{device_id}.pth...")
    torch.save(model_part.state_dict(), f"./lora_device_{device_id}.pth")
    logger.info(f"✓ Model saved successfully")

    # Cleanup
    if server:
        logger.info("Closing WebSocket server...")
        server.close()
        await server.wait_closed()
        logger.info("✓ Server closed")
    if next_ws:
        try:
            await next_ws.close()
            logger.info("✓ Client connection closed")
        except Exception:
            pass  # Already closed
    
    logger.info("="*70)
    logger.info(f"DEVICE {device_id} SHUTDOWN COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, required=True, help="Device ID (0 or 1)")
    args_parsed = parser.parse_args()
    if args_parsed.device_id not in [0, 1]:
        raise ValueError("Invalid device_id, must be 0 or 1")
    asyncio.run(client_main(args_parsed.device_id))


