#!/usr/bin/env python3
"""
Unified script for split learning with WebSocket communication.
Can run as either client (device_id=0) or server (device_id=1).

Usage:
  Terminal 1 (Server): python ws_client.py --device_id 1
  Terminal 2 (Client): python ws_client.py --device_id 0
"""

import argparse
import asyncio
import json
import math
import os
import pickle
import random
import time
import io
import zlib
from typing import Dict, Optional, Tuple, List

import torch
from torch.utils.data import DataLoader
from websockets import serve, connect
import websockets

import loralib as lora

from src.data_utils import FT_Dataset
from src.exp_utils import create_exp_dir
from src.optimizer import (
    add_optimizer_params,
    create_adam_optimizer_from_args,
    create_optimizer_scheduler,
)
from src.splitmodel import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Efficiently serialize a tensor with compression."""
    # Save tensor to bytes buffer
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserialize a compressed tensor."""
    # Decompress
    buffer = io.BytesIO(data)
    return torch.load(buffer) 


def serialize_message(data: Dict) -> bytes:
    """Serialize a message with tensor compression."""
    # Separate tensors from other data
    serialized = {}
    tensor_keys = []
    
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            serialized[key] = serialize_tensor(value)
            tensor_keys.append(key)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            serialized[key] = [serialize_tensor(t) if t is not None else None for t in value]
            tensor_keys.append(key)
        else:
            serialized[key] = value
    
    serialized['_tensor_keys'] = tensor_keys
    return pickle.dumps(serialized)


def deserialize_message(data: bytes) -> Dict:
    """Deserialize a message with compressed tensors."""
    deserialized = pickle.loads(data)
    tensor_keys = deserialized.pop('_tensor_keys', [])
    
    for key in tensor_keys:
        if isinstance(deserialized[key], list):
            deserialized[key] = [deserialize_tensor(t) if t is not None else None 
                                 for t in deserialized[key]]
        else:
            deserialized[key] = deserialize_tensor(deserialized[key])
    
    return deserialized


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def build_dataloader(
    path: str,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
    joint_lm: bool = False,
    vocab_path: Optional[str] = None,
) -> DataLoader:
    dataset = FT_Dataset(
        path,
        batch_size=batch_size,
        max_seq_length=seq_len,
        joint_lm=joint_lm,
        vocab_path=vocab_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=shuffle,
        pin_memory=False,
    )


def build_config(args: argparse.Namespace) -> GPT2Config:
    if args.model_card == "gpt2.sm":
        return GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_layers=args.client_layers,
        )
    if args.model_card == "gpt2.md":
        return GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_layers=args.client_layers,
        )
    if args.model_card == "gpt2.lg":
        return GPT2Config(
            n_embd=1280,
            n_layer=36,
            n_head=20,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_layers=args.client_layers,
        )
    raise ValueError(f"Unsupported model_card: {args.model_card}")


def save_lora_adapters(
    model_client: GPT2LMModel_Client,
    model_server: GPT2LMModel_Server,
    config: GPT2Config,
    output_dir: str,
    tag: str,
) -> None:
    payload = {
        "client": lora.lora_state_dict(model_client),
        "server": lora.lora_state_dict(model_server),
        "config": {
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "client_layers": config.client_layers,
            "seq_len": config.n_ctx,
            "vocab_size": config.vocab_size,
        },
        "tag": tag,
    }
    os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, os.path.join(output_dir, f"lora_adapters_{tag}.pt"))


# ==================== SERVER (Device 1) ====================

class ServerHandler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.server_device)
        self.config = build_config(args)
        
        print(f"[Server] Initializing on device: {self.device}")
        self.model_server = GPT2LMModel_Server(self.config).to(self.device)
        
        # Load pretrained weights
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        print(f"[Server] Loading pretrained weights from {args.init_checkpoint}")
        self.model_server.load_weight(checkpoint)
        
        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(self.model_server)
        
        self.optimizer_server = create_adam_optimizer_from_args(self.model_server, args)
        self.scheduler_server = create_optimizer_scheduler(self.optimizer_server, args)
        
        # Server-side statistics for logging
        self.step_count = 0
        self.forward_count = 0
        self.backward_count = 0
        self.total_forward_time = 0.0
        self.total_backward_time = 0.0
        self.last_log_step = 0
        self.last_log_time = None
        
        print(f"[Server] Ready and waiting for client connection on port {args.server_port}")

    async def handle_client(self, websocket):
        """Handle incoming requests from client."""
        print("[Server] Client connected!")
        
        try:
            async for message in websocket:
                data = deserialize_message(message)
                msg_type = data["type"]
                
                if msg_type == "forward":
                    # Receive hidden states and presents from client
                    hidden_states = data["hidden_states"].to(self.device)
                    presents = [p.to(self.device) if p is not None else None for p in data["presents"]]
                    input_shape = data["input_shape"]
                    labels = data["labels"].to(self.device)
                    mask = data["mask"].to(self.device)
                    label_smooth = data["label_smooth"]
                    
                    # Server forward pass
                    hidden_states.requires_grad_(True)
                    _, loss = self.model_server(
                        input_shape,
                        hidden_states,
                        presents,
                        lm_labels=labels,
                        lm_mask=mask,
                        label_smooth=label_smooth,
                    )
                    
                    loss = loss.mean()
                    
                    # Send loss back to client
                    response = {
                        "type": "forward_response",
                        "loss": loss.item(),
                        "hidden_states_id": id(hidden_states),
                    }
                    await websocket.send(serialize_message(response))
                    
                    # Store for backward pass
                    self.current_hidden_states = hidden_states
                    self.current_loss = loss
                
                elif msg_type == "backward":
                    grad_acc = data["grad_acc"]
                    
                    # Server backward pass
                    loss_for_backward = self.current_loss / grad_acc
                    loss_for_backward.backward()
                    
                    # Send gradients back to client
                    gradients = self.current_hidden_states.grad.cpu()
                    response = {
                        "type": "backward_response",
                        "gradients": gradients,
                    }
                    await websocket.send(serialize_message(response))
                    
                    # Cleanup
                    del self.current_hidden_states, self.current_loss
                
                elif msg_type == "optimizer_step":
                    clip = data["clip"]
                    current_step = data.get("current_step", 0)
                    total_steps = data.get("total_steps", 0)
                    loss = data.get("loss", 0.0)
                    
                    # Gradient clipping and optimizer step
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model_server.parameters(), clip)
                    
                    self.optimizer_server.step()
                    self.optimizer_server.zero_grad()
                    
                    if self.scheduler_server is not None:
                        self.scheduler_server.step()
                    
                    self.step_count += 1
                    
                    # Log every 10 steps
                    if self.step_count % 10 == 0:
                        current_time = time.time()
                        if self.last_log_time is not None:
                            steps_since_log = self.step_count - self.last_log_step
                            time_since_log = current_time - self.last_log_time
                            steps_per_sec = steps_since_log / time_since_log if time_since_log > 0 else 0
                            
                            # Calculate ETA
                            if total_steps > current_step and steps_per_sec > 0:
                                remaining_steps = total_steps - current_step
                                eta_seconds = remaining_steps / steps_per_sec
                                eta_minutes = eta_seconds / 60
                                eta_hours = eta_minutes / 60
                                
                                if eta_hours >= 1:
                                    eta_str = f"{eta_hours:.1f}h"
                                elif eta_minutes >= 1:
                                    eta_str = f"{eta_minutes:.1f}m"
                                else:
                                    eta_str = f"{eta_seconds:.0f}s"
                                
                                print(f"[Server] Step {current_step}/{total_steps} | Loss {loss:.4f} | "
                                      f"{steps_per_sec:.2f} steps/s | ETA: {eta_str}")
                            else:
                                print(f"[Server] Step {current_step} | Loss {loss:.4f} | "
                                      f"{steps_per_sec:.2f} steps/s")
                        else:
                            print(f"[Server] Step {current_step} | Loss {loss:.4f}")
                        
                        self.last_log_step = self.step_count
                        self.last_log_time = current_time
                    
                    response = {"type": "optimizer_step_response", "status": "ok"}
                    await websocket.send(serialize_message(response))
                
                elif msg_type == "get_server_state":
                    # Send server LoRA state dict
                    state_dict = lora.lora_state_dict(self.model_server)
                    response = {
                        "type": "server_state_response",
                        "state_dict": state_dict,
                    }
                    await websocket.send(serialize_message(response))
                
                elif msg_type == "set_train_mode":
                    self.model_server.train()
                    response = {"type": "mode_response", "status": "train"}
                    await websocket.send(serialize_message(response))
                
                elif msg_type == "set_eval_mode":
                    self.model_server.eval()
                    response = {"type": "mode_response", "status": "eval"}
                    await websocket.send(serialize_message(response))
                
                elif msg_type == "shutdown":
                    print("[Server] Received shutdown signal")
                    response = {"type": "shutdown_response", "status": "ok"}
                    await websocket.send(serialize_message(response))
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("[Server] Client disconnected")
        except Exception as e:
            print(f"[Server] Error: {e}")
            import traceback
            traceback.print_exc()


async def run_server(args: argparse.Namespace):
    """Run as server (Device 1)."""
    handler = ServerHandler(args)
    
    async with serve(
        handler.handle_client, 
        args.server_host, 
        args.server_port,
        max_size=100 * 1024 * 1024  # 100MB max message size
    ):
        print(f"[Server] Listening on ws://{args.server_host}:{args.server_port}")
        await asyncio.Future()  # Run forever


# ==================== CLIENT (Device 0) ====================

class ClientTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.client_device)
        self.config = build_config(args)
        
        print(f"[Client] Initializing on device: {self.device}")
        self.model_client = GPT2LMModel_Client(self.config).to(self.device)
        
        # Load pretrained weights
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        print(f"[Client] Loading pretrained weights from {args.init_checkpoint}")
        self.model_client.load_weight(checkpoint)
        
        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(self.model_client)
        
        self.optimizer_client = create_adam_optimizer_from_args(self.model_client, args)
        self.scheduler_client = create_optimizer_scheduler(self.optimizer_client, args)
        
        # Build dataloaders
        print("[Client] Loading datasets...")
        self.train_loader = build_dataloader(
            args.train_data,
            args.train_batch_size,
            args.seq_len,
            shuffle=True,
            joint_lm=False,
            vocab_path=args.vocab_path,
        )
        self.valid_loader = build_dataloader(
            args.valid_data,
            args.valid_batch_size,
            args.seq_len,
            shuffle=False,
            joint_lm=False,
            vocab_path=args.vocab_path,
        )
        
        # Calculate max_step if not provided
        if args.max_step is None:
            steps_per_epoch = len(self.train_loader)
            args.max_step = steps_per_epoch * args.epochs
            print(f"[Client] Calculated max_step: {args.max_step}")
        
        # Logging
        self.log_fn = create_exp_dir(args.output_dir)
        self.log_fn(str(args))
        
        self.websocket = None

    async def connect_to_server(self):
        """Connect to server."""
        max_retries = 10
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"[Client] Attempting to connect to server (attempt {attempt + 1}/{max_retries})...")
                self.websocket = await connect(
                    f"ws://{self.args.server_host}:{self.args.server_port}",
                    max_size=100 * 1024 * 1024  # 100MB max message size
                )
                print("[Client] Connected to server!")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Client] Connection failed: {e}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"[Client] Failed to connect after {max_retries} attempts")
                    return False
        return False

    async def train_epoch(self, epoch: int) -> Tuple[int, float]:
        """Train one epoch."""
        self.model_client.train()
        
        # Set server to train mode
        await self.websocket.send(serialize_message({"type": "set_train_mode"}))
        response = deserialize_message(await self.websocket.recv())
        
        avg_loss = AverageMeter()
        global_step = 0
        start_time = time.time()
        epoch_start_time = time.time()
        last_log_step = 0
        last_log_time = time.time()
        
        # Calculate total steps for this epoch
        total_steps_in_epoch = len(self.train_loader) // self.args.grad_acc
        if self.args.max_train_steps is not None:
            total_steps_in_epoch = min(total_steps_in_epoch, self.args.max_train_steps)
        
        self.optimizer_client.zero_grad()
        
        for step, batch in enumerate(self.train_loader, start=1):
            inputs = batch["input"].to(self.device)
            labels = batch["target"]
            mask = batch["mask"]
            
            # Client forward pass
            hidden_states, presents = self.model_client(inputs)
            
            # Send to server for forward pass
            message = {
                "type": "forward",
                "hidden_states": hidden_states.detach().cpu(),
                "presents": [p.detach().cpu() if p is not None else None for p in presents],
                "input_shape": inputs.shape,
                "labels": labels,
                "mask": mask,
                "label_smooth": self.args.label_smooth,
            }
            await self.websocket.send(serialize_message(message))
            
            # Receive loss from server
            response = deserialize_message(await self.websocket.recv())
            loss = response["loss"]
            
            # Request backward pass on server
            message = {
                "type": "backward",
                "grad_acc": self.args.grad_acc,
            }
            await self.websocket.send(serialize_message(message))
            
            # Receive gradients from server
            response = deserialize_message(await self.websocket.recv())
            gradients = response["gradients"].to(self.device)
            
            # Client backward pass
            hidden_states.backward(gradients)
            
            # Cleanup
            del hidden_states, presents, gradients
            if step % self.args.grad_acc == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Optimizer step
            if step % self.args.grad_acc == 0:
                if self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_client.parameters(), self.args.clip)
                
                global_step += 1
                avg_loss.update(loss)
                
                # Tell server to step (with extra info for logging)
                message = {
                    "type": "optimizer_step",
                    "clip": self.args.clip,
                    "current_step": global_step,
                    "total_steps": total_steps_in_epoch,
                    "loss": avg_loss.avg
                }
                await self.websocket.send(serialize_message(message))
                response = deserialize_message(await self.websocket.recv())
                
                # Client optimizer step
                self.optimizer_client.step()
                self.optimizer_client.zero_grad()
                
                if self.scheduler_client is not None:
                    self.scheduler_client.step()
                
                # Log every 10 steps
                if global_step % 10 == 0:
                    current_time = time.time()
                    steps_since_log = global_step - last_log_step
                    time_since_log = current_time - last_log_time
                    steps_per_sec = steps_since_log / time_since_log if time_since_log > 0 else 0
                    
                    # Calculate ETA
                    remaining_steps = total_steps_in_epoch - global_step
                    if remaining_steps > 0 and steps_per_sec > 0:
                        eta_seconds = remaining_steps / steps_per_sec
                        eta_minutes = eta_seconds / 60
                        eta_hours = eta_minutes / 60
                        
                        if eta_hours >= 1:
                            eta_str = f"{eta_hours:.1f}h"
                        elif eta_minutes >= 1:
                            eta_str = f"{eta_minutes:.1f}m"
                        else:
                            eta_str = f"{eta_seconds:.0f}s"
                        
                        print(
                            f"[Client] [Epoch {epoch}] Step {global_step}/{total_steps_in_epoch} | "
                            f"Loss {avg_loss.avg:.4f} | PPL {math.exp(avg_loss.avg):.2f} | "
                            f"{steps_per_sec:.2f} steps/s | ETA: {eta_str}"
                        )
                    else:
                        print(
                            f"[Client] [Epoch {epoch}] Step {global_step}/{total_steps_in_epoch} | "
                            f"Loss {avg_loss.avg:.4f} | PPL {math.exp(avg_loss.avg):.2f} | "
                            f"{steps_per_sec:.2f} steps/s"
                        )
                    
                    last_log_step = global_step
                    last_log_time = current_time
                
                # Legacy logging (keep for compatibility with original log_interval)
                if global_step % self.args.log_interval == 0 and global_step % 10 != 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[Client] [Epoch {epoch}] Step {global_step} | "
                        f"Loss {avg_loss.avg:.4f} | PPL {math.exp(avg_loss.avg):.2f} | "
                        f"Time {elapsed:.1f}s"
                    )
                    start_time = time.time()
                
                if self.args.max_train_steps is not None and global_step >= self.args.max_train_steps:
                    break
        
        return global_step, avg_loss.avg

    async def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model_client.eval()
        
        # Set server to eval mode
        await self.websocket.send(serialize_message({"type": "set_eval_mode"}))
        response = deserialize_message(await self.websocket.recv())
        
        avg_loss = AverageMeter()
        
        with torch.no_grad():
            for batch in self.valid_loader:
                inputs = batch["input"].to(self.device)
                labels = batch["target"]
                mask = batch["mask"]
                
                # Client forward pass
                hidden_states, presents = self.model_client(inputs)
                
                # Send to server for forward pass (no backward needed)
                message = {
                    "type": "forward",
                    "hidden_states": hidden_states.cpu(),
                    "presents": [p.cpu() if p is not None else None for p in presents],
                    "input_shape": inputs.shape,
                    "labels": labels,
                    "mask": mask,
                    "label_smooth": self.args.label_smooth,
                }
                await self.websocket.send(serialize_message(message))
                
                # Receive loss from server
                response = deserialize_message(await self.websocket.recv())
                loss = response["loss"]
                avg_loss.update(loss)
        
        return avg_loss.avg, math.exp(avg_loss.avg)

    async def save_lora_adapters(self, tag: str):
        """Save LoRA adapters from both client and server."""
        # Get server state dict
        message = {"type": "get_server_state"}
        await self.websocket.send(serialize_message(message))
        response = deserialize_message(await self.websocket.recv())
        server_state_dict = response["state_dict"]
        
        # Save combined state
        payload = {
            "client": lora.lora_state_dict(self.model_client),
            "server": server_state_dict,
            "config": {
                "n_embd": self.config.n_embd,
                "n_layer": self.config.n_layer,
                "n_head": self.config.n_head,
                "client_layers": self.config.client_layers,
                "seq_len": self.config.n_ctx,
                "vocab_size": self.config.vocab_size,
            },
            "tag": tag,
        }
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_path = os.path.join(self.args.output_dir, f"lora_adapters_{tag}.pt")
        torch.save(payload, save_path)
        print(f"[Client] Saved LoRA adapters to {save_path}")

    async def train(self):
        """Main training loop."""
        if not await self.connect_to_server():
            print("[Client] Failed to connect to server. Exiting.")
            return
        
        try:
            total_steps = 0
            best_val = float("inf")
            
            for epoch in range(1, self.args.epochs + 1):
                print(f"\n[Client] Starting epoch {epoch}/{self.args.epochs}")
                steps, train_loss = await self.train_epoch(epoch)
                total_steps += steps
                
                msg = f"[epoch {epoch}] train_loss={train_loss:.4f} ppl={math.exp(train_loss):.2f}"
                print(msg)
                self.log_fn(msg)
                
                # Evaluation
                val_loss, val_ppl = await self.evaluate()
                msg = f"[epoch {epoch}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
                print(msg)
                self.log_fn(msg)
                
                # Save best model
                if val_loss < best_val:
                    best_val = val_loss
                    await self.save_lora_adapters(f"epoch{epoch}")
                
                if self.args.max_train_steps is not None and total_steps >= self.args.max_train_steps:
                    break
            
            # Save final model
            await self.save_lora_adapters("final")
            
            print("\n[Client] Training completed!")
            
        finally:
            # Shutdown server
            try:
                await self.websocket.send(serialize_message({"type": "shutdown"}))
                await self.websocket.recv()
            except:
                pass
            await self.websocket.close()


async def run_client(args: argparse.Namespace):
    """Run as client (Device 0) with data and training logic."""
    trainer = ClientTrainer(args)
    await trainer.train()


# ==================== MAIN ====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified split learning script for client/server training."
    )
    
    # Device configuration
    parser.add_argument("--device_id", type=int, required=True, choices=[0, 1],
                        help="Device ID: 0=client (with data), 1=server")
    
    # Network configuration
    parser.add_argument("--server_host", type=str, default="localhost",
                        help="Server hostname/IP")
    parser.add_argument("--server_port", type=int, default=8765,
                        help="Server WebSocket port")
    
    # Model configuration
    parser.add_argument("--init_checkpoint", type=str, 
                        default="../pretrained_checkpoints/gpt2-pytorch_model.bin",
                        help="Path to base GPT-2 checkpoint")
    parser.add_argument("--model_card", type=str, default="gpt2.sm",
                        choices=["gpt2.sm", "gpt2.md", "gpt2.lg"])
    parser.add_argument("--client_layers", type=int, default=6,
                        help="Number of layers on client side")
    
    # Data configuration (only needed for client/device_id=0)
    parser.add_argument("--train_data", type=str, 
                        default="../data/e2e/train.jsonl",
                        help="Path to training jsonl file")
    parser.add_argument("--valid_data", type=str,
                        default="../data/e2e/valid.jsonl", 
                        help="Path to validation jsonl file")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="Path to vocab directory")
    
    # Training configuration
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=1,
                        help="Validation batch size")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--grad_acc", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--label_smooth", type=float, default=0.0,
                        help="Label smoothing")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Maximum training steps")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # LoRA configuration
    parser.add_argument("--lora_dim", type=int, default=4,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for logs and adapters")
    
    # Device configuration
    parser.add_argument("--client_device", type=str, default="cuda:0",
                        help="Client device")
    parser.add_argument("--server_device", type=str, default="cuda:0",
                        help="Server device")
    
    add_optimizer_params(parser)
    
    return parser.parse_args()


async def main():
    args = parse_args()
    set_random_seed(args.seed)
    
    print("=" * 80)
    print(f"Starting as {'SERVER (Device 1)' if args.device_id == 1 else 'CLIENT (Device 0)'}")
    print("=" * 80)
    
    if args.device_id == 1:
        # Run as server
        await run_server(args)
    else:
        # Run as client
        await run_client(args)


if __name__ == "__main__":
    asyncio.run(main())
