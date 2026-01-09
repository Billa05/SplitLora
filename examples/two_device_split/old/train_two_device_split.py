import argparse
import math
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

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


def move_hidden_for_server(
    hidden_states: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return hidden_states.detach().to(device).requires_grad_(True)


def move_presents(
    presents: List[Optional[torch.Tensor]],
    device: torch.device,
) -> List[Optional[torch.Tensor]]:
    moved: List[Optional[torch.Tensor]] = []
    for item in presents:
        if item is None:
            moved.append(None)
        else:
            moved.append(item.detach().to(device))
    return moved


def train_epoch(
    epoch: int,
    model_client: GPT2LMModel_Client,
    model_server: GPT2LMModel_Server,
    optimizer_client: torch.optim.Optimizer,
    optimizer_server: torch.optim.Optimizer,
    scheduler_client: Optional[torch.optim.lr_scheduler._LRScheduler],
    scheduler_server: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    args: argparse.Namespace,
) -> Tuple[int, float]:
    model_client.train()
    model_server.train()

    client_device = torch.device(args.client_device)
    server_device = torch.device(args.server_device)

    avg_loss = AverageMeter()
    global_step = 0
    start_time = time.time()
    epoch_start_time = time.time()

    optimizer_client.zero_grad()
    optimizer_server.zero_grad()
    
    # Calculate total steps for this epoch for ETA calculation
    total_steps_in_epoch = len(dataloader) // args.grad_acc
    if len(dataloader) % args.grad_acc != 0:
        total_steps_in_epoch += 1

    for step, batch in enumerate(dataloader, start=1):
        inputs = batch["input"].to(client_device)
        labels = batch["target"].to(server_device)
        mask = batch["mask"].to(server_device)

        # Client forward pass
        hidden_states, presents = model_client(inputs)
        server_input = move_hidden_for_server(hidden_states, server_device)
        presents_server = move_presents(presents, server_device)

        # Server forward and loss computation
        _, loss = model_server(
            inputs.shape,
            server_input,
            presents_server,
            lm_labels=labels,
            lm_mask=mask,
            label_smooth=args.label_smooth,
        )

        loss = loss.mean()
        loss_for_backward = loss / args.grad_acc
        
        # Server backward pass
        loss_for_backward.backward()

        # Get gradient for client-side backward pass
        dfx_client = server_input.grad.clone().detach().to(client_device)
        
        # Client backward pass
        hidden_states.backward(dfx_client)
        
        # Clean up intermediate tensors
        del server_input, presents_server, dfx_client
        if step % args.grad_acc == 0:
            del hidden_states, presents
            if client_device.type == 'cuda' or server_device.type == 'cuda':
                torch.cuda.empty_cache()

        if step % args.grad_acc == 0:
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model_server.parameters(), args.clip)
                torch.nn.utils.clip_grad_norm_(model_client.parameters(), args.clip)

            optimizer_server.step()
            optimizer_client.step()
            optimizer_server.zero_grad()
            optimizer_client.zero_grad()

            if scheduler_server is not None:
                scheduler_server.step()
            if scheduler_client is not None:
                scheduler_client.step()

            global_step += 1
            avg_loss.update(loss.item())

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                time_per_step = (time.time() - epoch_start_time) / global_step
                remaining_steps = total_steps_in_epoch - global_step
                eta_seconds = time_per_step * remaining_steps
                eta_minutes = eta_seconds / 60
                
                print(
                    f"[epoch {epoch}] step {global_step}/{total_steps_in_epoch} | "
                    f"loss {avg_loss.avg:.4f} | ppl {math.exp(avg_loss.avg):.2f} | "
                    f"time {elapsed:.1f}s | ETA {eta_minutes:.1f}m"
                )
                start_time = time.time()

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

    return global_step, avg_loss.avg


@torch.no_grad()
def evaluate(
    model_client: GPT2LMModel_Client,
    model_server: GPT2LMModel_Server,
    dataloader: DataLoader,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    model_client.eval()
    model_server.eval()

    client_device = torch.device(args.client_device)
    server_device = torch.device(args.server_device)

    avg_loss = AverageMeter()

    for batch in dataloader:
        inputs = batch["input"].to(client_device)
        labels = batch["target"].to(server_device)
        mask = batch["mask"].to(server_device)

        hidden_states, presents = model_client(inputs)
        server_input = hidden_states.to(server_device)
        presents_server = move_presents(presents, server_device)

        _, loss = model_server(
            inputs.shape,
            server_input,
            presents_server,
            lm_labels=labels,
            lm_mask=mask,
            label_smooth=args.label_smooth,
        )
        avg_loss.update(loss.mean().item())

    return avg_loss.avg, math.exp(avg_loss.avg)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split GPT-2 fine-tuning with configurable client/server layers."
    )
    parser.add_argument("--train_data", required=True, help="Path to training jsonl file.")
    parser.add_argument("--valid_data", required=True, help="Path to validation jsonl file.")
    parser.add_argument("--init_checkpoint", required=True, help="Path to base GPT-2 checkpoint.")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to vocab directory (if data needs tokenization).")
    parser.add_argument("--output_dir", type=str, default="./two_device_outputs", help="Where to store logs and adapters.")
    parser.add_argument("--model_card", type=str, default="gpt2.sm", choices=["gpt2.sm", "gpt2.md", "gpt2.lg"])
    parser.add_argument("--client_layers", type=int, default=None, help="Number of transformer blocks that remain on the client side.")
    parser.add_argument("--client_device", type=str, default="cuda:0", help="Torch device string for the client partition.")
    parser.add_argument("--server_device", type=str, default="cuda:0", help="Torch device string for the server partition.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--seq_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--grad_acc", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--clip", type=float, default=0.0, help="Gradient clipping value.")
    parser.add_argument("--label_smooth", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional maximum number of optimizer steps.")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging frequency (optimizer steps).")
    parser.add_argument("--eval_interval", type=int, default=None, help="Optional evaluation frequency (optimizer steps).")
    parser.add_argument("--save_interval", type=int, default=None, help="Optional adapter save frequency (optimizer steps).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--lora_dim", type=int, default=4, help="LoRA rank for attention projections.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")

    add_optimizer_params(parser)

    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    client_device = torch.device(args.client_device)
    server_device = torch.device(args.server_device)

    config = build_config(args)

    log_fn = create_exp_dir(args.output_dir)
    log_fn(str(args))

    model_client = GPT2LMModel_Client(config).to(client_device)
    model_server = GPT2LMModel_Server(config).to(server_device)

    checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    print("Loading pretrained weights from", args.init_checkpoint)
    model_client.load_weight(checkpoint)
    model_server.load_weight(checkpoint)

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(model_client)
        lora.mark_only_lora_as_trainable(model_server)

    # Build dataloaders first to calculate max_step if needed
    train_loader = build_dataloader(
        args.train_data,
        args.train_batch_size,
        args.seq_len,
        shuffle=True,
        joint_lm=False,
        vocab_path=args.vocab_path,
    )
    valid_loader = build_dataloader(
        args.valid_data,
        args.valid_batch_size,
        args.seq_len,
        shuffle=False,
        joint_lm=False,
        vocab_path=args.vocab_path,
    )

    # Calculate max_step if not provided
    if args.max_step is None:
        steps_per_epoch = len(train_loader)
        args.max_step = steps_per_epoch * args.epochs
        print(f"Calculated max_step: {args.max_step} (steps_per_epoch={steps_per_epoch} * epochs={args.epochs})")

    optimizer_client = create_adam_optimizer_from_args(model_client, args)
    optimizer_server = create_adam_optimizer_from_args(model_server, args)

    scheduler_client = create_optimizer_scheduler(optimizer_client, args)
    scheduler_server = create_optimizer_scheduler(optimizer_server, args)

    total_steps = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        steps, train_loss = train_epoch(
            epoch,
            model_client,
            model_server,
            optimizer_client,
            optimizer_server,
            scheduler_client,
            scheduler_server,
            train_loader,
            args,
        )
        total_steps += steps
        log_fn(f"[epoch {epoch}] train_loss={train_loss:.4f} ppl={math.exp(train_loss):.2f}")

        val_loss, val_ppl = evaluate(model_client, model_server, valid_loader, args)
        log_fn(f"[epoch {epoch}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        if val_loss < best_val:
            best_val = val_loss
            save_lora_adapters(model_client, model_server, config, args.output_dir, f"epoch{epoch}")

        if args.max_train_steps is not None and total_steps >= args.max_train_steps:
            break

    save_lora_adapters(model_client, model_server, config, args.output_dir, "final")


if __name__ == "__main__":
    main()

