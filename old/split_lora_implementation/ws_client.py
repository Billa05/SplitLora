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

from data_utils import FT_Dataset
from splitmodel import GPT2Config, GPT2LMModel_Client
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
    model_card: str = "gpt2.sm"
    max_step: int = 2000
    
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
    "server_uri": "ws://127.0.0.1:8765",
    "valid_data_path": "./data/e2e/valid.jsonl",
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


async def client_main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger = logging.getLogger("ws_client")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ClientArgs()
    config = build_config(args.model_card, args)

    model_client = GPT2LMModel_Client(config).to(device)
    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(model_client)
    optimizer_client = create_adam_optimizer_from_args(model_client, args)

    # data
    train_data = FT_Dataset(CONFIG["train_data_path"], args.train_batch_size, args.seq_len)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, drop_last=True)

    async with websockets.connect(CONFIG["server_uri"], max_size=None, max_queue=None) as ws:
        logger.info(f"Connected to server {CONFIG['server_uri']}")
        await send_json(ws, {"type": "hello", "role": "client", "client_id": client_id})
        hello = await recv_json(ws)
        assert hello.get("type") == "hello" and hello.get("role") == "server"
        logger.info("Handshake complete")

        # Wait for server signal to start training
        start_msg = await recv_json(ws)
        assert start_msg.get("type") == "start_training"
        logger.info("Received start signal from server. Beginning training.")

        cfg = await recv_json(ws)
        assert cfg.get("type") == "config"
        args.seq_len = cfg["seq_len"]
        args.lora_dim = cfg["lora_dim"]
        args.grad_acc = cfg["grad_acc"]
        logger.info(f"Received config: seq_len={args.seq_len} lora_dim={args.lora_dim} grad_acc={args.grad_acc}")

        train_step = 0
        aggregate_step = 100
        pending_updates = []
        logger.info(f"Starting training with {len(train_loader)} batches")
        for batch_idx, batch in enumerate(train_loader):
            logger.info(f"Processing batch {batch_idx + 1}")
            if train_step >= args.max_step:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            _input = batch["input"]
            _target = batch["target"]
            _mask = batch["mask"]

            model_client.train()
            hidden_states, presents, w_client = model_client(_input)

            # send forward payload
            await send_json(ws, {"type": "forward"})
            buf = io.BytesIO(); torch.save(hidden_states.detach().cpu(), buf); await send_bytes(ws, buf.getvalue())
            buf = io.BytesIO(); torch.save(presents, buf); await send_bytes(ws, buf.getvalue())
            buf = io.BytesIO(); torch.save(_target.detach().cpu(), buf); await send_bytes(ws, buf.getvalue())
            buf = io.BytesIO(); torch.save(_mask.detach().cpu(), buf); await send_bytes(ws, buf.getvalue())

            # recv gradient
            while True:
                msg = await recv_json(ws)
                if msg.get("type") == "backward":
                    grad_bytes = await recv_bytes(ws)
                    grad = torch.load(io.BytesIO(grad_bytes), map_location=device)
                    print(f"Received backward for step {train_step+1}")
                    break
                elif msg.get("type") == "lora_update":
                    update_bytes = await recv_bytes(ws)
                    update_sd = torch.load(io.BytesIO(update_bytes), map_location="cpu")
                    pending_updates.append(update_sd)
                    logger.info("Received LoRA update, pending application")
                else:
                    logger.warning(f"Unexpected message type while waiting for backward: {msg.get('type')}")
                    # Skip unknown messages

            # apply gradient to client-side boundary
            hidden_states.backward(grad.to(device))
            optimizer_client.step()
            optimizer_client.zero_grad()
            logger.info(f"Step {train_step+1}: loss={msg.get('loss'):.4f} grad_received=True")

            # Apply any pending LoRA updates after the local step
            for update_sd in pending_updates:
                w_client_updated = model_client.state_dict()
                for k, v in update_sd.items():
                    pref = "transformer_Client." + k if not k.startswith("transformer_Client.") else k
                    if pref in w_client_updated:
                        w_client_updated[pref] = v
                model_client.load_state_dict(w_client_updated)
                logger.info("Applied pending LoRA update")
            pending_updates.clear()

            train_step += 1

            # send LoRA weights periodically for aggregation
            if train_step % aggregate_step == 0:
                # extract LoRA-only keys from w_client
                lora_only: Dict[str, torch.Tensor] = {}
                for k, v in w_client.items():
                    if k.endswith("lora_A") or k.endswith("lora_B"):
                        lora_only[k] = v.detach().cpu()
                await send_json(ws, {"type": "lora_state"})
                await send_bytes(ws, state_dict_to_bytes(lora_only))

                reply = await recv_json(ws)
                if reply.get("type") == "lora_update":
                    update_bytes = await recv_bytes(ws)
                    update_sd = torch.load(io.BytesIO(update_bytes), map_location="cpu")
                    pending_updates.append(update_sd)
                    logger.info("Received LoRA update from reply")
                else:
                    logger.debug("Server acknowledged LoRA state (no aggregation this step)")

        # Save LoRA adapter after training
        adapter_save_path = f"./adapter_client_{client_id}_final.pth"
        torch.save(model_client.state_dict(), adapter_save_path)
        logger.info(f"Adapter saved to {adapter_save_path}")
        await send_json(ws, {"type": "bye"})
        logger.info("Training finished, bye sent to server")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True, help="Client ID (0, 1, or 2)")
    args_parsed = parser.parse_args()
    client_id = args_parsed.client_id
    CONFIG["train_data_path"] = f"./data/e2e/train{client_id}.jsonl"
    asyncio.run(client_main())


