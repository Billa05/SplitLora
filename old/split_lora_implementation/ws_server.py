import asyncio
import itertools
import json
import os
import io
import logging
from typing import Dict, List

import torch
import loralib as lora
import websockets

from data_utils import FT_Dataset
from splitmodel import GPT2Config, GPT2LMModel_Server, GPT2LMModel_Client
from optimizer import create_adam_optimizer_from_args, create_optimizer_scheduler
from gpu import distributed_sync, cleanup
from ws_utils import (
    send_json,
    recv_json,
    send_bytes,
    recv_bytes,
    state_dict_to_bytes,
    bytes_to_state_dict,
)


class SimpleArgs:
    # Minimal subset needed from existing script
    random_seed: int = 42
    fp16: bool = False
    rank: int = 0
    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 500
    lora_dim: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    label_smooth: float = 0.0
    grad_acc: int = 1
    clip: float = 0.0
    max_epoch: int = 1
    max_step: int = 2000
    model_card: str = "gpt2.sm"  # or gpt2.md
    world_size: int = 1
    
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
    "host": "0.0.0.0",
    "port": 8765,
    "num_clients": 3, 
    "seq_len": 128,
    "train_batch_size": 4,
    "valid_batch_size": 4,
    "init_checkpoint": None,
}


def build_config(model_card: str, args: SimpleArgs) -> GPT2Config:
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


def fed_avg(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    w_avg = {k: v.clone() for k, v in weights[0].items()}
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = w_avg[k] / float(len(weights))
    return w_avg


async def handle_client(ws, state):
    logger = logging.getLogger("ws_server")
    device = state["device"]
    model_server: GPT2LMModel_Server = state["model_server"]
    optimizer_server = state["optimizer_server"]
    scheduler_server = state["scheduler_server"]
    args: SimpleArgs = state["args"]
    connected_clients = state["connected_clients"]
    w_locals_client = state["w_locals_client"]

    try:
        peer = ws.remote_address
        logger.info(f"Client connected from {peer}")
        await send_json(ws, {"type": "hello", "role": "server"})
        hello = await recv_json(ws)
        assert hello.get("type") == "hello" and hello.get("role") == "client"
        client_id = hello.get("client_id")
        logger.info(f"Handshake complete with {peer}, client_id={client_id}")
        connected_clients.append(ws)

        # Wait for all clients to connect before starting training
        await state["start_event"].wait()

        # Send start_training signal
        await send_json(ws, {"type": "start_training"})
        logger.info(f"Sent start_training to {peer}")

        # Send server-side config for split
        await send_json(ws, {
            "type": "config",
            "seq_len": CONFIG["seq_len"],
            "lora_dim": args.lora_dim,
            "grad_acc": args.grad_acc,
        })
        logger.info(f"Sent config to {peer}: seq_len={CONFIG['seq_len']} lora_dim={args.lora_dim} grad_acc={args.grad_acc}")

        train_step = 0
        aggregate_step = 100

        while True:
            msg = await recv_json(ws)
            mtype = msg.get("type")

            if mtype == "forward":
                # receive hidden_states and presents
                logger.debug(f"[{peer}] Receiving forward payload at step {train_step}")
                hidden_bytes = await recv_bytes(ws)
                presents_bytes = await recv_bytes(ws)
                labels_bytes = await recv_bytes(ws)
                mask_bytes = await recv_bytes(ws)

                hidden_states = torch.load(
                    io.BytesIO(hidden_bytes), map_location=device
                ).to(device).requires_grad_(True)
                presents = torch.load(io.BytesIO(presents_bytes), map_location=device)
                lm_labels = torch.load(io.BytesIO(labels_bytes), map_location=device)
                lm_mask = torch.load(io.BytesIO(mask_bytes), map_location=device)

                _, lm_loss = model_server(
                    (lm_labels.shape[0], CONFIG["seq_len"]),
                    hidden_states,
                    presents,
                    lm_labels=lm_labels,
                    lm_mask=lm_mask,
                    label_smooth=args.label_smooth,
                )
                loss = lm_loss.mean()
                loss.backward()

                dfx_client = hidden_states.grad.detach().cpu()
                optimizer_server.step()
                optimizer_server.zero_grad()
                if scheduler_server is not None:
                    scheduler_server.step()

                # send gradient back
                buffer = io.BytesIO()
                torch.save(dfx_client, buffer)
                await send_json(ws, {"type": "backward", "ok": True, "loss": float(loss.item())})
                await send_bytes(ws, buffer.getvalue())
                train_step += 1
                logger.info(f"[{peer}] Step {train_step}: loss={float(loss.item()):.4f} gradient_sent=True")

            elif mtype == "lora_state":
                # receive LoRA weights for aggregation
                lora_bytes = await recv_bytes(ws)
                w_client = bytes_to_state_dict(lora_bytes, device)
                w_locals_client.append(w_client)
                logger.debug(f"[{peer}] Received LoRA state (buffered {len(w_locals_client)}) at step {train_step}")

                if train_step > 0 and train_step % aggregate_step == 0:
                    # FedAvg LoRA-only keys
                    if len(w_locals_client) >= CONFIG["num_clients"]:
                        w_avg = fed_avg(w_locals_client[-CONFIG["num_clients"]:])
                        w_locals_client.clear()
                        # send update to all connected clients
                        for client_ws in connected_clients:
                            try:
                                await send_json(client_ws, {"type": "lora_update"})
                                await send_bytes(client_ws, state_dict_to_bytes(w_avg))
                            except Exception:
                                pass  # client may have disconnected
                        logger.info(f"[{peer}] Performed LoRA FedAvg at step {train_step} and sent update to {len(connected_clients)} clients")
                    else:
                        await send_json(ws, {"type": "lora_ack"})
                        logger.debug(f"[{peer}] LoRA ack sent (not enough clients for aggregation)")

            elif mtype == "bye":
                connected_clients.remove(ws)
                await send_json(ws, {"type": "bye"})
                logger.info(f"[{peer}] Client signaled bye. Closing handler.")
                break

    except Exception as e:
        logging.getLogger("ws_server").exception("Error in client handler")
        try:
            await send_json(ws, {"type": "error", "message": str(e)})
        except Exception:
            pass
        finally:
            if ws in connected_clients:
                connected_clients.remove(ws)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = SimpleArgs()
    config = build_config(args.model_card, args)

    model_server = GPT2LMModel_Server(config).to(device)
    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(model_server)

    optimizer_server = create_adam_optimizer_from_args(model_server, args)
    scheduler_server = create_optimizer_scheduler(optimizer_server, args)

    connected_clients = []
    w_locals_client = []
    state = {
        "device": device,
        "model_server": model_server,
        "optimizer_server": optimizer_server,
        "scheduler_server": scheduler_server,
        "args": args,
        "connected_clients": connected_clients,
        "w_locals_client": w_locals_client,
        "start_event": asyncio.Event(),
    }

    async def handler(ws):
        await handle_client(ws, state)

    async with websockets.serve(handler, CONFIG["host"], CONFIG["port"], max_size=None, max_queue=None):
        logging.getLogger("ws_server").info(f"Server listening on {CONFIG['host']}:{CONFIG['port']}")
        # Wait for all clients to connect
        while True:
            if len(connected_clients) >= CONFIG["num_clients"]:
                # Send start signal to all clients
                state["start_event"].set()
                break
            await asyncio.sleep(1)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())


