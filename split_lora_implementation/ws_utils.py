import asyncio
import io
import json
from typing import Dict, Any

import torch


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(t.cpu(), buffer)
    return buffer.getvalue()


def bytes_to_tensor(b: bytes, device: torch.device) -> torch.Tensor:
    buffer = io.BytesIO(b)
    t = torch.load(buffer, map_location=device)
    return t.to(device)


def state_dict_to_bytes(sd: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save({k: v.cpu() for k, v in sd.items()}, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(b: bytes, device: torch.device) -> Dict[str, torch.Tensor]:
    buffer = io.BytesIO(b)
    sd = torch.load(buffer, map_location=device)
    return {k: v.to(device) for k, v in sd.items()}


async def send_json(ws, payload: Dict[str, Any]):
    await ws.send(json.dumps(payload))


async def recv_json(ws) -> Dict[str, Any]:
    msg = await ws.recv()
    return json.loads(msg)


async def send_bytes(ws, b: bytes):
    await ws.send(b)


async def recv_bytes(ws) -> bytes:
    msg = await ws.recv()
    if isinstance(msg, bytes):
        return msg
    # Some clients may base64 or string-encode; enforce bytes usage.
    raise RuntimeError("Expected bytes frame, got text")


