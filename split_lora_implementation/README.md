# Federated Split Learning with LoRA using WebSockets

## Overview

This project implements a **federated learning** system for fine-tuning a GPT-2 model using **split learning** and **LoRA (Low-Rank Adaptation)**. The architecture divides the model into client-side (first few layers) and server-side (remaining layers) components. Multiple clients train locally on their data, and the server aggregates updates via WebSockets for synchronized, privacy-preserving training.

### Key Concepts

- **Federated Learning**: Clients train on local data without sharing it. Only model updates (LoRA parameters) are exchanged.
- **Split Learning**: The model is split: clients process input through initial layers, send intermediate representations to the server, which completes the forward pass, computes loss, and sends gradients back.
- **LoRA**: Efficient fine-tuning by adapting low-rank matrices instead of full weights, reducing parameters and memory.
- **WebSockets**: Real-time communication between server and clients for synchronous training.
- **Multi-Client Synchronization**: Server waits for all clients to connect before starting training, ensuring coordinated updates.

The system supports 3 clients by default, aggregating LoRA updates every 100 steps using Federated Averaging (FedAvg).

## Prerequisites

- Python 3.8+
- PyTorch (with CUDA support for GPU)
- `websockets` library
- `loralib` for LoRA
- `transformers` for tokenizer (inference only)
- Access to training data (e.g., E2E dataset in `./data/e2e/`)

## Installation and Setup

1. **Clone or navigate to the project directory**:
   ```
   cd /path/to/split_lora_implementation
   ```

2. **Install dependencies**:
   ```
   pip install torch websockets loralib transformers
   ```
   Ensure PyTorch is installed with CUDA if using GPU.

3. **Prepare data**:
   - Place training data in `./data/e2e/train{client_id}.jsonl` (e.g., `train0.jsonl`, `train1.jsonl`, `train2.jsonl`).
   - Ensure validation data in `./data/e2e/valid.jsonl`.

4. **Directory structure**:
   ```
   split_lora_implementation/
   ├── ws_server.py          # WebSocket server
   ├── ws_client.py           # WebSocket client
   ├── ws_utils.py            # Utilities for WebSocket communication
   ├── infer_adapter.py       # Inference script
   ├── data_utils.py          # Data loading utilities
   ├── splitmodel.py          # GPT-2 model definitions
   ├── optimizer.py           # Optimizer setup
   └── data/
       └── e2e/
           ├── train0.jsonl
           ├── train1.jsonl
           ├── train2.jsonl
           └── valid.jsonl
   ```

## Running the System

### Step 1: Start the Server

The server hosts the server-side model, handles client connections, and coordinates training.

1. Open a terminal and run:
   ```
   python ws_server.py
   ```
   - The server listens on `0.0.0.0:8765`.
   - It waits for 3 clients to connect (configurable in `CONFIG["num_clients"]`).
   - Once all clients connect, it sends a "start_training" signal and config to each client.

### Step 2: Start the Clients

Each client runs on a separate machine or terminal, training on local data.

For each client (0, 1, 2):

1. Open a new terminal and run:
   ```
   python ws_client.py --client_id 0
   ```
   Replace `0` with `1` or `2` for other clients.

2. Clients connect to `ws://127.0.0.1:8765` (update `CONFIG["server_uri"]` if needed).

3. Upon connection:
   - Handshake with server.
   - Wait for "start_training" signal.
   - Receive config (seq_len, lora_dim, grad_acc).
   - Begin training loop.

### Training Process Flow

1. **Initialization**:
   - Server initializes server-side model (layers 3-11 for GPT-2 small).
   - Clients initialize client-side model (layers 0-2) with LoRA.

2. **Training Loop** (per client):
   - Load batch from local data.
   - Forward pass through client model → send hidden_states, presents, labels, mask to server.
   - Server: Forward through server model → compute loss → backward → send gradients back.
   - Client: Apply gradients → update client model.
   - Every 100 steps: Client sends LoRA parameters to server.
   - Server aggregates LoRA from all clients (FedAvg) and sends updates back.

3. **Synchronization**:
   - Clients process batches asynchronously but synchronize LoRA aggregation.
   - Server ensures all clients contribute before aggregating.

4. **Termination**:
   - After max_steps (2000), clients save adapters (`adapter_client_{id}_final.pth`) and send "bye".

### Step 3: Run Inference

After training, use the saved adapters for inference.

1. Run the inference script:
   ```
   python infer_adapter.py
   ```
   - Loads and averages LoRA from all adapters.
   - Merges into client model.
   - Performs full forward pass (client + server) for next word prediction.

Example output:
```
Input: The quick brown fox
Next word prediction: jumps
```

## Files Description

- **`ws_server.py`**: Manages server-side logic, client connections, model aggregation.
- **`ws_client.py`**: Handles client-side training, communication with server.
- **`ws_utils.py`**: Helper functions for sending/receiving JSON and bytes over WebSockets.
- **`infer_adapter.py`**: Loads adapters, averages LoRA, merges, and runs inference.
- **`splitmodel.py`**: Defines GPT2Model_Client, GPT2Model_Server, configs.
- **`data_utils.py`**: Dataset class for loading JSONL data.
- **`optimizer.py`**: Optimizer setup.

## Configuration

- **Server Config** (`ws_server.py`):
  - `num_clients`: Number of clients (default 3).
  - `seq_len`: Sequence length (128).
  - `lora_dim`: LoRA rank (8).
  - `max_step`: Training steps (2000).

- **Client Config** (`ws_client.py`):
  - `train_batch_size`: Batch size (1).
  - `lr`: Learning rate (0.00001).

Adjust in the respective files as needed.

## Troubleshooting

- **Connection Issues**: Ensure server is running before clients. Check firewall for port 8765.
- **Data Not Found**: Verify `./data/e2e/train{client_id}.jsonl` exists.
- **CUDA Errors**: Ensure GPU memory is sufficient; reduce batch_size if needed.
- **Assertion Errors**: Check logs for message mismatches; ensure correct client count.
- **Inference Errors**: Confirm adapter paths in `infer_adapter.py`.

## Flow Summary

1. Setup environment and data.
2. Run server → waits for clients.
3. Run 3 clients → connect, wait for start.
4. Training: Clients send forward, server responds with backward, periodic LoRA aggregation.
5. Clients save adapters.
6. Run inference with averaged adapters for predictions.

This setup enables efficient, privacy-preserving fine-tuning across distributed clients.
