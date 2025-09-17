# pip install fastapi "uvicorn[standard]" torch
import torch
import io
from fastapi import FastAPI, Request, Response
import uvicorn
from split_gpt2 import GPT2Config, GPT2LMModel_Server

# --- Configuration ---
DEVICE = "cpu"
HOST = "0.0.0.0"  # Listen on all available network interfaces
PORT = 8000

# --- Model Loading ---
print("Initializing server model...")
config = GPT2Config(lora_attn_dim=4, lora_attn_alpha=16)
server_model = GPT2LMModel_Server(config).to(DEVICE)
server_model.eval()  # Set to eval mode, as it's not being trained
print("--> Server model loaded successfully.")

# --- FastAPI App ---
app = FastAPI()

@app.post("/forward")
async def forward_pass(request: Request):
    print("\n[+] Received a request on /forward")
    
    # 1. Deserialize (unpack) the incoming data
    data_bytes = await request.body()
    buffer = io.BytesIO(data_bytes)
    client_data = torch.load(buffer)
    
    hidden_states_client = client_data["hidden_states"].to(DEVICE)
    presents_client = client_data["presents"].to(DEVICE)
    input_ids_shape = client_data["input_shape"]
    labels = client_data["labels"].to(DEVICE)
    
    print(f"--> Received hidden_states of shape: {hidden_states_client.shape}")

    # 2. Run the server model's forward pass
    with torch.no_grad(): # No need to track gradients on the server
        _, loss = server_model(
            input_ids_shape=input_ids_shape,
            hidden_states_client=hidden_states_client,
            presents_client=presents_client,
            lm_labels=labels
        )
    
    print(f"--> Calculated loss: {loss.item():.4f}")

    # 3. Serialize (package) the loss and send it back
    loss_buffer = io.BytesIO()
    torch.save(loss.cpu(), loss_buffer)
    loss_bytes = loss_buffer.getvalue()

    return Response(content=loss_bytes, media_type="application/octet-stream")

if __name__ == "__main__":
    print(f"Starting server at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)