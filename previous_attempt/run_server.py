# pip install fastapi "uvicorn[standard]" torch
import torch
import io
from fastapi import FastAPI, Request, Response
import uvicorn
from split_gpt2 import GPT2Config, GPT2LMModel_Server
from transformers import AutoModelForCausalLM 

# --- Configuration ---
DEVICE = "cpu"
HOST = "0.0.0.0"  # Listen on all available network interfaces
PORT = 8000

# --- Model Loading ---
print("Initializing server model...")
config = GPT2Config(lora_attn_dim=4, lora_attn_alpha=16)
server_model = GPT2LMModel_Server(config).to(DEVICE)
# --- NEW: Load Pre-trained GPT-2 Weights ---
print("Downloading and loading pre-trained GPT-2 weights...")
# 1. Load the standard, full GPT-2 model to get its weights
pretrained_model = AutoModelForCausalLM.from_pretrained("gpt2")
state_dict = pretrained_model.state_dict()
# 2. Use the custom load_weight function to transfer them
server_model.load_weight(state_dict)
del pretrained_model # Free up memory
print("--> Pre-trained weights loaded into server model.")

server_model.eval()
print("--> Server model is ready.")

# --- FastAPI App ---
app = FastAPI()

@app.post("/forward")
async def forward_pass(request: Request):
    print("\n[+] Received a request on /forward")
    
    # 1. Deserialize the incoming data
    data_bytes = await request.body()
    buffer = io.BytesIO(data_bytes)
    client_data = torch.load(buffer)
    
    hidden_states_client = client_data["hidden_states"].to(DEVICE)
    presents_client = [p.to(DEVICE) for p in client_data["presents"]]
    input_ids_shape = client_data["input_shape"]
    labels = client_data["labels"].to(DEVICE)
    
    # IMPORTANT: We must tell PyTorch to track gradients for this tensor
    hidden_states_client.requires_grad_()
    
    print(f"--> Received hidden_states of shape: {hidden_states_client.shape}")

    # 2. Run the server model's forward pass AND backward pass
    # We remove torch.no_grad() to build the server-side graph
    server_model.zero_grad() # Clear any old gradients
    _, loss = server_model(
        input_ids_shape=input_ids_shape,
        hidden_states_client=hidden_states_client,
        presents_client=presents_client,
        lm_labels=labels
    )
    
    print(f"--> Calculated loss: {loss.item():.4f}")
    
    # Run backpropagation on the server to get the gradient of the hidden states
    loss.backward()
    
    # 3. Get the gradient and send IT back to the client
    grad_to_send = hidden_states_client.grad.cpu()
    print(f"--> Sending back gradient of shape: {grad_to_send.shape}")

    grad_buffer = io.BytesIO()
    torch.save(grad_to_send, grad_buffer)
    grad_bytes = grad_buffer.getvalue()

    return Response(content=grad_bytes, media_type="application/octet-stream")

@app.post("/generate")
async def generate_pass(request: Request):
    print("\n[+] Received a request on /generate")
    
    # 1. Deserialize the incoming data
    data_bytes = await request.body()
    buffer = io.BytesIO(data_bytes)
    client_data = torch.load(buffer)
    
    hidden_states_client = client_data["hidden_states"].to(DEVICE)
    presents_client = [p.to(DEVICE) for p in client_data["presents"]]
    input_ids_shape = client_data["input_shape"]
    
    # 2. Run the server model's forward pass to get logits
    with torch.no_grad():
        lm_logits, _ = server_model(
            input_ids_shape=input_ids_shape,
            hidden_states_client=hidden_states_client,
            presents_client=presents_client
        )
    
    print(f"--> Returning logits of shape: {lm_logits.shape}")

    # 3. Serialize the logits and send them back
    logits_buffer = io.BytesIO()
    torch.save(lm_logits.cpu(), logits_buffer)
    logits_bytes = logits_buffer.getvalue()

    return Response(content=logits_bytes, media_type="application/octet-stream")

if __name__ == "__main__":
    print(f"Starting server at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)