# run_client_inference.py
import torch
import requests
import io
from transformers import AutoTokenizer
from split_gpt2 import GPT2Config, GPT2LMModel_Client

# --- Configuration ---
# !!! IMPORTANT: Replace with your friend's local IP address !!!
SERVER_URL = "http://172.30.153.97:8000/generate" 
ADAPTER_PATH = "final_client_lora_adapter.pt"
DEVICE = "cpu"

# --- Model Loading ---
print("Step 1: Loading tokenizer and client model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(lora_attn_dim=4, lora_attn_alpha=16)
client_model = GPT2LMModel_Client(config).to(DEVICE)

print(f"--> Loading fine-tuned adapter weights from '{ADAPTER_PATH}'...")
client_model.load_state_dict(torch.load(ADAPTER_PATH))
client_model.eval()
print("--> Client model ready for inference.")

# --- Inference Loop ---
prompt = "To be yourself in a world that is constantly trying to make you"
print(f"\nStep 2: Running distributed inference...\nPrompt: '{prompt}'")

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
max_length = 50

with torch.no_grad():
    for _ in range(max_length - len(input_ids[0])):
        # 1. Run client forward pass
        hidden_states_client, presents_client, _ = client_model(input_ids)

        # 2. Package data and send to server
        data_to_send = {
            "hidden_states": hidden_states_client.cpu(),
            "presents": [p.cpu() for p in presents_client],
            "input_shape": input_ids.shape
        }
        buffer = io.BytesIO()
        torch.save(data_to_send, buffer)
        payload_bytes = buffer.getvalue()
        
        response = requests.post(SERVER_URL, data=payload_bytes, headers={'Content-Type': 'application/octet-stream'})
        
        if response.status_code != 200:
            print("Error from server:", response.text)
            break

        # 3. Get logits and determine next token
        logits_buffer = io.BytesIO(response.content)
        lm_logits = torch.load(logits_buffer).to(DEVICE)
        
        next_token_id = torch.argmax(lm_logits[:, -1, :], dim=-1)
        
        # 4. Append new token and continue
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

# --- Final Output ---
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("\n=====================")
print("  Generated Text")
print("=====================")
print(generated_text)