# run_split_inference.py
import torch
from transformers import AutoTokenizer

# Import the custom classes
from split_gpt2 import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server

# =================================================================================
# 1. Configuration
# =================================================================================
MODEL_NAME = "gpt2"
CLIENT_MODEL_PATH = "gpt2-split-lora-client.pt"
DEVICE = "cpu" # Change to "cuda" if you have a GPU

# =================================================================================
# 2. Load Models and Tokenizer
# =================================================================================
print("Step 2: Loading models and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(
    lora_attn_dim=4,
    lora_attn_alpha=16,
)
print(f"--> Model configured with LoRA r={config.lora_attn_dim} and alpha={config.lora_attn_alpha}")

client_model = GPT2LMModel_Client(config).to(DEVICE)
server_model = GPT2LMModel_Server(config).to(DEVICE)

print(f"--> Loading fine-tuned weights for the client from '{CLIENT_MODEL_PATH}'...")
client_model.load_state_dict(torch.load(CLIENT_MODEL_PATH))
print("--> Weights loaded successfully.")

client_model.eval()
server_model.eval()
print("--> Models are in evaluation mode.")

# =================================================================================
# 3. Manual Inference Loop (Autoregressive Generation)
# =================================================================================
print("\nStep 3: Running inference...")
prompt = "To be yourself in a world that is constantly trying to make you"
print(f"Prompt: '{prompt}'")

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
max_length = 50
generated_tokens = list(input_ids[0].tolist())

print(f"\n--> Starting autoregressive generation from initial sequence of shape: {input_ids.shape}")
with torch.no_grad():
    for i in range(max_length - len(input_ids[0])):
        print(f"\n[Generating token {i+1}/{max_length - len(generated_tokens)}]")
        current_input = torch.tensor([generated_tokens]).to(DEVICE)
        print(f"  Current input shape: {current_input.shape}")

        # 1. Client Forward Pass
        print("  1. ==> Running Client forward pass...")
        hidden_states_client, presents_client, _ = client_model(current_input)
        print(f"  <== Client output (hidden_states) shape: {hidden_states_client.shape}")

        # 2. Server Forward Pass
        print("  2. ==> Sending hidden states to Server for forward pass...")
        lm_logits, _ = server_model(
            input_ids_shape=current_input.shape,
            hidden_states_client=hidden_states_client,
            presents_client=presents_client
        )
        print(f"  <== Server output (logits) shape: {lm_logits.shape}")

        # 3. Get the next token
        next_token_logits = lm_logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()

        # 4. Append the new token and continue the loop
        generated_tokens.append(next_token_id)
        decoded_token = tokenizer.decode([next_token_id])
        print(f"  4. ==> Chose next token ID: {next_token_id} ('{decoded_token.strip()}')")
        
        if next_token_id == tokenizer.eos_token_id:
            print("  --> End-of-sequence token generated. Stopping.")
            break

# Decode and print the result
print("\n--> Generation complete.")
final_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("\n=================================")
print("        Final Generated Text")
print("=================================")
print(final_output)