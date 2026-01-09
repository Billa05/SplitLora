import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import loralib as lora

# Import the custom classes (ensure split_gpt2.py is in the same directory or path)
from split_gpt2 import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server

# Configuration
MODEL_NAME = "gpt2"
CLIENT_MODEL_PATH = "gpt2-split-lora-client.pt"
DEVICE = "cpu"  # Change to "cuda" if you have a GPU
MAX_LENGTH = 50  # Maximum length of generated text
PROMPT = "To be yourself in a world that is constantly trying to make you"

# Load Tokenizer
print("Step 1: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("--> Tokenizer loaded successfully.")

# Initialize Models
print("\nStep 2: Initializing split client-server model with LoRA...")
config = GPT2Config(
    vocab_size_or_config_json_file=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    lora_attn_dim=4,
    lora_attn_alpha=16,
    lora_dropout=0.05,
)
client_model = GPT2LMModel_Client(config).to(DEVICE)
server_model = GPT2LMModel_Server(config).to(DEVICE)

# Load Pre-trained Weights
print("--> Loading pre-trained weights from Hugging Face 'gpt2'...")
pretrained_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
pretrained_state_dict = pretrained_model.state_dict()

# Clean state dictionary by removing "transformer." prefix
clean_state_dict = {}
for key, value in pretrained_state_dict.items():
    if key.startswith("transformer."):
        clean_state_dict[key[len("transformer."):]] = value
    else:
        clean_state_dict[key] = value

print("--> Distributing weights to the split client and server models...")
client_model.load_weight(clean_state_dict)
server_model.load_weight(clean_state_dict)
server_model.set_tied()  # Ensure tied embeddings for server model

# Load trained client model weights
print(f"--> Loading trained client model weights from {CLIENT_MODEL_PATH}...")
client_model.load_state_dict(torch.load(CLIENT_MODEL_PATH, map_location=DEVICE))

# Mark only LoRA parameters as trainable (for consistency, though not needed for inference)
if config.lora_attn_dim > 0:
    lora.mark_only_lora_as_trainable(client_model)
    lora.mark_only_lora_as_trainable(server_model)

# Free up memory
del pretrained_model
del pretrained_state_dict
del clean_state_dict
print("--> Models initialized successfully.")

client_model.eval()
server_model.eval()

# Inference
print("\nStep 3: Running inference...")
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
generated_ids = input_ids.clone()

with torch.no_grad():
    past = None
    for _ in range(MAX_LENGTH - input_ids.size(1)):
        # Client Forward Pass
        hidden_states, presents, _ = client_model(input_ids, past=past)
        
        # Server Forward Pass
        lm_logits, presents = server_model(
            input_ids_shape=input_ids.shape,
            hidden_states_client=hidden_states,
            presents_client=presents
        )
        
        # Get the logits for the last token
        next_token_logits = lm_logits[:, -1, :]
        
        # Greedy sampling: select the token with the highest probability
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append the new token to the generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        input_ids = next_token
        
        # Update past key-value pairs for the next iteration
        past = presents

# Decode the generated token IDs to text
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nGenerated Text:")
print(generated_text)
print("\n--> Inference complete!")