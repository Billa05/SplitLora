import torch
from splitmodel import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server
import loralib as lora
from transformers import GPT2Tokenizer

# Set your config to match training
config = GPT2Config(
    vocab_size_or_config_json_file=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    layer_norm_epsilon=1e-5,
    lora_attn_dim=8,
    lora_attn_alpha=16,
    lora_dropout=0.0,
)

def load_adapters_and_infer(adapter_paths, input_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_client = GPT2LMModel_Client(config).to(device)
    
    # Mark LoRA as trainable to add LoRA parameters
    lora.mark_only_lora_as_trainable(model_client)
    
    # Collect and average LoRA parameters from all adapters
    lora_params = {}
    for path in adapter_paths:
        state = torch.load(path, map_location=device)
        for k, v in state.items():
            if k.startswith("transformer_Client.") and ("lora" in k):
                new_k = k  # Keep the full key with prefix
                if new_k not in lora_params:
                    lora_params[new_k] = []
                lora_params[new_k].append(v)
    
    for k in lora_params:
        lora_params[k] = torch.stack(lora_params[k]).mean(0)
    
    # Load averaged LoRA into model_client
    state_dict = model_client.state_dict()
    for k, v in lora_params.items():
        state_dict[k] = v
    model_client.load_state_dict(state_dict)
    
    # Merge LoRA into base weights
    for module in model_client.modules():
        if hasattr(module, 'merge'):
            module.merge()
    
    # Server model (base, no fine-tuning)
    model_server = GPT2LMModel_Server(config).to(device)
    
    model_client.eval()
    model_server.eval()
    
    with torch.no_grad():
        hidden_states, presents, _ = model_client(input_tensor)
        input_shape = input_tensor.shape
        lm_logits, _ = model_server(input_shape, hidden_states, presents)
    
    return lm_logits

if __name__ == "__main__":
    # List of all adapter paths
    adapter_paths = [
        "./adapter_client_0_final.pth",
        "./adapter_client_1_final.pth",
        "./adapter_client_2_final.pth"
    ]
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    
    # Example input text
    input_text = "The quick brown fox"
    input_tensor = tokenizer.encode(input_text, return_tensors='pt').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    lm_logits = load_adapters_and_infer(adapter_paths, input_tensor)
    
    # Predict next token
    predicted_token_id = torch.argmax(lm_logits[:, -1, :], dim=-1).item()
    predicted_word = tokenizer.decode(predicted_token_id)
    
    print(f"Input: {input_text}")
    print(f"Next word prediction: {predicted_word}")
    print("Inference complete. Logits shape:", lm_logits.shape)
