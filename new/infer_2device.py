"""
Simple inference script for 2-device pipeline with trained LoRA adapters.
CORRECTED VERSION: Uses proper layer-wise split model.
Usage: python infer_2device.py --prompt "Your text here"
"""

import torch
from transformers import GPT2Tokenizer, GPT2Config
from splitmodel import GPT2SplitPart, get_lora_config
import argparse


def generate_text(prompt, max_length=50, temperature=1.0, top_k=50):
    """Generate text using the 2-device pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Device 0 model (embeddings + layers 0-5)
    print("Loading Device 0 (embeddings + layers 0-5)...")
    config = GPT2Config()
    lora_config = get_lora_config(r=8, alpha=16, dropout=0.0)
    
    model_0 = GPT2SplitPart(config, 0, 6, has_embeddings=True, has_lm_head=False, lora_config=lora_config).to(device)
    
    # Load saved state dict
    state_dict_0 = torch.load("./lora_device_0.pth", map_location=device)
    model_0.load_state_dict(state_dict_0, strict=False)
    model_0.eval()
    
    # Load Device 1 model (layers 6-11 + LM head)
    print("Loading Device 1 (layers 6-11 + LM head)...")
    model_1 = GPT2SplitPart(config, 6, 12, has_embeddings=False, has_lm_head=True, lora_config=lora_config).to(device)
    
    # Load saved state dict
    state_dict_1 = torch.load("./lora_device_1.pth", map_location=device)
    model_1.load_state_dict(state_dict_1, strict=False)
    model_1.eval()
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_length} tokens...\n")
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward through Device 0
            hidden_states, _ = model_0(input_ids=generated)
            
            # Forward through Device 1
            logits, _ = model_1(hidden_states=hidden_states)
            
            # Get next token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation with 2-device LoRA pipeline")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum tokens to generate (default: 50)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering (default: 50)")
    
    args = parser.parse_args()
    
    generated = generate_text(
        args.prompt, 
        args.max_length, 
        args.temperature, 
        args.top_k
    )
    
    print(f"Generated text:\n{generated}\n")
