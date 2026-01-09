import argparse
import torch
import loralib as lora
from typing import Optional, List

from src.splitmodel import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server


def load_lora_adapters(
    adapter_path: str,
    model_client: GPT2LMModel_Client,
    model_server: GPT2LMModel_Server,
):
    """Load LoRA adapters into client and server models."""
    checkpoint = torch.load(adapter_path, map_location="cpu")
    
    print(f"\nLoading LoRA adapters from: {adapter_path}")
    print(f"Adapter tag: {checkpoint.get('tag', 'N/A')}")
    
    # Load client LoRA weights
    client_lora = checkpoint["client"]
    model_client.load_state_dict(client_lora, strict=False)
    
    # Load server LoRA weights
    server_lora = checkpoint["server"]
    model_server.load_state_dict(server_lora, strict=False)
    
    print("LoRA adapters loaded successfully!")
    return checkpoint.get("config", {})


def generate_text(
    model_client: GPT2LMModel_Client,
    model_server: GPT2LMModel_Server,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    device_client: torch.device = torch.device("cuda:0"),
    device_server: torch.device = torch.device("cuda:0"),
) -> str:
    """Generate text using the split model with LoRA adapters."""
    model_client.eval()
    model_server.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device_client)
    
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            current_input = torch.tensor([generated]).to(device_client)
            
            # Client forward
            hidden_states, presents = model_client(current_input)
            
            # Move to server
            server_input = hidden_states.to(device_server)
            presents_server = [p.to(device_server) if p is not None else None for p in presents]
            
            # Server forward
            logits, _ = model_server(
                current_input.shape,
                server_input,
                presents_server,
                lm_labels=None,
                lm_mask=None,
            )
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for end of sequence
            if next_token == tokenizer.encoder.get('<|endoftext|>', 50256):
                break
            
            generated.append(next_token)
    
    return tokenizer.decode(generated)


def build_config_from_checkpoint(adapter_checkpoint: dict, args: argparse.Namespace) -> GPT2Config:
    """Build config from adapter checkpoint or command line args."""
    config_dict = adapter_checkpoint.get("config", {})
    
    if args.model_card == "gpt2.sm":
        return GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_layers=config_dict.get("client_layers", args.client_layers),
        )
    elif args.model_card == "gpt2.md":
        return GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_layers=config_dict.get("client_layers", args.client_layers),
        )
    elif args.model_card == "gpt2.lg":
        return GPT2Config(
            n_embd=1280,
            n_layer=36,
            n_head=20,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            client_layers=config_dict.get("client_layers", args.client_layers),
        )
    raise ValueError(f"Unsupported model_card: {args.model_card}")


def compare_models(args: argparse.Namespace):
    """Compare base model vs fine-tuned model generations."""
    from transformers import GPT2Tokenizer
    
    device_client = torch.device(args.client_device)
    device_server = torch.device(args.server_device)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load adapter checkpoint to get config
    adapter_checkpoint = torch.load(args.lora_adapters, map_location="cpu")
    config = build_config_from_checkpoint(adapter_checkpoint, args)
    
    # Initialize models
    print(f"Initializing {args.model_card} split model...")
    print(f"  Client layers: {config.client_layers}")
    print(f"  Server layers: {config.n_layer - config.client_layers}")
    
    model_client = GPT2LMModel_Client(config).to(device_client)
    model_server = GPT2LMModel_Server(config).to(device_server)
    
    # Load base checkpoint
    print(f"\nLoading base model from: {args.init_checkpoint}")
    base_checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    model_client.load_weight(base_checkpoint)
    model_server.load_weight(base_checkpoint)
    
    # Test prompts
    test_prompts = [
        "The restaurant is located",
        "This is a family friendly place",
        "The price range is moderate and",
        args.custom_prompt if args.custom_prompt else None,
    ]
    test_prompts = [p for p in test_prompts if p is not None]
    
    print("\n" + "="*80)
    print("COMPARING BASE MODEL vs FINE-TUNED MODEL")
    print("="*80)
    
    for prompt in test_prompts:
        print(f"\n{'─'*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'─'*80}")
        
        # Generate with base model (no LoRA)
        print("\n[BASE MODEL OUTPUT]")
        base_output = generate_text(
            model_client, model_server, tokenizer, prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device_client=device_client,
            device_server=device_server,
        )
        print(base_output)
        
        # Load LoRA adapters
        load_lora_adapters(args.lora_adapters, model_client, model_server)
        
        # Generate with fine-tuned model
        print("\n[FINE-TUNED MODEL OUTPUT]")
        finetuned_output = generate_text(
            model_client, model_server, tokenizer, prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device_client=device_client,
            device_server=device_server,
        )
        print(finetuned_output)
        
        # Reload base weights for next comparison
        model_client.load_weight(base_checkpoint)
        model_server.load_weight(base_checkpoint)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Inference with split GPT-2 LoRA adapters"
    )
    parser.add_argument(
        "--lora_adapters",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint (e.g., outputs/lora_adapters_final.pt)",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        required=True,
        help="Path to base GPT-2 checkpoint",
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default="gpt2.sm",
        choices=["gpt2.sm", "gpt2.md", "gpt2.lg"],
        help="Model size",
    )
    parser.add_argument(
        "--client_layers",
        type=int,
        default=6,
        help="Number of layers on client side",
    )
    parser.add_argument(
        "--client_device",
        type=str,
        default="cuda:0",
        help="Device for client partition",
    )
    parser.add_argument(
        "--server_device",
        type=str,
        default="cuda:0",
        help="Device for server partition",
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default=None,
        help="Custom prompt to test",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=4,
        help="LoRA rank (should match training)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (should match training)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (should match training)",
    )
    
    args = parser.parse_args()
    compare_models(args)


if __name__ == "__main__":
    main()
