#!/usr/bin/env python3
"""
Inference script specifically designed for E2E NLG dataset format.
This script properly formats the context as the model expects.
"""
import argparse
import torch
import loralib as lora
from typing import Optional, List

from src.splitmodel import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server

try:
    from src.encoder import get_encoder
    ENCODER_AVAILABLE = True
except ImportError:
    try:
        from encoder import get_encoder
        ENCODER_AVAILABLE = True
    except ImportError:
        ENCODER_AVAILABLE = False


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


def generate_text_e2e(
    model_client: GPT2LMModel_Client,
    model_server: GPT2LMModel_Server,
    encoder,
    context: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device_client: torch.device = torch.device("cuda:0"),
    device_server: torch.device = torch.device("cuda:0"),
) -> str:
    """
    Generate completion for E2E NLG dataset context.
    
    The E2E dataset format is:
    - Input (context): "name : X | Type : Y | ..."
    - Output (completion): Natural language description
    
    During training, the model sees: [context_tokens] + [BOS] + [completion_tokens] + [EOS]
    During inference, we provide: [context_tokens] + [BOS] and let it generate the completion.
    """
    model_client.eval()
    model_server.eval()
    
    # Tokenize the context
    context_tokens, _ = encoder.encode(context)
    
    # Add BOS token to signal start of completion
    # GPT-2 uses token 50256 as both BOS and EOS
    bos_token = 50256
    input_ids = context_tokens + [bos_token]
    
    generated = input_ids.copy()
    
    print(f"Context tokens: {len(context_tokens)}")
    print(f"Starting generation with {len(generated)} tokens...")
    
    with torch.no_grad():
        for step in range(max_length):
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
            
            # Get next token logits (only use the last position)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for end of sequence
            if next_token == 50256:  # EOS token
                print(f"Generated {step} tokens (reached EOS)")
                break
            
            generated.append(next_token)
    
    # Decode the full sequence
    full_text = encoder.decode(generated)
    
    # Extract just the generated completion (after the context)
    # The context + BOS should be skipped
    context_plus_bos = encoder.decode(input_ids)
    if full_text.startswith(context_plus_bos):
        completion = full_text[len(context_plus_bos):].strip()
    else:
        # Fallback: decode only the generated tokens
        completion = encoder.decode(generated[len(input_ids):]).strip()
    
    return completion, full_text


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


def main():
    parser = argparse.ArgumentParser(
        description="E2E NLG inference with split GPT-2 LoRA adapters"
    )
    parser.add_argument(
        "--lora_adapters",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        required=True,
        help="Path to base GPT-2 checkpoint",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="../vocab",
        help="Path to vocabulary directory (default: ../vocab)",
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default="gpt2.sm",
        choices=["gpt2.sm", "gpt2.md", "gpt2.lg"],
    )
    parser.add_argument(
        "--client_layers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--client_device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--server_device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower = more focused)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--context",
        type=str,
        help="E2E context to generate from (e.g., 'name : The Golden Dragon | Type : restaurant | ...')",
    )
    
    args = parser.parse_args()
    
    # Initialize encoder
    if not ENCODER_AVAILABLE:
        print("ERROR: Encoder not available. Please check that encoder.py is accessible.")
        return
    
    print(f"Loading encoder from: {args.vocab_path}")
    encoder = get_encoder(args.vocab_path)
    
    device_client = torch.device(args.client_device)
    device_server = torch.device(args.server_device)
    
    # Load adapter checkpoint to get config
    adapter_checkpoint = torch.load(args.lora_adapters, map_location="cpu")
    config = build_config_from_checkpoint(adapter_checkpoint, args)
    
    # Initialize models
    print(f"\nInitializing {args.model_card} split model...")
    print(f"  Client layers: {config.client_layers}")
    print(f"  Server layers: {config.n_layer - config.client_layers}")
    
    model_client = GPT2LMModel_Client(config).to(device_client)
    model_server = GPT2LMModel_Server(config).to(device_server)
    
    # Load base checkpoint
    print(f"\nLoading base model from: {args.init_checkpoint}")
    base_checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    model_client.load_weight(base_checkpoint)
    model_server.load_weight(base_checkpoint)
    
    # Test contexts
    test_contexts = [
        "name : The Golden Dragon | Type : restaurant | food : Chinese | price : moderate",
        "name : Cafe Mocha | Type : coffee shop | food : Fast food | price : cheap | customer rating : 4 out of 5 | area : riverside",
        "name : The Kings Head | Type : pub | price : more than £30 | customer rating : high | near : Train Station",
    ]
    
    if args.context:
        test_contexts = [args.context]
    
    print("\n" + "="*80)
    print("E2E NLG GENERATION - COMPARING BASE vs FINE-TUNED MODEL")
    print("="*80)
    
    for context in test_contexts:
        print(f"\n{'─'*80}")
        print(f"CONTEXT: {context}")
        print(f"{'─'*80}")
        
        # Generate with BASE model (no LoRA)
        print("\n[BASE MODEL]")
        try:
            completion_base, full_base = generate_text_e2e(
                model_client, model_server, encoder, context,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                device_client=device_client,
                device_server=device_server,
            )
            print(f"Completion: {completion_base}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Load LoRA adapters
        load_lora_adapters(args.lora_adapters, model_client, model_server)
        
        # Generate with FINE-TUNED model
        print("\n[FINE-TUNED MODEL WITH LORA]")
        try:
            completion_ft, full_ft = generate_text_e2e(
                model_client, model_server, encoder, context,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                device_client=device_client,
                device_server=device_server,
            )
            print(f"Completion: {completion_ft}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Reload base weights for next comparison
        model_client.load_weight(base_checkpoint)
        model_server.load_weight(base_checkpoint)
    
    print("\n" + "="*80)
    print("If fine-tuning worked properly, the fine-tuned model should generate")
    print("natural restaurant descriptions similar to the training data.")
    print("="*80)


if __name__ == "__main__":
    main()
