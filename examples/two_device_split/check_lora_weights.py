#!/usr/bin/env python3
"""
Script to analyze LoRA adapter weights and determine training quality.
This helps you understand if your model is undertrained, well-trained, or overtrained.
"""
import argparse
import torch
import numpy as np
from pathlib import Path


def analyze_lora_weights(checkpoint_path: str):
    """Analyze LoRA adapter weights to assess training quality."""
    
    print("="*80)
    print(f"ANALYZING LORA ADAPTERS: {checkpoint_path}")
    print("="*80)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print(f"\nCheckpoint Info:")
    print(f"  Tag: {checkpoint.get('tag', 'N/A')}")
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"  Model: {config.get('n_layer', 'N/A')} layers, {config.get('n_embd', 'N/A')} dims")
        print(f"  Split: {config.get('client_layers', 'N/A')} client / {config.get('n_layer', 12) - config.get('client_layers', 6)} server layers")
    
    # Analyze client and server separately
    partitions = {
        'CLIENT': checkpoint.get('client', {}),
        'SERVER': checkpoint.get('server', {})
    }
    
    all_stats = []
    
    for partition_name, state_dict in partitions.items():
        if not state_dict:
            continue
            
        print(f"\n{'─'*80}")
        print(f"{partition_name} PARTITION")
        print(f"{'─'*80}")
        
        lora_a_weights = []
        lora_b_weights = []
        
        # Collect all LoRA weights
        for name, param in state_dict.items():
            if 'lora_A' in name:
                lora_a_weights.append(param.flatten())
            elif 'lora_B' in name:
                lora_b_weights.append(param.flatten())
        
        if not lora_a_weights and not lora_b_weights:
            print(f"  ⚠️  No LoRA weights found in {partition_name}")
            continue
        
        # Concatenate all weights
        if lora_a_weights:
            all_lora_a = torch.cat(lora_a_weights)
        if lora_b_weights:
            all_lora_b = torch.cat(lora_b_weights)
        
        # Calculate statistics
        print(f"\n📊 Weight Statistics:")
        
        if lora_a_weights:
            print(f"\n  LoRA A (down-projection) matrices:")
            print(f"    • Total parameters: {all_lora_a.numel():,}")
            print(f"    • Mean (absolute):  {all_lora_a.abs().mean().item():.6f}")
            print(f"    • Std deviation:    {all_lora_a.std().item():.6f}")
            print(f"    • Max (absolute):   {all_lora_a.abs().max().item():.6f}")
            print(f"    • % non-zero:       {(all_lora_a != 0).float().mean().item() * 100:.2f}%")
            
        if lora_b_weights:
            print(f"\n  LoRA B (up-projection) matrices:")
            print(f"    • Total parameters: {all_lora_b.numel():,}")
            print(f"    • Mean (absolute):  {all_lora_b.abs().mean().item():.6f}")
            print(f"    • Std deviation:    {all_lora_b.std().item():.6f}")
            print(f"    • Max (absolute):   {all_lora_b.abs().max().item():.6f}")
            print(f"    • % non-zero:       {(all_lora_b != 0).float().mean().item() * 100:.2f}%")
        
        # Combined analysis
        if lora_a_weights and lora_b_weights:
            all_weights = torch.cat([all_lora_a, all_lora_b])
            mean_abs = all_weights.abs().mean().item()
            
            print(f"\n  Combined LoRA weights:")
            print(f"    • Total parameters: {all_weights.numel():,}")
            print(f"    • Mean (absolute):  {mean_abs:.6f}")
            print(f"    • Std deviation:    {all_weights.std().item():.6f}")
            print(f"    • Max (absolute):   {all_weights.abs().max().item():.6f}")
            
            all_stats.append({
                'partition': partition_name,
                'mean_abs': mean_abs,
                'total_params': all_weights.numel()
            })
            
            # Training quality assessment
            print(f"\n  🎯 Training Quality Assessment:")
            if mean_abs < 0.0001:
                status = "❌ NOT TRAINED"
                desc = "Weights are essentially zero - training likely failed"
                color = "red"
            elif mean_abs < 0.001:
                status = "⚠️  MINIMALLY TRAINED"
                desc = "Very little learning - needs much more training"
                color = "orange"
            elif mean_abs < 0.01:
                status = "⚡ UNDER-TRAINED"
                desc = "Some learning occurred, but needs more epochs/data"
                color = "yellow"
            elif mean_abs < 0.05:
                status = "✅ PARTIALLY TRAINED"
                desc = "Decent learning - should work for simple tasks"
                color = "lightgreen"
            elif mean_abs < 0.2:
                status = "🎉 WELL TRAINED"
                desc = "Good learning - model should perform well"
                color = "green"
            else:
                status = "🔥 HEAVILY TRAINED"
                desc = "Strong learning - check for overfitting"
                color = "red"
            
            print(f"    Status: {status}")
            print(f"    {desc}")
    
    # Overall assessment
    if all_stats:
        print(f"\n{'='*80}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*80}")
        
        avg_mean_abs = np.mean([s['mean_abs'] for s in all_stats])
        total_params = sum([s['total_params'] for s in all_stats])
        
        print(f"\n  Average mean absolute weight: {avg_mean_abs:.6f}")
        print(f"  Total LoRA parameters: {total_params:,}")
        
        print(f"\n  📈 Training Quality Scale:")
        print(f"  {'─'*76}")
        print(f"  │ NOT trained:        0.0001 - 0.001   (essentially still at zero)        │")
        print(f"  │ MINIMALLY trained:  0.001  - 0.01    (very little learning)             │")
        print(f"  │ UNDER-trained:      0.01   - 0.05    (some learning, needs more)        │")
        print(f"  │ PARTIALLY trained:  0.05   - 0.1     (decent learning)                  │")
        print(f"  │ WELL trained:       0.1    - 0.2     (good learning) ✅                 │")
        print(f"  │ HEAVILY trained:    > 0.2            (might be overfitting) ⚠️          │")
        print(f"  {'─'*76}")
        
        print(f"\n  Your model: {avg_mean_abs:.6f} ", end="")
        if avg_mean_abs < 0.0001:
            print("→ ❌ NOT TRAINED")
            print("\n  💡 Recommendation: Check if training ran properly. Loss should decrease.")
        elif avg_mean_abs < 0.001:
            print("→ ⚠️  MINIMALLY TRAINED")
            print("\n  💡 Recommendation: Train for many more epochs (10x current).")
        elif avg_mean_abs < 0.01:
            print("→ ⚡ UNDER-TRAINED")
            print("\n  💡 Recommendation: Train for 3-5x more epochs or increase learning rate.")
        elif avg_mean_abs < 0.05:
            print("→ ✅ PARTIALLY TRAINED")
            print("\n  💡 Recommendation: Model should work, but more training may improve quality.")
        elif avg_mean_abs < 0.1:
            print("→ ✅ WELL TRAINED")
            print("\n  💡 This looks good! Model should perform well on the task.")
        elif avg_mean_abs < 0.2:
            print("→ 🎉 WELL TRAINED")
            print("\n  💡 Excellent! Model has learned the task well.")
        else:
            print("→ 🔥 HEAVILY TRAINED")
            print("\n  💡 Check validation metrics. If val loss increased, you may be overfitting.")
    
    print("\n" + "="*80 + "\n")


def compare_checkpoints(checkpoint_paths: list):
    """Compare multiple checkpoints to see training progression."""
    
    print("="*80)
    print("COMPARING MULTIPLE CHECKPOINTS")
    print("="*80)
    
    results = []
    
    for path in checkpoint_paths:
        if not Path(path).exists():
            print(f"\n⚠️  Checkpoint not found: {path}")
            continue
        
        checkpoint = torch.load(path, map_location="cpu")
        tag = checkpoint.get('tag', Path(path).stem)
        
        all_weights = []
        for partition in ['client', 'server']:
            state_dict = checkpoint.get(partition, {})
            for name, param in state_dict.items():
                if 'lora_A' in name or 'lora_B' in name:
                    all_weights.append(param.flatten())
        
        if all_weights:
            combined = torch.cat(all_weights)
            mean_abs = combined.abs().mean().item()
            std = combined.std().item()
            max_abs = combined.abs().max().item()
            
            results.append({
                'path': path,
                'tag': tag,
                'mean_abs': mean_abs,
                'std': std,
                'max_abs': max_abs,
                'total_params': combined.numel()
            })
    
    if results:
        print(f"\n{'Checkpoint':<30} {'Tag':<15} {'Mean(abs)':<12} {'Std':<12} {'Max(abs)':<12} {'Status'}")
        print("─" * 110)
        
        for r in results:
            status = ""
            if r['mean_abs'] < 0.001:
                status = "❌ Not trained"
            elif r['mean_abs'] < 0.01:
                status = "⚡ Under-trained"
            elif r['mean_abs'] < 0.1:
                status = "✅ Partially trained"
            elif r['mean_abs'] < 0.2:
                status = "🎉 Well trained"
            else:
                status = "🔥 Heavily trained"
            
            print(f"{Path(r['path']).name:<30} {r['tag']:<15} {r['mean_abs']:<12.6f} {r['std']:<12.6f} {r['max_abs']:<12.6f} {status}")
        
        if len(results) > 1:
            print("\n📈 Training Progression:")
            for i in range(1, len(results)):
                change = results[i]['mean_abs'] - results[i-1]['mean_abs']
                pct_change = (change / results[i-1]['mean_abs']) * 100 if results[i-1]['mean_abs'] > 0 else 0
                direction = "↗️" if change > 0 else "↘️"
                print(f"  {results[i-1]['tag']} → {results[i]['tag']}: {direction} {change:+.6f} ({pct_change:+.1f}%)")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LoRA adapter weights to assess training quality"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a single LoRA checkpoint to analyze"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Paths to multiple checkpoints to compare"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all checkpoints in ./outputs directory"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Find all checkpoints in outputs
        output_dir = Path("./outputs")
        if output_dir.exists():
            checkpoints = sorted(output_dir.glob("lora_adapters_*.pt"))
            if checkpoints:
                print(f"Found {len(checkpoints)} checkpoint(s) in ./outputs/\n")
                compare_checkpoints([str(p) for p in checkpoints])
                print("\nDetailed analysis of final checkpoint:\n")
                if checkpoints:
                    analyze_lora_weights(str(checkpoints[-1]))
            else:
                print("No checkpoints found in ./outputs/")
        else:
            print("./outputs/ directory not found")
    
    elif args.compare:
        compare_checkpoints(args.compare)
    
    elif args.checkpoint:
        analyze_lora_weights(args.checkpoint)
    
    else:
        # Default: analyze final checkpoint if it exists
        default_path = "./outputs/lora_adapters_final.pt"
        if Path(default_path).exists():
            print("No arguments provided. Analyzing default checkpoint:\n")
            analyze_lora_weights(default_path)
        else:
            print("Error: No checkpoint specified and default not found.")
            print("\nUsage:")
            print("  python check_lora_weights.py --checkpoint ./outputs/lora_adapters_final.pt")
            print("  python check_lora_weights.py --all")
            print("  python check_lora_weights.py --compare ./outputs/lora_adapters_epoch*.pt")


if __name__ == "__main__":
    main()
