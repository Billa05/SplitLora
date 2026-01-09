#!/usr/bin/env python3
"""
Check LoRA adapter parameters to verify training progress.
This script helps diagnose if the model actually learned anything.
"""

import torch
import sys
from pathlib import Path


def check_lora_adapter(device_id, state_dict):
    """Check LoRA parameters for a single device."""
    print(f"\n{'='*70}")
    print(f"  DEVICE {device_id} LoRA ADAPTER ANALYSIS")
    print(f"{'='*70}\n")
    
    # Get all LoRA parameter keys
    lora_a_keys = [k for k in state_dict.keys() if 'lora_A' in k]
    lora_b_keys = [k for k in state_dict.keys() if 'lora_B' in k]
    
    print(f"Found {len(lora_a_keys)} LoRA_A matrices and {len(lora_b_keys)} LoRA_B matrices")
    print()
    
    # Analyze LoRA_A parameters (should stay relatively small)
    print("LoRA_A Statistics (initialized randomly, should change slightly):")
    print("-" * 70)
    a_stats = []
    for key in lora_a_keys[:3]:  # Show first 3
        tensor = state_dict[key]
        stats = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'max': tensor.max().item(),
            'min': tensor.min().item(),
        }
        a_stats.append(stats)
        print(f"  {key}")
        print(f"    Shape: {tensor.shape}")
        print(f"    Mean: {stats['mean']:>10.6f}  |  Std: {stats['std']:>10.6f}")
        print(f"    Min:  {stats['min']:>10.6f}  |  Max: {stats['max']:>10.6f}")
        print()
    
    # Analyze LoRA_B parameters (initialized to zero, should grow during training)
    print("LoRA_B Statistics (initialized to zero, SHOULD BE NON-ZERO after training):")
    print("-" * 70)
    b_stats = []
    all_close_to_zero = True
    
    for i, key in enumerate(lora_b_keys):
        tensor = state_dict[key]
        stats = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'max': tensor.max().item(),
            'min': tensor.min().item(),
            'abs_max': tensor.abs().max().item(),
        }
        b_stats.append(stats)
        
        # Check if this tensor has significant values
        if stats['abs_max'] > 0.01:
            all_close_to_zero = False
        
        # Show first 3 in detail
        if i < 3:
            print(f"  {key}")
            print(f"    Shape: {tensor.shape}")
            print(f"    Mean: {stats['mean']:>10.6f}  |  Std: {stats['std']:>10.6f}")
            print(f"    Min:  {stats['min']:>10.6f}  |  Max: {stats['max']:>10.6f}")
            print(f"    Abs Max: {stats['abs_max']:>10.6f}")
            print()
    
    # Summary statistics across all LoRA_B matrices
    avg_std = sum(s['std'] for s in b_stats) / len(b_stats)
    avg_abs_max = sum(s['abs_max'] for s in b_stats) / len(b_stats)
    max_abs_max = max(s['abs_max'] for s in b_stats)
    
    print(f"  Summary across all {len(b_stats)} LoRA_B matrices:")
    print(f"    Average Std:    {avg_std:>10.6f}")
    print(f"    Average |Max|:  {avg_abs_max:>10.6f}")
    print(f"    Largest |Max|:  {max_abs_max:>10.6f}")
    print()
    
    # Diagnosis
    print("DIAGNOSIS:")
    print("-" * 70)
    
    if all_close_to_zero or avg_std < 0.001:
        print("  ❌ PROBLEM: LoRA_B parameters are too small!")
        print("     The model has NOT learned anything meaningful.")
        print()
        print("  Possible causes:")
        print("     • Learning rate too low (should be ~1e-4 for LoRA)")
        print("     • Not enough training steps")
        print("     • Gradients not flowing properly")
        print("     • Optimizer not updating parameters")
        status = "NOT_TRAINED"
    elif avg_std < 0.01:
        print("  ⚠️  WARNING: LoRA_B parameters are quite small")
        print("     The model has learned a little, but probably not enough.")
        print()
        print("  Recommendations:")
        print("     • Train for more steps")
        print("     • Consider increasing learning rate")
        status = "UNDER_TRAINED"
    elif avg_std < 0.05:
        print("  ✓ OK: LoRA_B parameters show some learning")
        print("     The model has learned, but could benefit from more training.")
        print()
        print("  Recommendations:")
        print("     • Current training is working")
        print("     • More steps would improve quality")
        status = "PARTIALLY_TRAINED"
    else:
        print("  ✓✓ GOOD: LoRA_B parameters show strong learning")
        print("     The model has learned meaningful patterns.")
        print()
        print("  Status: Training appears successful!")
        status = "WELL_TRAINED"
    
    return status, avg_std, avg_abs_max


def compare_with_expected():
    """Compare current values with expected ranges."""
    print("\n" + "="*70)
    print("  EXPECTED VALUES (for reference)")
    print("="*70 + "\n")
    
    print("LoRA_B Standard Deviation ranges:")
    print("  • NOT trained:      0.0001 - 0.001  (essentially still at zero)")
    print("  • UNDER-trained:    0.001  - 0.01   (some learning, not enough)")
    print("  • PARTIALLY trained: 0.01   - 0.05   (decent learning)")
    print("  • WELL trained:     0.05   - 0.2    (good learning)")
    print("  • OVER-trained:     > 0.2           (might be overfitting)")
    print()


def main():
    print("\n" + "="*70)
    print("  LoRA ADAPTER PARAMETER CHECKER")
    print("="*70)
    
    # Check if files exist
    device_0_path = Path("./lora_device_0.pth")
    device_1_path = Path("./lora_device_1.pth")
    
    if not device_0_path.exists() or not device_1_path.exists():
        print("\n❌ ERROR: LoRA adapter files not found!")
        print(f"   Looking for: {device_0_path.absolute()}")
        print(f"            and: {device_1_path.absolute()}")
        print("\n   Please train the model first using: ./train.sh")
        sys.exit(1)
    
    # Load both adapters
    print("\nLoading LoRA adapters...")
    state_0 = torch.load(device_0_path, map_location='cpu')
    state_1 = torch.load(device_1_path, map_location='cpu')
    print("✓ Loaded successfully")
    
    # Analyze each device
    status_0, std_0, max_0 = check_lora_adapter(0, state_0)
    status_1, std_1, max_1 = check_lora_adapter(1, state_1)
    
    # Show expected values
    compare_with_expected()
    
    # Overall summary
    print("\n" + "="*70)
    print("  OVERALL SUMMARY")
    print("="*70 + "\n")
    
    print(f"Device 0: {status_0:15s} (avg std: {std_0:.6f}, avg |max|: {max_0:.6f})")
    print(f"Device 1: {status_1:15s} (avg std: {std_1:.6f}, avg |max|: {max_1:.6f})")
    print()
    
    # Overall recommendation
    if status_0 == "NOT_TRAINED" or status_1 == "NOT_TRAINED":
        print("❌ OVERALL STATUS: NOT TRAINED")
        print("\nRECOMMENDATIONS:")
        print("  1. Increase learning rate to 1e-4 (currently might be too low)")
        print("  2. Train for at least 10,000 steps (1 epoch)")
        print("  3. Check that gradients are flowing properly")
        print("  4. Verify optimizer is actually updating parameters")
        print("\n  Run: ./train.sh")
    elif status_0 == "UNDER_TRAINED" or status_1 == "UNDER_TRAINED":
        print("⚠️  OVERALL STATUS: UNDER-TRAINED")
        print("\nRECOMMENDATIONS:")
        print("  1. Train for more steps (try 10,000-20,000 steps)")
        print("  2. Current learning rate seems okay, just needs more time")
        print("\n  Run: ./train.sh")
    elif status_0 == "PARTIALLY_TRAINED" or status_1 == "PARTIALLY_TRAINED":
        print("✓ OVERALL STATUS: PARTIALLY TRAINED")
        print("\nRECOMMENDATIONS:")
        print("  • Model should generate reasonable text")
        print("  • More training would improve quality")
        print("  • Test with: python infer_2device.py --prompt \"...<|endoftext|>\"")
    else:
        print("✓✓ OVERALL STATUS: WELL TRAINED")
        print("\nRECOMMENDATIONS:")
        print("  • Model should generate good quality text")
        print("  • Test with: python infer_2device.py --prompt \"...<|endoftext|>\"")
        print("  • Can continue training for even better quality if desired")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
