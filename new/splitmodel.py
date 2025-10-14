"""
Split GPT-2 model for distributed training with LoRA adapters.
"""

import torch
from torch import nn
from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig


class GPT2SplitPart(nn.Module):
    """
    Split part of GPT-2 model for distributed training.
    
    Args:
        config: GPT2Config object
        start_layer: Starting layer index for this device
        end_layer: Ending layer index for this device
        has_embeddings: Whether this device has the embedding layer
        has_lm_head: Whether this device has the LM head
        lora_config: LoRA configuration
    """
    
    def __init__(self, config, start_layer, end_layer, has_embeddings=False, has_lm_head=False, lora_config=None):
        super().__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.has_embeddings = has_embeddings
        self.has_lm_head = has_lm_head
        
        # Load pretrained GPT-2 model
        self.base_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.base_model.config = config
        
        # Freeze all parameters initially
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the layers in this part
        if has_embeddings:
            for param in self.base_model.transformer.wte.parameters():
                param.requires_grad = True
            for param in self.base_model.transformer.wpe.parameters():
                param.requires_grad = True
        
        for i in range(start_layer, end_layer):
            for param in self.base_model.transformer.h[i].parameters():
                param.requires_grad = True
        
        if has_lm_head:
            for param in self.base_model.transformer.ln_f.parameters():
                param.requires_grad = True
            for param in self.base_model.lm_head.parameters():
                param.requires_grad = True
        
        # Apply LoRA to trainable parts
        if lora_config:
            self.base_model = get_peft_model(self.base_model, lora_config)
    
    def forward(self, input_ids=None, hidden_states=None, past_key_values=None, labels=None, **kwargs):
        """
        Forward pass through this device's portion of the model.
        
        Args:
            input_ids: Input token IDs (for first device with embeddings)
            hidden_states: Hidden states from previous device
            past_key_values: Cached key-value pairs for generation
            labels: Target labels for loss computation
            
        Returns:
            Tuple of (output, auxiliary_info)
        """
        if self.has_embeddings:
            # First part: process from input_ids
            outputs = self.base_model(
                input_ids=input_ids, 
                past_key_values=past_key_values, 
                output_hidden_states=True, 
                **kwargs
            )
            return outputs.hidden_states[-1], outputs.past_key_values
            
        elif self.has_lm_head:
            # Last part: process hidden_states to logits
            outputs = self.base_model(
                inputs_embeds=hidden_states, 
                past_key_values=past_key_values, 
                labels=labels, 
                **kwargs
            )
            return outputs.logits, outputs.loss
            
        else:
            # Middle part: process hidden_states
            outputs = self.base_model(
                inputs_embeds=hidden_states, 
                past_key_values=past_key_values, 
                output_hidden_states=True, 
                **kwargs
            )
            return outputs.hidden_states[-1], outputs.past_key_values


def get_lora_config(r=8, alpha=16, dropout=0.0):
    """
    Create LoRA configuration.
    
    Args:
        r: LoRA rank (number of low-rank matrices)
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate for LoRA layers
        
    Returns:
        LoraConfig object
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]  # GPT-2 attention modules
    )
