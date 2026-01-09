"""
Split GPT-2 model for distributed training with LoRA adapters.
CORRECTED VERSION: Only processes assigned layers, not the entire model.
"""

import torch
from torch import nn
from transformers import GPT2LMHeadModel
from peft import LoraConfig


class GPT2SplitPart(nn.Module):
    """
    Split part of GPT-2 model for distributed training.
    
    This implementation only processes the layers assigned to this device,
    not the entire model.
    
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
        self.config = config
        
        # Load pretrained GPT-2 model to extract components
        print(f"Loading GPT-2 components for layers {start_layer}-{end_layer}...")
        full_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Extract only the components we need
        if has_embeddings:
            self.wte = full_model.transformer.wte  # Token embeddings
            self.wpe = full_model.transformer.wpe  # Position embeddings
            self.drop = full_model.transformer.drop  # Dropout
            print(f"  - Loaded embeddings")
        
        # Extract only assigned transformer layers
        self.h = nn.ModuleList([
            full_model.transformer.h[i] for i in range(start_layer, end_layer)
        ])
        print(f"  - Loaded {len(self.h)} transformer layers")
        
        if has_lm_head:
            self.ln_f = full_model.transformer.ln_f  # Final layer norm
            self.lm_head = full_model.lm_head  # Language model head
            print(f"  - Loaded LM head")
        
        # Delete full model to free memory
        del full_model
        
        # Apply LoRA to the extracted layers
        if lora_config:
            print(f"  - Applying LoRA (r={lora_config.r}, alpha={lora_config.lora_alpha})...")
            for i, layer in enumerate(self.h):
                # Apply LoRA to attention projections
                layer.attn.c_attn = self._wrap_with_lora(
                    layer.attn.c_attn, lora_config
                )
                layer.attn.c_proj = self._wrap_with_lora(
                    layer.attn.c_proj, lora_config
                )
            print(f"  - LoRA applied to {len(self.h)} layers")
    
    def _wrap_with_lora(self, conv1d_layer, lora_config):
        """Wrap a Conv1D layer with LoRA adaptation."""
        import torch.nn.functional as F
        
        class LoRAConv1D(nn.Module):
            def __init__(self, base_layer, r, alpha, dropout):
                super().__init__()
                self.base_layer = base_layer
                self.r = r
                self.alpha = alpha
                self.scaling = alpha / r
                
                # Freeze base layer
                for param in self.base_layer.parameters():
                    param.requires_grad = False
                
                # GPT-2's Conv1D has weight shape (nx, nf) where nx=in_features, nf=out_features
                # Get dimensions from the Conv1D layer
                nx = base_layer.weight.shape[0]  # input features
                nf = base_layer.weight.shape[1]  # output features
                
                # LoRA parameters
                self.lora_A = nn.Parameter(torch.randn(nx, r) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(r, nf))
                self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            
            def forward(self, x):
                # Base output (Conv1D applies: x @ weight + bias)
                result = self.base_layer(x)
                # LoRA adaptation
                lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
                return result + lora_out
        
        return LoRAConv1D(
            conv1d_layer, 
            lora_config.r, 
            lora_config.lora_alpha, 
            lora_config.lora_dropout
        )
    
    def forward(self, input_ids=None, hidden_states=None, attention_mask=None, labels=None, mask=None, **kwargs):
        """
        Forward pass through ONLY this device's portion of the model.
        
        Args:
            input_ids: Input token IDs (for first device with embeddings)
            hidden_states: Hidden states from previous device
            attention_mask: Attention mask
            labels: Target labels for loss computation
            mask: Loss mask to ignore padding tokens
            
        Returns:
            Tuple of (output, auxiliary_info)
        """
        # First device: embed tokens
        if self.has_embeddings:
            if input_ids is None:
                raise ValueError("input_ids must be provided for first device")
            
            # Token embeddings
            inputs_embeds = self.wte(input_ids)
            
            # Position embeddings
            seq_length = input_ids.size(1)
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeds = self.wpe(position_ids)
            
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.drop(hidden_states)
        
        if hidden_states is None:
            raise ValueError("Either input_ids or hidden_states must be provided")
        
        # Process through ONLY assigned layers
        for i, layer in enumerate(self.h):
            actual_layer_idx = self.start_layer + i
            # Each layer returns (hidden_states, present_key_value, attention_weights)
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
        
        # Last device: compute logits and loss
        if self.has_lm_head:
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                # Shift for causal LM: predict next token
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Calculate loss with optional masking
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Apply mask if provided to ignore padding
                if mask is not None:
                    shift_mask = mask[..., 1:].contiguous()
                    loss = loss * shift_mask.view(-1)
                    loss = loss.sum() / (shift_mask.sum() + 1e-8)
                else:
                    loss = loss.mean()
            
            return logits, loss
        
        # Middle/first device: return hidden states
        return hidden_states, None


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
