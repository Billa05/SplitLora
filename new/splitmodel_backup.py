#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024, FDU
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the FDU nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict 
import copy
import math

import torch
from torch import nn
from transformers import GPT2Config as HF_GPT2Config, GPT2Model, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, PeftModel


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class GPT2SplitPart(nn.Module):
    def __init__(self, config, start_layer, end_layer, has_embeddings=False, has_lm_head=False, lora_config=None):
        super().__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.has_embeddings = has_embeddings
        self.has_lm_head = has_lm_head
        
        # Load base GPT-2 model
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
        if self.has_embeddings:
            # First part: process from input_ids
            outputs = self.base_model(input_ids=input_ids, past_key_values=past_key_values, output_hidden_states=True, **kwargs)
            return outputs.hidden_states[-1], outputs.past_key_values
        elif self.has_lm_head:
            # Last part: process hidden_states to logits
            outputs = self.base_model(inputs_embeds=hidden_states, past_key_values=past_key_values, labels=labels, **kwargs)
            return outputs.logits, outputs.loss
        else:
            # Middle part: process hidden_states
            outputs = self.base_model(inputs_embeds=hidden_states, past_key_values=past_key_values, output_hidden_states=True, **kwargs)
            return outputs.hidden_states[-1], outputs.past_key_values

# Config for LoRA
def get_lora_config(r=8, alpha=16, dropout=0.0):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]  # Adjust for GPT-2 attention
    )