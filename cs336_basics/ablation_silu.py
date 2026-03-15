import torch
import math
import numpy as np
import os
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import BinaryIO, IO, Optional, Iterable

# Import all components from the original transformer
from cs336_basics.transformer import (
    Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding,
    softmax, scaled_dot_product_attention,
    MultiheadSelfAttention, MultiheadSelfAttentionWithRope,
    AdamW, lr_cosine_schedule, gradient_clipping,
    get_batch, save_checkpoint, load_checkpoint, cross_entropy
)

class SiLU(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.w2 = Linear(d_ff, d_model, device=self.device, dtype=self.dtype)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu_w1x = w1_x * torch.sigmoid(w1_x)

        output = self.w2(silu_w1x)
        return output
    
class TransformerBlockWithSiLU(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            weights: Optional[dict[str, Tensor]]=None,
    ):
        super().__init__()
        self.rms_1 = RMSNorm(d_model)
        self.rms_2 = RMSNorm(d_model)

        self.mha = MultiheadSelfAttentionWithRope(d_model, num_heads)
        self.ffn = SiLU(d_model, d_ff)

        with torch.no_grad():
            if weights is not None and len(weights) > 0:
                self.rms_1.load_state_dict({"W": weights["ln1.weight"]})
                self.rms_2.load_state_dict({"W": weights["ln2.weight"]})

                self.mha.load_state_dict({
                "Q_weights.W": weights["attn.q_proj.weight"],
                "K_weights.W": weights["attn.k_proj.weight"],
                "V_weights.W": weights["attn.v_proj.weight"],
                "O_weights.W": weights["attn.output_proj.weight"]
            })
                self.ffn.load_state_dict({
                "w1.W": weights["ffn.w1.weight"],
                "w2.W": weights["ffn.w2.weight"]
            })
                    

    def forward(self, in_features, max_seq_len, theta, token_positions: Tensor| None = None ):
        rms1_output = self.rms_1(in_features)
        mha_output = self.mha(rms1_output, max_seq_len, theta, token_positions)

        residual_1 = in_features + mha_output

        rms2_output = self.rms_2(residual_1)
        ffn_output = self.ffn(rms2_output)

        return ffn_output + residual_1
    
class TransformerWithSiLU(torch.nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, vocab_size, weights: Optional[dict[str, Tensor]]=None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.token_embedding = Embedding(vocab_size, d_model)
        if weights is None:
            weights = {}
        for i in range(num_layers):
            prefix = f"layers.{i}."
            block_weights = {
                k.replace(prefix, ""): v for k, v in weights.items() if k.startswith(prefix)
            }

            transformer_block = TransformerBlockWithSiLU(d_model, num_heads, d_ff, block_weights)

            self.layers.append(transformer_block)

        self.final_rms = RMSNorm(d_model)

        self.output_embedding = Linear(d_model, vocab_size)

        with torch.no_grad():
            if len(weights) > 0:
                self.final_rms.W.copy_(weights["ln_final.weight"])
                self.token_embedding.embedding.copy_(weights["token_embeddings.weight"])
                self.output_embedding.W.copy_(weights["lm_head.weight"])

    def forward(self, in_indices, context_length, rope_theta):
        x = self.token_embedding(in_indices)
        for layer in self.layers:
            x = layer(x, context_length, rope_theta)

        x = self.final_rms(x)
        logits = self.output_embedding(x)

        return logits
