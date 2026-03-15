"""
Ablation: Transformer without RMSNorm
Based on cs336_basics/transformer.py - removes all RMSNorm layers
"""
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


class TransformerBlockNoRMSNorm(torch.nn.Module):
    """Transformer block with RMSNorm removed."""
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            weights: Optional[dict[str, Tensor]] = None,
    ):
        super().__init__()
        # NO RMSNorm - removed for ablation
        self.mha = MultiheadSelfAttentionWithRope(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)

        with torch.no_grad():
            if weights is not None and len(weights) > 0:
                # Skip loading RMSNorm weights
                self.mha.load_state_dict({
                    "Q_weights.W": weights["attn.q_proj.weight"],
                    "K_weights.W": weights["attn.k_proj.weight"],
                    "V_weights.W": weights["attn.v_proj.weight"],
                    "O_weights.W": weights["attn.output_proj.weight"]
                })
                self.ffn.load_state_dict({
                    "w1.W": weights["ffn.w1.weight"],
                    "w2.W": weights["ffn.w2.weight"],
                    "w3.W": weights["ffn.w3.weight"]
                })

    def forward(self, in_features, max_seq_len, theta, token_positions: Tensor | None = None):
        # Skip RMSNorm before MHA - pass input directly
        mha_output = self.mha(in_features, max_seq_len, theta, token_positions)
        residual_1 = in_features + mha_output

        # Skip RMSNorm before FFN - pass residual directly
        ffn_output = self.ffn(residual_1)
        return ffn_output + residual_1


class TransformerNoRMSNorm(torch.nn.Module):
    """Full Transformer LM with all RMSNorms removed."""
    def __init__(self, d_model, num_layers, num_heads, d_ff, vocab_size, weights: Optional[dict[str, Tensor]] = None):
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
            transformer_block = TransformerBlockNoRMSNorm(d_model, num_heads, d_ff, block_weights)
            self.layers.append(transformer_block)

        # NO final RMSNorm
        self.output_embedding = Linear(d_model, vocab_size)

        with torch.no_grad():
            if len(weights) > 0:
                self.token_embedding.embedding.copy_(weights["token_embeddings.weight"])
                self.output_embedding.W.copy_(weights["lm_head.weight"])

    def forward(self, in_indices, context_length, rope_theta):
        x = self.token_embedding(in_indices)
        for layer in self.layers:
            x = layer(x, context_length, rope_theta)

        # Skip final RMSNorm - pass directly to output embedding
        logits = self.output_embedding(x)
        return logits
