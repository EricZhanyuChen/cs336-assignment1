"""
Ablation: Transformer without RoPE (NoPE)
Based on cs336_basics/transformer.py - removes Rotary Positional Embeddings
"""
import torch
import math
import numpy as np
import os
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import BinaryIO, IO, Optional, Iterable

from cs336_basics.transformer import (
    Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding,
    softmax, scaled_dot_product_attention,
    MultiheadSelfAttention, MultiheadSelfAttentionWithRope,
    AdamW, lr_cosine_schedule, gradient_clipping,
    get_batch, save_checkpoint, load_checkpoint, cross_entropy
)


class MultiheadSelfAttentionWithoutRope(torch.nn.Module):
    """Multi-head self-attention without RoPE positional embeddings."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.Q_weights = Linear(d_model, d_model)
        self.K_weights = Linear(d_model, d_model)
        self.V_weights = Linear(d_model, d_model)
        self.O_weights = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads

    def forward(self, in_features, max_seq_len, theta, token_positions=None):
        Q, K, V = self.Q_weights(in_features), self.K_weights(in_features), self.V_weights(in_features)
        Q, K, V = Q.float(), K.float(), V.float()

        seq_len = Q.shape[-2]
        pre_dim = Q.shape[:-2]

        Q_reshaped = Q.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K_reshaped = K.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V_reshaped = V.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        # No RoPE - use Q and K directly
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        attn_output = scaled_dot_product_attention(Q_reshaped, K_reshaped, V_reshaped, mask)

        attn_transposed = attn_output.transpose(-3, -2).reshape(*pre_dim, seq_len, self.d_model)
        return self.O_weights(attn_transposed)


class TransformerBlockNoPE(torch.nn.Module):
    """Pre-norm Transformer block using attention without RoPE."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, eps: float, weights: Optional[dict[str, Tensor]] = None):
        super().__init__()
        self.rmsnorm_1 = RMSNorm(d_model, eps)
        self.rmsnorm_2 = RMSNorm(d_model, eps)
        self.mha = MultiheadSelfAttentionWithoutRope(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)

        with torch.no_grad():
            if weights is not None and len(weights) > 0:
                self.rmsnorm_1.load_state_dict({"W": weights["ln1.weight"]})
                self.rmsnorm_2.load_state_dict({"W": weights["ln2.weight"]})
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

    def forward(self, in_features, max_seq_len, theta, token_positions=None):
        # Pre-norm style (same as original transformer.py)
        rms1_output = self.rmsnorm_1(in_features)
        mha_output = self.mha(rms1_output, max_seq_len, theta, token_positions)
        residual_1 = in_features + mha_output

        rms2_output = self.rmsnorm_2(residual_1)
        ffn_output = self.ffn(rms2_output)
        return ffn_output + residual_1


class TransformerNoPE(torch.nn.Module):
    """Full Transformer LM without positional embeddings (NoPE)."""
    def __init__(self, d_model, num_layers, num_heads, d_ff, vocab_size, eps, weights: Optional[dict[str, Tensor]] = None):
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
            block = TransformerBlockNoPE(d_model, num_heads, d_ff, eps, block_weights)
            self.layers.append(block)

        self.final_rmsnorm = RMSNorm(d_model, eps)
        self.output_embedding = Linear(d_model, vocab_size)

        with torch.no_grad():
            if len(weights) > 0:
                self.final_rmsnorm.load_state_dict({"W": weights["ln_final.weight"]})
                self.token_embedding.embedding.copy_(weights["token_embeddings.weight"])
                self.output_embedding.W.copy_(weights["lm_head.weight"])

    def forward(self, in_indices, context_length, rope_theta):
        x = self.token_embedding(in_indices)
        for layer in self.layers:
            x = layer(x, context_length, rope_theta)
        x = self.final_rmsnorm(x)
        logits = self.output_embedding(x)
        return logits
