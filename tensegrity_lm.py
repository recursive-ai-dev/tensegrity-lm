"""
TensegrityLM — a byte-level decoder-only language model.

Single-file, checkpointable, trainable implementation combining:
  - Rotary positional embeddings (RoPE)              [Geometry]
  - RMSNorm pre-norm transformer blocks              [Algebraic Math]
  - Grouped-Query Attention (GQA)                    [Algorithms]
  - Alternating local sliding-window / global causal attention [Space]
  - SwiGLU feed-forward, alternating dense / sparse Mixture-of-Experts
    with Switch-style load-balancing loss            [Social Dynamics]
  - Learnable per-layer residual gain + activation-RMS
    "homeostatic" set-point regularization           [Biology]
  - Depth-scaled stochastic layer drop (LayerDrop)   [Evolution]
  - Cosine LR schedule with warmup ("temperature annealing") [Physics]
  - Optional confidence head for adaptive sampling temperature
                                                      [Psychology]
  - Byte-level tokenizer, zero external deps         [English Language]

Honesty note: every individual mechanism above is established in the
literature (RoPE, GQA, SwiGLU, MoE+load-balancing, RMSNorm, LayerDrop,
LayerScale-style gains, cosine schedules). The specific combination —
particularly the homeostatic residual-gain regularization framed as an
explicit activation-RMS set-point loss, paired with depth-scaled layer
drop and alternating local/global GQA — is a coherent but UNVALIDATED
recombination. Train it, measure it, don't assume it beats a vanilla
transformer of equal size until you've run the comparison yourself.

Usage:
    python tensegrity_lm.py prepare --data_path mydata.txt --out_dir data/mycorpus
    python tensegrity_lm.py train    --data_dir data/mycorpus --out_dir runs/run1
    python tensegrity_lm.py train    --data_dir data/mycorpus --out_dir runs/run1 --init_from resume
    python tensegrity_lm.py generate --ckpt runs/run1/ckpt_best.pt --prompt "Hello" --max_new_tokens 200
    python tensegrity_lm.py eval     --ckpt runs/run1/ckpt_best.pt --data_dir data/mycorpus
"""

import argparse
import dataclasses
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Tokenizer
# =============================================================================

class ByteTokenizer:
    """
    Byte-level tokenizer. Vocabulary = 256 raw byte values + 3 special tokens:
        256 = BOS (beginning of sequence)
        257 = EOS (end of sequence / document separator)
        258 = PAD (padding, unused by default but reserved)

    Zero training required, lossless for any UTF-8 (or arbitrary binary) input.
    """
    VOCAB_SIZE = 259
    BOS = 256
    EOS = 257
    PAD = 258

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))

    def encode_bytes(self, data: bytes) -> List[int]:
        return list(data)

    def decode(self, ids: List[int]) -> str:
        byte_vals = bytes(i for i in ids if 0 <= i <= 255)
        return byte_vals.decode("utf-8", errors="replace")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TensegrityConfig:
    # Core sizes
    vocab_size: int = ByteTokenizer.VOCAB_SIZE
    block_size: int = 256          # max sequence length / context window
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6                # query heads
    n_kv_head: int = 2              # key/value heads (n_head % n_kv_head == 0) -> GQA
    dropout: float = 0.0

    # Attention pattern (local/global alternation -> "Space" mapping)
    sliding_window: int = 128       # local attention window size
    global_every: int = 3           # every Nth layer (1-indexed) uses full causal attention

    # Feed-forward / MoE
    ffn_mult: float = 8.0 / 3.0     # SwiGLU hidden-dim multiplier (matches ~4x ReLU param count)
    ffn_align: int = 64             # round FFN hidden dim up to multiple of this
    moe_every: int = 2              # every Nth layer (1-indexed) uses MoE FFN instead of dense
    n_experts: int = 4
    moe_top_k: int = 2
    moe_aux_weight: float = 0.01    # load-balancing auxiliary loss weight

    # Homeostatic residual gain regularization ("Biology")
    homeo_target: float = 1.0       # target activation RMS for each sublayer output
    homeo_weight: float = 0.01      # weight of homeostasis loss term
    gain_init: Optional[float] = None  # if None, uses 1/sqrt(2*n_layer) (DeepNet-style)

    # Stochastic depth ("Evolution") — linear schedule, 0 at layer 1 -> layerdrop_max at layer n_layer
    layerdrop_max: float = 0.1

    # Confidence head ("Psychology") — predicts P(token correct) for adaptive sampling temperature
    use_confidence_head: bool = True
    confidence_weight: float = 0.05

    # Misc
    bias: bool = False               # use bias in Linear layers
    init_std: float = 0.02
    grad_checkpoint: bool = False    # gradient checkpointing for memory efficiency (training only)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        head_dim = self.n_embd // self.n_head
        assert head_dim % 2 == 0, "head_dim (n_embd // n_head) must be even for RoPE"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head for GQA"
        assert self.n_experts >= 1, "n_experts must be >= 1"
        assert 1 <= self.moe_top_k <= self.n_experts, "moe_top_k must be in [1, n_experts]"
        assert self.global_every >= 1
        assert self.moe_every >= 1
        assert 0.0 <= self.layerdrop_max < 1.0
        assert self.sliding_window >= 1
        if self.gain_init is None:
            self.gain_init = 1.0 / math.sqrt(2.0 * self.n_layer)


# =============================================================================
# Core building blocks
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute norm in fp32 for stability regardless of input dtype (e.g. under autocast)
        in_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_fp32 * rms).to(in_dtype)
        return out * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0,
                     device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for RoPE. Returns tensors of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to x.
    x:   (B, H, T, D)  where D is the head dimension (even)
    cos: (T, D // 2)
    sin: (T, D // 2)
    """
    B, H, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_ = cos[:T].view(1, 1, T, half)
    sin_ = sin[:T].view(1, 1, T, half)
    # Standard "rotate half" RoPE formulation
    out1 = x1 * cos_ - x2 * sin_
    out2 = x2 * cos_ + x1 * sin_
    return torch.cat([out1, out2], dim=-1)


def activation_rms(x: torch.Tensor) -> torch.Tensor:
    """Mean RMS of activations across the feature dimension, averaged over batch/tokens."""
    return x.float().pow(2).mean(dim=-1).sqrt().mean()


# =============================================================================
# Attention (Grouped-Query, with alternating local/global masks)
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TensegrityConfig, is_global: bool):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.is_global = is_global
        self.sliding_window = cfg.sliding_window
        self.dropout_p = cfg.dropout

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B, nH, T, D)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, nKV, T, D)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, nKV, T, D)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (B, nH, T, D)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # attn_mask: (T, T) bool, True = keep / attend, False = masked out
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,  # causality is encoded directly in attn_mask
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.o_proj(y)
        y = self.resid_dropout(y)
        return y


def build_attention_masks(block_size: int, sliding_window: int, device=None) -> Dict[str, torch.Tensor]:
    """
    Build boolean attention masks of shape (block_size, block_size).
    True = positions that may attend to each other (query i attends to key j).
    """
    idx = torch.arange(block_size, device=device)
    i = idx.view(-1, 1)
    j = idx.view(1, -1)
    causal = j <= i
    local = causal & (j > (i - sliding_window))
    return {"global": causal, "local": local}


# =============================================================================
# Feed-forward: SwiGLU (dense) and Mixture-of-Experts
# =============================================================================

def ffn_hidden_dim(n_embd: int, mult: float, align: int) -> int:
    hidden = int(n_embd * mult)
    hidden = align * ((hidden + align - 1) // align)
    return max(hidden, align)


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int, bias: bool, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MoEFeedForward(nn.Module):
    """
    Top-k routed Mixture of Experts with SwiGLU experts and a Switch-Transformer
    style load-balancing auxiliary loss.
    """

    def __init__(self, cfg: TensegrityConfig, hidden_dim: int):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.moe_top_k
        self.n_embd = cfg.n_embd
        self.aux_weight = cfg.moe_aux_weight

        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(cfg.n_embd, hidden_dim, cfg.bias, cfg.dropout) for _ in range(cfg.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # (N, C), N = B*T
        N = x_flat.shape[0]

        # Routing: compute in fp32 for numerical stability
        router_logits = self.router(x_flat).float()           # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)        # (N, E)

        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (N, K)
        # Re-normalize the top-k probabilities so they sum to 1 per token
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(x_flat)

        for e in range(self.n_experts):
            # tokens where expert e is among the chosen top-k
            expert_mask = (topk_idx == e)              # (N, K) bool
            if not expert_mask.any():
                continue
            token_idx, slot_idx = expert_mask.nonzero(as_tuple=True)  # indices into N and K
            if token_idx.numel() == 0:
                continue
            tokens_in = x_flat.index_select(0, token_idx)             # (M, C)
            expert_out = self.experts[e](tokens_in)                    # (M, C)
            weights = topk_probs[token_idx, slot_idx].unsqueeze(-1).to(expert_out.dtype)  # (M, 1)
            weighted = (expert_out * weights).to(out.dtype)
            out.index_add_(0, token_idx, weighted)

        out = out.reshape(B, T, C)

        # --- Load-balancing auxiliary loss (Switch Transformer formulation) ---
        # f_e = fraction of tokens for which expert e is in the top-1 (hard assignment proxy)
        # P_e = mean router probability assigned to expert e (soft, differentiable)
        with torch.no_grad():
            top1_idx = topk_idx[:, 0]  # (N,)
            one_hot = F.one_hot(top1_idx, num_classes=self.n_experts).float()  # (N, E)
            f = one_hot.mean(dim=0)  # (E,)
        P = router_probs.mean(dim=0)  # (E,) — carries gradient
        aux_loss = self.aux_weight * self.n_experts * (f * P).sum()

        return out, aux_loss


# =============================================================================
# Transformer block
# =============================================================================

class Block(nn.Module):
    def __init__(self, cfg: TensegrityConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        is_global = ((layer_idx + 1) % cfg.global_every == 0)
        is_moe = ((layer_idx + 1) % cfg.moe_every == 0)
        self.is_moe = is_moe

        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, is_global=is_global)
        self.ln2 = RMSNorm(cfg.n_embd)

        hidden_dim = ffn_hidden_dim(cfg.n_embd, cfg.ffn_mult, cfg.ffn_align)
        if is_moe:
            self.ffn = MoEFeedForward(cfg, hidden_dim)
        else:
            self.ffn = SwiGLU(cfg.n_embd, hidden_dim, cfg.bias, cfg.dropout)

        # Homeostatic residual gains (learnable per-layer, per-sublayer scalars)
        self.attn_gain = nn.Parameter(torch.tensor(float(cfg.gain_init)))
        self.ffn_gain = nn.Parameter(torch.tensor(float(cfg.gain_init)))
        self.homeo_target = cfg.homeo_target

        # Depth-scaled stochastic depth probability
        n_layer = cfg.n_layer
        if n_layer > 1:
            self.drop_p = cfg.layerdrop_max * (layer_idx / (n_layer - 1))
        else:
            self.drop_p = 0.0

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stochastic depth: during training, randomly skip the entire block (batch-level)
        if self.training and self.drop_p > 0.0:
            if torch.rand((), device=x.device).item() < self.drop_p:
                zero = x.new_zeros(())
                return x, zero, zero

        attn_out = self.attn(self.ln1(x), cos, sin, attn_mask)
        homeo_attn = (activation_rms(attn_out) - self.homeo_target).pow(2)
        x = x + self.attn_gain * attn_out

        if self.is_moe:
            ffn_out, moe_aux = self.ffn(self.ln2(x))
        else:
            ffn_out = self.ffn(self.ln2(x))
            moe_aux = x.new_zeros(())
        homeo_ffn = (activation_rms(ffn_out) - self.homeo_target).pow(2)
        x = x + self.ffn_gain * ffn_out

        homeo = homeo_attn + homeo_ffn
        return x, homeo, moe_aux


# =============================================================================
# Full model
# =============================================================================

class TensegrityLM(nn.Module):
    def __init__(self, cfg: TensegrityConfig):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpoint = cfg.grad_checkpoint

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        # Optional confidence head: predicts P(argmax prediction == target)
        if cfg.use_confidence_head:
            self.confidence_head = nn.Linear(cfg.n_embd, 1, bias=True)
        else:
            self.confidence_head = None

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        masks = build_attention_masks(cfg.block_size, cfg.sliding_window)
        self.register_buffer("mask_global", masks["global"], persistent=False)
        self.register_buffer("mask_local", masks["local"], persistent=False)

        self.apply(self._init_weights)

        n_moe = sum(1 for b in self.blocks if b.is_moe)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[TensegrityLM] layers={cfg.n_layer} (moe_layers={n_moe}) "
              f"n_embd={cfg.n_embd} n_head={cfg.n_head} n_kv_head={cfg.n_kv_head} "
              f"vocab={cfg.vocab_size} block_size={cfg.block_size}")
        print(f"[TensegrityLM] total parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        idx:     (B, T) integer token ids, T <= block_size
        targets: (B, T) integer token ids (next-token targets), or None for inference

        Returns:
            logits: (B, T, vocab_size)
            losses: dict of scalar tensors (only if targets is not None), else None
            confidence: (B, T) sigmoid confidence scores (only if targets is None and head enabled), else None
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.token_emb(idx)  # (B, T, C)
        x = self.drop(x)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        mask_global = self.mask_global[:T, :T]
        mask_local = self.mask_local[:T, :T]

        total_homeo = x.new_zeros(())
        total_moe_aux = x.new_zeros(())
        n_active = 0

        use_checkpoint = self.grad_checkpoint and self.training and x.requires_grad

        for block in self.blocks:
            attn_mask = mask_global if block.attn.is_global else mask_local
            if use_checkpoint:
                x, homeo, moe_aux = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, attn_mask, use_reentrant=False
                )
            else:
                x, homeo, moe_aux = block(x, cos, sin, attn_mask)
            total_homeo = total_homeo + homeo
            total_moe_aux = total_moe_aux + moe_aux
            n_active += 1

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1,
            )

            denom = max(1, n_active * 2)
            avg_homeo = total_homeo / denom

            n_moe_layers = sum(1 for b in self.blocks if b.is_moe)
            avg_moe_aux = total_moe_aux / max(1, n_moe_layers)

            losses = {
                "ce_loss": ce_loss,
                "homeo_loss": avg_homeo,
                "moe_aux_loss": avg_moe_aux,
            }

            total_loss = ce_loss + self.cfg.homeo_weight * avg_homeo + avg_moe_aux

            if self.confidence_head is not None:
                conf_logits = self.confidence_head(x).squeeze(-1)  # (B, T)
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)  # (B, T)
                    correct = (preds == targets).float()
                    valid = (targets != -1).float()
                conf_loss_raw = F.binary_cross_entropy_with_logits(
                    conf_logits, correct, reduction="none"
                )
                denom_c = valid.sum().clamp_min(1.0)
                conf_loss = (conf_loss_raw * valid).sum() / denom_c
                losses["confidence_loss"] = conf_loss
                total_loss = total_loss + self.cfg.confidence_weight * conf_loss

            losses["total_loss"] = total_loss
            return logits, losses, None

        else:
            confidence = None
            if self.confidence_head is not None:
                conf_logits = self.confidence_head(x).squeeze(-1)  # (B, T)
                confidence = torch.sigmoid(conf_logits)
            return logits, None, confidence

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                  temperature: float = 1.0, top_k: Optional[int] = None,
                  top_p: Optional[float] = None,
                  use_confidence_temp: bool = False,
                  conf_temp_min: float = 0.5, conf_temp_max: float = 1.5,
                  stop_on_eos: bool = True) -> torch.Tensor:
        """
        idx: (B, T) starting context.
        Returns (B, T + max_new_tokens) — may be shorter if EOS is hit and stop_on_eos
        is True (in which case generation stops for the whole batch at that step).
        """
        self.eval()
        for _ in range(max_new_tokens):
            T = idx.size(1)
            idx_cond = idx if T <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _, confidence = self.forward(idx_cond, targets=None)
            logits = logits[:, -1, :]  # (B, V)

            eff_temp = temperature
            if use_confidence_temp and confidence is not None:
                conf = confidence[:, -1]  # (B,)
                # low confidence -> higher temperature (more exploration)
                scale = conf_temp_max - conf * (conf_temp_max - conf_temp_min)
                eff_temp = temperature * scale.mean().item()
                eff_temp = max(eff_temp, 1e-4)

            logits = logits / eff_temp

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                cutoff = cumprobs > top_p
                cutoff[:, 1:] = cutoff[:, :-1].clone()
                cutoff[:, 0] = False
                sorted_logits[cutoff] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_tok], dim=1)

            if stop_on_eos and (next_tok == ByteTokenizer.EOS).all():
                break

        return idx

    # -------------------------------------------------------------------
    # Optimizer configuration
    # -------------------------------------------------------------------
    def configure_optimizer(self, weight_decay: float, learning_rate: float,
                             betas: Tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        seen = set()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
        return optimizer


# =============================================================================
# Data pipeline
# =============================================================================

def prepare_data(data_path: str, out_dir: str, val_fraction: float = 0.1, seed: int = 1337):
    """
    Read a text file or directory of files, byte-encode (with EOS separators between
    documents), split into train/val, and write uint16 .bin files for memmap access.
    """
    os.makedirs(out_dir, exist_ok=True)
    tok = ByteTokenizer()

    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "**", "*"), recursive=True))
        files = [f for f in files if os.path.isfile(f)]
    else:
        files = [data_path]

    if not files:
        raise ValueError(f"No files found at {data_path}")

    all_ids: List[int] = []
    for f in files:
        with open(f, "rb") as fh:
            raw = fh.read()
        all_ids.extend(tok.encode_bytes(raw))
        all_ids.append(ByteTokenizer.EOS)

    arr = np.array(all_ids, dtype=np.uint16)
    n = len(arr)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    if n_train <= 0:
        raise ValueError("Dataset too small after train/val split; provide more data.")

    train_arr = arr[:n_train]
    val_arr = arr[n_train:]

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    meta = {
        "vocab_size": ByteTokenizer.VOCAB_SIZE,
        "n_train_tokens": int(n_train),
        "n_val_tokens": int(n_val),
        "source_files": files,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[prepare] wrote {n_train:,} train tokens -> {train_path}")
    print(f"[prepare] wrote {n_val:,} val tokens -> {val_path}")


def get_batch(data_dir: str, split: str, batch_size: int, block_size: int,
               device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    path = os.path.join(data_dir, f"{split}.bin")
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(arr)
    if n <= block_size + 1:
        raise ValueError(
            f"{split}.bin has only {n} tokens, need > block_size+1 ({block_size + 1}). "
            f"Use a larger corpus or smaller --block_size."
        )
    max_start = n - block_size - 1
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([arr[s:s + block_size].astype(np.int64) for s in starts])
    y = np.stack([arr[s + 1:s + 1 + block_size].astype(np.int64) for s in starts])
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    if device == "cuda":
        x_t = x_t.pin_memory().to(device, non_blocking=True)
        y_t = y_t.pin_memory().to(device, non_blocking=True)
    else:
        x_t = x_t.to(device)
        y_t = y_t.to(device)
    return x_t, y_t


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                     cfg: TensegrityConfig, step: int, best_val_loss: float,
                     rng_state: Dict[str, Any]):
    raw_model = model.module if hasattr(model, "module") else model
    ckpt = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "best_val_loss": best_val_loss,
        "rng_state": rng_state,
    }
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, device: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]):
    if state is None:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass


# =============================================================================
# Learning rate schedule (cosine with warmup — "temperature annealing")
# =============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# Device / dtype helpers
# =============================================================================

def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[requested]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


# =============================================================================
# Training
# =============================================================================

def build_config_from_args(args) -> TensegrityConfig:
    cfg_kwargs = {}
    for f in dataclasses.fields(TensegrityConfig):
        if hasattr(args, f.name):
            val = getattr(args, f.name)
            if val is not None:
                cfg_kwargs[f.name] = val
    return TensegrityConfig(**cfg_kwargs)


def cmd_train(args):
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
    else:
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = pick_device(args.device)

    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)

    os.makedirs(args.out_dir, exist_ok=True)

    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if master_process:
            print(f"[data] {meta['n_train_tokens']:,} train tokens, "
                  f"{meta['n_val_tokens']:,} val tokens, vocab={meta['vocab_size']}")

    device_type = "cuda" if "cuda" in device else ("mps" if device == "mps" else "cpu")
    dtype = pick_dtype(args.dtype, device_type)
    if master_process:
        print(f"[train] device={device} dtype={dtype}")

    ckpt_resume = None
    if args.init_from == "resume":
        latest_path = os.path.join(args.out_dir, "ckpt_latest.pt")
        if os.path.exists(latest_path):
            ckpt_resume = load_checkpoint(latest_path, device)
            if master_process:
                print(f"[resume] loaded checkpoint from {latest_path} at step {ckpt_resume['step']}")
        else:
            if master_process:
                print(f"[resume] no checkpoint found at {latest_path}, starting from scratch")

    if ckpt_resume is not None:
        cfg = TensegrityConfig(**ckpt_resume["config"])
    else:
        cfg = build_config_from_args(args)

    model = TensegrityLM(cfg)
    model.to(device)

    if ckpt_resume is not None:
        model.load_state_dict(ckpt_resume["model_state_dict"])

    optimizer = model.configure_optimizer(
        weight_decay=args.weight_decay, learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2), device_type=device_type,
    )
    if ckpt_resume is not None:
        optimizer.load_state_dict(ckpt_resume["optimizer_state_dict"])
        restore_rng_state(ckpt_resume.get("rng_state"))

    start_step = ckpt_resume["step"] if ckpt_resume is not None else 0
    best_val_loss = ckpt_resume["best_val_loss"] if ckpt_resume is not None else float("inf")

    raw_model = model
    if args.compile:
        try:
            model = torch.compile(model)
            if master_process:
                print("[train] torch.compile enabled")
        except Exception as e:
            if master_process:
                print(f"[train] torch.compile failed ({e}), continuing without it")

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        raw_model = model.module

    use_amp = dtype in (torch.float16, torch.bfloat16) and device_type in ("cuda", "cpu")
    amp_dtype = dtype if use_amp else torch.float32
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16 and device_type == "cuda"))

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_file = open(log_path, "a") if master_process else None

    t0 = time.time()
    step = start_step
    running_loss = 0.0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.learning_rate, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for micro_step in range(args.grad_accum_steps):
            x, y = get_batch(args.data_dir, "train", args.batch_size, cfg.block_size, device)

            if is_ddp:
                model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)

            with torch.amp.autocast(device_type=device_type if device_type != "mps" else "cpu",
                                      dtype=amp_dtype, enabled=use_amp):
                logits, losses, _ = model(x, y)
                loss = losses["total_loss"] / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += losses["total_loss"].item() / args.grad_accum_steps

        if args.grad_clip > 0.0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 and master_process:
            dt = time.time() - t0
            avg_loss = running_loss / max(1, args.log_interval if step > start_step else 1)
            ce = losses["ce_loss"].item()
            homeo = losses["homeo_loss"].item()
            moe_aux = losses["moe_aux_loss"].item()
            conf = losses.get("confidence_loss")
            conf_str = f" conf={conf.item():.4f}" if conf is not None else ""
            print(f"step {step:6d} | lr {lr:.2e} | total {losses['total_loss'].item():.4f} "
                  f"(ce {ce:.4f} homeo {homeo:.4f} moe {moe_aux:.4f}{conf_str}) | {dt:.1f}s")
            log_file.write(json.dumps({
                "step": step, "lr": lr, "total_loss": losses["total_loss"].item(),
                "ce_loss": ce, "homeo_loss": homeo, "moe_aux_loss": moe_aux,
                "time": dt,
            }) + "\n")
            log_file.flush()
            running_loss = 0.0

        if step > 0 and step % args.eval_interval == 0 and master_process:
            val_loss = evaluate(raw_model, args.data_dir, cfg, device, device_type, amp_dtype,
                                  use_amp, args.eval_iters)
            print(f"step {step:6d} | val_loss {val_loss:.4f}")
            log_file.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
            log_file.flush()

            rng_state = capture_rng_state()
            save_checkpoint(os.path.join(args.out_dir, "ckpt_latest.pt"), raw_model, optimizer,
                             cfg, step, best_val_loss, rng_state)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(args.out_dir, "ckpt_best.pt"), raw_model, optimizer,
                                 cfg, step, best_val_loss, rng_state)
                print(f"step {step:6d} | new best val_loss {best_val_loss:.4f}, checkpoint saved")

        step += 1

    if master_process:
        val_loss = evaluate(raw_model, args.data_dir, cfg, device, device_type, amp_dtype,
                              use_amp, args.eval_iters)
        rng_state = capture_rng_state()
        save_checkpoint(os.path.join(args.out_dir, "ckpt_latest.pt"), raw_model, optimizer,
                         cfg, step, best_val_loss, rng_state)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(os.path.join(args.out_dir, "ckpt_best.pt"), raw_model, optimizer,
                             cfg, step, best_val_loss, rng_state)
        print(f"[train] done. final val_loss={val_loss:.4f} best_val_loss={best_val_loss:.4f}")
        log_file.close()

    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model: nn.Module, data_dir: str, cfg: TensegrityConfig, device: str,
             device_type: str, amp_dtype: torch.dtype, use_amp: bool, eval_iters: int) -> float:
    model.eval()
    losses = []
    for split in ["val"]:
        for _ in range(eval_iters):
            x, y = get_batch(data_dir, split, batch_size=8, block_size=cfg.block_size, device=device)
            with torch.amp.autocast(device_type=device_type if device_type != "mps" else "cpu",
                                      dtype=amp_dtype, enabled=use_amp):
                _, loss_dict, _ = model(x, y)
            losses.append(loss_dict["ce_loss"].item())
    model.train()
    return float(np.mean(losses))


# =============================================================================
# Generation / Eval CLI commands
# =============================================================================

def cmd_generate(args):
    device = pick_device(args.device)
    ckpt = load_checkpoint(args.ckpt, device)
    cfg = TensegrityConfig(**ckpt["config"])
    model = TensegrityLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tok = ByteTokenizer()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.prompt:
        ids = tok.encode(args.prompt)
    else:
        ids = []
    if args.add_bos:
        ids = [ByteTokenizer.BOS] + ids
    if not ids:
        ids = [ByteTokenizer.BOS]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        use_confidence_temp=args.use_confidence_temp,
        stop_on_eos=not args.no_stop_on_eos,
    )

    out_ids = out[0].tolist()
    text = tok.decode(out_ids)
    print(text)


def cmd_eval(args):
    device = pick_device(args.device)
    ckpt = load_checkpoint(args.ckpt, device)
    cfg = TensegrityConfig(**ckpt["config"])
    model = TensegrityLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    device_type = "cuda" if "cuda" in device else ("mps" if device == "mps" else "cpu")
    dtype = pick_dtype(args.dtype, device_type)
    use_amp = dtype in (torch.float16, torch.bfloat16) and device_type in ("cuda", "cpu")

    val_loss = evaluate(model, args.data_dir, cfg, device, device_type, dtype, use_amp, args.eval_iters)
    print(f"val_loss = {val_loss:.4f}")
    print(f"val_perplexity = {math.exp(val_loss):.4f}")


# =============================================================================
# Argument parsing
# =============================================================================

def add_model_args(p: argparse.ArgumentParser):
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--sliding_window", type=int, default=None)
    p.add_argument("--global_every", type=int, default=None)
    p.add_argument("--ffn_mult", type=float, default=None)
    p.add_argument("--ffn_align", type=int, default=None)
    p.add_argument("--moe_every", type=int, default=None)
    p.add_argument("--n_experts", type=int, default=None)
    p.add_argument("--moe_top_k", type=int, default=None)
    p.add_argument("--moe_aux_weight", type=float, default=None)
    p.add_argument("--homeo_target", type=float, default=None)
    p.add_argument("--homeo_weight", type=float, default=None)
    p.add_argument("--layerdrop_max", type=float, default=None)
    p.add_argument("--use_confidence_head", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--confidence_weight", type=float, default=None)
    p.add_argument("--bias", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--grad_checkpoint", type=lambda x: x.lower() == "true", default=None)


def main():
    parser = argparse.ArgumentParser(description="TensegrityLM — byte-level LM, single-file implementation")
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prep = sub.add_parser("prepare", help="Tokenize a text file/directory into train/val .bin files")
    p_prep.add_argument("--data_path", type=str, required=True)
    p_prep.add_argument("--out_dir", type=str, required=True)
    p_prep.add_argument("--val_fraction", type=float, default=0.1)
    p_prep.add_argument("--seed", type=int, default=1337)

    # train
    p_train = sub.add_parser("train", help="Train (or resume) a model")
    p_train.add_argument("--data_dir", type=str, required=True)
    p_train.add_argument("--out_dir", type=str, required=True)
    p_train.add_argument("--init_from", type=str, default="scratch", choices=["scratch", "resume"])
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.add_argument("--batch_size", type=int, default=12)
    p_train.add_argument("--grad_accum_steps", type=int, default=1)
    p_train.add_argument("--max_steps", type=int, default=5000)
    p_train.add_argument("--warmup_steps", type=int, default=100)
    p_train.add_argument("--learning_rate", type=float, default=3e-4)
    p_train.add_argument("--min_lr", type=float, default=3e-5)
    p_train.add_argument("--weight_decay", type=float, default=0.1)
    p_train.add_argument("--beta1", type=float, default=0.9)
    p_train.add_argument("--beta2", type=float, default=0.95)
    p_train.add_argument("--grad_clip", type=float, default=1.0)
    p_train.add_argument("--log_interval", type=int, default=10)
    p_train.add_argument("--eval_interval", type=int, default=250)
    p_train.add_argument("--eval_iters", type=int, default=20)
    p_train.add_argument("--compile", action="store_true")
    add_model_args(p_train)

    # generate
    p_gen = sub.add_parser("generate", help="Generate text from a checkpoint")
    p_gen.add_argument("--ckpt", type=str, required=True)
    p_gen.add_argument("--device", type=str, default="auto")
    p_gen.add_argument("--prompt", type=str, default="")
    p_gen.add_argument("--add_bos", action="store_true")
    p_gen.add_argument("--max_new_tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top_k", type=int, default=None)
    p_gen.add_argument("--top_p", type=float, default=None)
    p_gen.add_argument("--use_confidence_temp", action="store_true")
    p_gen.add_argument("--no_stop_on_eos", action="store_true")
    p_gen.add_argument("--seed", type=int, default=None)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint on val data")
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--data_dir", type=str, required=True)
    p_eval.add_argument("--device", type=str, default="auto")
    p_eval.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p_eval.add_argument("--eval_iters", type=int, default=50)

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_data(args.data_path, args.out_dir, args.val_fraction, args.seed)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
