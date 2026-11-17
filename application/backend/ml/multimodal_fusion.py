"""Multimodal cross-attention fusion for the CODE-15% engine (Requirement 10).

The 10-element Clinical_Vector (order in ``CLINICAL_FEATURES``) is normalized
with fixed reference values and scales, clipped to [-5, 5], and turned into 10
clinical tokens. These tokens attend (as queries) over the 16 ECG tokens
projected from the ResNet1d Deep_Features ``(B, 320, 16)``. The pooled result
gives a bounded Neural_Risk_Adjustment in [-5, 5] points.

The weights are not trained. ``build_fusion_module`` initializes them from a
fixed seed inside a forked RNG, so every process gets the same module and the
global torch RNG state is left unchanged (Req 10.4).

Adapted from ``MultiModalCrossAttention`` in
``REPO/TAVI-MM-Fed-PPM/custom_multimodal_training_plan.py``.

This module imports torch at module level. ``ml/code15_engine.py`` imports it
only when torch is available.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

CLINICAL_FEATURES = (
    "age", "is_male", "bp_sys", "bp_dia", "glucose", "cholesterol",
    "creatinine", "heart_age_gap", "st_amp", "active_arrhythmia_count",
)
CLINICAL_REF = (50.0, 0.5, 120.0, 80.0, 90.0, 180.0, 1.0, 0.0, 0.0, 0.0)
CLINICAL_SCALE = (15.0, 0.5, 20.0, 10.0, 30.0, 40.0, 0.3, 10.0, 0.1, 1.0)
FUSION_SEED = 1515
EMBED = 128
HEADS = 4
NEURAL_LIMIT = 5.0
CLIP = 5.0


@dataclass
class FusionOutput:
    neural_adjustment: Tensor  # (B,) in [-NEURAL_LIMIT, NEURAL_LIMIT]
    attention: Tensor          # (B, 10, 16), each row sums to 1


class MultiModalCrossAttention(nn.Module):
    """Clinical tokens (queries) attend over ECG tokens (keys and values)."""

    def __init__(
        self,
        ecg_channels: int = 320,
        n_ecg_tokens: int = 16,
        n_clinical: int = len(CLINICAL_FEATURES),
        embed_dim: int = EMBED,
        num_heads: int = HEADS,
    ) -> None:
        super().__init__()
        if n_clinical != len(CLINICAL_FEATURES):
            raise ValueError(f"n_clinical must be {len(CLINICAL_FEATURES)}")
        self.ecg_channels = ecg_channels
        self.n_ecg_tokens = n_ecg_tokens
        self.n_clinical = n_clinical

        self.register_buffer("clin_ref", torch.tensor(CLINICAL_REF, dtype=torch.float32))
        self.register_buffer("clin_scale", torch.tensor(CLINICAL_SCALE, dtype=torch.float32))

        self.ecg_proj = nn.Linear(ecg_channels, embed_dim)
        self.ecg_pos = nn.Parameter(torch.randn(n_ecg_tokens, embed_dim) * 0.02)
        self.clin_value = nn.Parameter(torch.randn(n_clinical, embed_dim) * embed_dim ** -0.5)
        self.clin_pos = nn.Parameter(torch.randn(n_clinical, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.head = nn.Linear(embed_dim, 1)

    def normalize_clinical(self, cv: Tensor) -> Tensor:
        """``clip((cv - ref) / scale, -5, 5)``; non-finite inputs become 0."""
        z = (cv.to(self.clin_ref.dtype) - self.clin_ref) / self.clin_scale
        z = torch.nan_to_num(z, nan=0.0, posinf=CLIP, neginf=-CLIP)
        return z.clamp(-CLIP, CLIP)

    def forward(self, ecg_features: Tensor, clinical: Tensor) -> FusionOutput:
        if ecg_features.dim() != 3 or ecg_features.shape[1:] != (self.ecg_channels, self.n_ecg_tokens):
            raise ValueError(
                f"ecg_features must have shape (B, {self.ecg_channels}, {self.n_ecg_tokens}), "
                f"got {tuple(ecg_features.shape)}"
            )
        if clinical.dim() != 2 or clinical.shape[1] != self.n_clinical:
            raise ValueError(
                f"clinical must have shape (B, {self.n_clinical}), got {tuple(clinical.shape)}"
            )
        if clinical.shape[0] != ecg_features.shape[0]:
            raise ValueError("ecg_features and clinical batch sizes differ")

        feats = ecg_features.to(self.ecg_proj.weight.dtype)
        ecg = self.ecg_proj(feats.transpose(1, 2)) + self.ecg_pos          # (B, 16, 128)
        z = self.normalize_clinical(clinical)                               # (B, 10)
        clin = z[..., None] * self.clin_value + self.clin_pos               # (B, 10, 128)

        out, weights = self.attn(
            clin, ecg, ecg, need_weights=True, average_attn_weights=True
        )                                                                   # (B, 10, 128), (B, 10, 16)
        pooled = (out + clin).mean(1)                                       # (B, 128)
        neural = NEURAL_LIMIT * torch.tanh(self.head(pooled)).squeeze(-1)   # (B,)
        return FusionOutput(neural_adjustment=neural, attention=weights)


def build_fusion_module(seed: int = FUSION_SEED) -> MultiModalCrossAttention:
    """Build the fusion module from a fixed seed without touching the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = MultiModalCrossAttention()
    return module.eval()
