from __future__ import annotations
import math
import torch

POOLING_MODES = {
    "mean_full",
    "mean_prefix_50",
    "mean_prefix_100",
    "mean_segment_0",
    "mean_sp",
    "mean_tp",
    "mean_tail_250_500",
    "concat_bio_all",
}

def _mean_or_zeros(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((x.shape[-1],), dtype=x.dtype)
    return x.mean(dim=0)

def pool_residue_embeddings(emb: torch.Tensor, mode: str) -> torch.Tensor:
    L, D = emb.shape

    if mode == "mean_full":
        return emb.mean(dim=0)

    if mode == "mean_prefix_50":
        return _mean_or_zeros(emb[: min(50, L)])

    if mode == "mean_prefix_100":
        return _mean_or_zeros(emb[: min(100, L)])

    if mode == "mean_segment_0":
        n = max(1, int(math.ceil(0.10 * L)))
        return _mean_or_zeros(emb[: min(n, L)])

    if mode == "mean_sp":
        return _mean_or_zeros(emb[: min(30, L)])

    if mode == "mean_tp":
        return _mean_or_zeros(emb[min(30, L): min(100, L)])

    if mode == "mean_tail_250_500":
        return _mean_or_zeros(emb[min(250, L): min(500, L)])

    if mode == "concat_bio_all":
        parts = [
            pool_residue_embeddings(emb, "mean_full"),
            pool_residue_embeddings(emb, "mean_segment_0"),
            pool_residue_embeddings(emb, "mean_sp"),
            pool_residue_embeddings(emb, "mean_prefix_50"),
            pool_residue_embeddings(emb, "mean_prefix_100"),
            pool_residue_embeddings(emb, "mean_tp"),
        ]
        return torch.cat(parts, dim=0)

    raise ValueError(f"Unknown pooling mode: {mode}")
