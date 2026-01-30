from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    data_path: str
    save_dir: str

    backbone: str                 # "ProtT5-XL-UniRef50"
    prott5_name: str
    esm1b_name: str
    last_k: int

    nterm_max_res: int
    pooling: str

    use_homology_aware_cv: bool
    min_seq_id: float
    k_folds: int
    graphpart_threads: int
    prefer_graphpart_partitions: bool

    seed: int
    lr: float
    weight_decay: float
    h1: int
    h2: int
    dropout: float
    label_smoothing: float
    batch_size: int
    epochs: int
    early_stop_patience: int
    force_deterministic: bool

    device: str
    fp16: bool
    embed_batch_size: int

    cache_embeddings: bool
