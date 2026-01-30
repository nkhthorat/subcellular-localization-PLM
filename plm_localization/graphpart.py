from __future__ import annotations
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False

def ensure_graphpart_dependencies() -> None:
    if shutil.which("graphpart") is None:
        raise RuntimeError("graphpart CLI not found on PATH. Install: pip install graph-part")
    if shutil.which("needleall") is None:
        raise RuntimeError("EMBOSS needleall not found on PATH (required for graphpart needle mode).")

def pick_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def run_graphpart(
    df: pd.DataFrame,
    save_dir: str,
    min_seq_id: float,
    k_folds: int,
    threads: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ensure_graphpart_dependencies()
    os.makedirs(save_dir, exist_ok=True)

    fasta_path = os.path.join(save_dir, "graphpart_input.fasta")
    out_csv = os.path.join(save_dir, "graphpart_assignments.csv")

    with open(fasta_path, "w") as f:
        for sid, label, seq in zip(df["seq_id"], df["label"], df["sequence_clean"]):
            f.write(f">{sid}|label={label}|priority=0\n{seq}\n")

    cmd = [
        "graphpart", "needle",
        "--fasta-file", fasta_path,
        "--threshold", str(min_seq_id),
        "--out-file", out_csv,
        "--labels-name", "label",
        "--partitions", str(k_folds),
        "--threads", str(threads),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"GraphPart failed (exit code {res.returncode}).")

    assign = pd.read_csv(out_csv)
    id_col = pick_first_existing(list(assign.columns), ["AC", "seq_id", "id", "name", "ac"])
    if id_col is None:
        raise RuntimeError(f"Cannot find ID column in GraphPart output: {list(assign.columns)}")

    ids_in_assign = set(assign[id_col].astype(str).tolist())
    keep_idx = np.array([i for i, sid in enumerate(df["seq_id"].astype(str)) if sid in ids_in_assign], dtype=int)
    removed_idx = np.array([i for i, sid in enumerate(df["seq_id"].astype(str)) if sid not in ids_in_assign], dtype=int)

    assign = assign[assign[id_col].astype(str).isin(df["seq_id"].iloc[keep_idx].astype(str))].copy()
    return assign, keep_idx, removed_idx

def build_splits(
    df_kept: pd.DataFrame,
    assign: pd.DataFrame,
    k_folds: int,
    seed: int,
    prefer_partitions: bool = True,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    cols = list(assign.columns)
    id_col = pick_first_existing(cols, ["AC", "seq_id", "id", "name", "ac"])
    cluster_col = pick_first_existing(cols, ["cluster", "group", "component", "cc", "connected_component"])
    part_col = pick_first_existing(cols, ["partition", "fold", "part", "set"])

    if id_col is None or cluster_col is None:
        raise RuntimeError(f"GraphPart output missing required columns: {cols}")

    sid_to_cluster: Dict[str, int] = {
        str(sid): int(cl) for sid, cl in zip(assign[id_col].astype(str), assign[cluster_col])
    }
    groups = np.array([sid_to_cluster[str(sid)] for sid in df_kept["seq_id"].astype(str)], dtype=int)
    y = df_kept["label_enc"].values

    if prefer_partitions and part_col is not None:
        sid_to_part: Dict[str, int] = {
            str(sid): int(p) for sid, p in zip(assign[id_col].astype(str), assign[part_col])
        }
        fold_ids = np.array([sid_to_part[str(sid)] for sid in df_kept["seq_id"].astype(str)], dtype=int)

        uniq = np.unique(fold_ids)
        if uniq.min() == 1 and uniq.max() == k_folds:
            fold_ids = fold_ids - 1
            uniq = np.unique(fold_ids)

        if len(uniq) == k_folds:
            splits = []
            all_idx = np.arange(len(df_kept))
            for f in sorted(uniq.tolist()):
                va = np.where(fold_ids == f)[0]
                tr = np.setdiff1d(all_idx, va)
                splits.append((tr, va))
            return splits, y, groups

    # fallback
    if HAS_STRATIFIED_GROUP_KFOLD:
        sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits = list(sgkf.split(np.zeros(len(df_kept)), y, groups=groups))
        return splits, y, groups

    gkf = GroupKFold(n_splits=k_folds)
    splits = list(gkf.split(np.zeros(len(df_kept)), y, groups=groups))
    return splits, y, groups
