#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import torch

from plm_localization.config import Config
from plm_localization.utils import setup_logger, set_seed, stable_json, sha1_hex
from plm_localization.data import load_dataset
from plm_localization.graphpart import run_graphpart, build_splits
from plm_localization.embeddings import compute_features
from plm_localization.train import train_one_fold
from plm_localization.eval import compute_oof_metrics, confusion_matrix_df


def _json_default(obj):
    """Minimal helper so json.dump can serialize numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.xlsx")
    p.add_argument("--save-dir", default="runs/run1")
    p.add_argument("--backbone", choices=["prott5", "esm1b"], default="prott5")
    p.add_argument("--nterm", type=int, default=250)
    p.add_argument("--pooling", default="mean_prefix_50")
    p.add_argument("--seed", type=int, default=200)
    args = p.parse_args()

    logger = setup_logger("INFO")
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(
        data_path=args.data,
        save_dir=args.save_dir,

        backbone=args.backbone,
        prott5_name="Rostlab/prot_t5_xl_uniref50",
        esm1b_name="facebook/esm1b_t33_650M_UR50S",
        last_k=4,

        nterm_max_res=args.nterm,
        pooling=args.pooling,

        use_homology_aware_cv=True,
        min_seq_id=0.30,
        k_folds=5,
        graphpart_threads=2,
        prefer_graphpart_partitions=True,

        seed=args.seed,
        lr=1e-3,
        weight_decay=1e-5,
        h1=512,
        h2=128,
        dropout=0.3,
        label_smoothing=0.05,
        batch_size=64,
        epochs=50,
        early_stop_patience=8,
        force_deterministic=False,

        device=device,
        fp16=(device == "cuda"),
        embed_batch_size=8,

        cache_embeddings=True,
    )

    set_seed(cfg.seed, cfg.force_deterministic)

    # Load + GraphPart
    df, le = load_dataset(cfg.data_path)
    joblib.dump(le, os.path.join(cfg.save_dir, "label_encoder.pkl"))

    gp_dir = os.path.join(cfg.save_dir, "graphpart")
    assign, keep_idx, removed_idx = run_graphpart(
        df, gp_dir, cfg.min_seq_id, cfg.k_folds, cfg.graphpart_threads
    )
    df = df.iloc[keep_idx].reset_index(drop=True)

    splits, y, groups = build_splits(df, assign, cfg.k_folds, cfg.seed, prefer_partitions=True)

    # Features
    X = compute_features(df["sequence_clean"].astype(str).tolist(), cfg)

    # OOF
    oof_pred = np.full(len(df), -1, dtype=np.int64)
    for fold_i, (tr, va) in enumerate(splits, 1):
        model, scaler = train_one_fold(X[tr], y[tr], X[va], y[va], cfg, n_classes=len(le.classes_))
        with torch.no_grad():
            Xva_s = scaler.transform(X[va])
            logits = model(torch.tensor(Xva_s, dtype=torch.float32).to(torch.device(cfg.device)))
            preds = torch.argmax(logits, 1).cpu().numpy()
        oof_pred[va] = preds

    metrics = compute_oof_metrics(y, oof_pred, class_names=list(le.classes_))
    logger.info(
        "OOF Acc=%.4f | Macro-F1=%.4f | Weighted-F1=%.4f | MCC=%.4f",
        metrics["acc"], metrics["macro_f1"], metrics["weighted_f1"], metrics["mcc"]
    )

    cm_df = confusion_matrix_df(metrics["cm"], list(le.classes_))
    cm_df.to_csv(os.path.join(cfg.save_dir, "oof_confusion_matrix.csv"), index=True)

    with open(os.path.join(cfg.save_dir, "oof_report.txt"), "w") as f:
        f.write(metrics["report"])

    with open(os.path.join(cfg.save_dir, "summary.json"), "w") as f:
        json.dump(
            metrics | {"config": json.loads(stable_json(cfg.__dict__))},
            f,
            indent=2,
            default=_json_default,
        )


if __name__ == "__main__":
    main()
