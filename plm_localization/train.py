from __future__ import annotations
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from .config import Config
from .model import PooledMLP

def train_one_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Config,
    n_classes: int,
) -> Tuple[PooledMLP, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_va)

    device = torch.device(cfg.device)
    model = PooledMLP(Xtr.shape[1], n_classes, cfg.h1, cfg.h2, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)

    best_f1, best_state, no_improve = -1.0, None, 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            for xb, yb in va_loader:
                logits = model(xb.to(device))
                preds = torch.argmax(logits, 1).cpu().numpy().tolist()
                y_pred.extend(preds)
                y_true.extend(yb.numpy().tolist())

        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1m > best_f1 + 1e-4:
            best_f1 = f1m
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.early_stop_patience:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, scaler
