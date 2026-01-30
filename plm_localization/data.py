from __future__ import annotations
import re
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

VALID_AA = set(list("ACDEFGHIKLMNPQRSTVWY") + ["X"])

def clean_and_map(seq: str) -> str:
    s = re.sub(r"\s+", "", str(seq).upper())
    s = re.sub(r"[UZOB]", "X", s)
    return "".join(ch if ch in VALID_AA else "X" for ch in s)

def load_dataset(path: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = df.rename(columns={"Protein sequence": "sequence", "Categories": "label"})
    df = df.dropna(subset=["sequence", "label"]).copy()
    df["sequence"] = df["sequence"].astype(str)
    df["label"] = df["label"].astype(str)

    df["seq_id"] = [f"seq_{i:05d}" for i in range(len(df))]
    df["sequence_clean"] = df["sequence"].map(clean_and_map)

    empty = df["sequence_clean"].str.len().fillna(0).astype(int) == 0
    df = df.loc[~empty].reset_index(drop=True)

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    return df, le
