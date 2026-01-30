from __future__ import annotations
import hashlib
import json
import logging
import random
import sys
from typing import Any

import numpy as np
import torch

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("plm_localization")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def set_seed(seed: int, force_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if force_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def stable_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
