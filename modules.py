import os
import re
import copy
import random
import hashlib
import argparse
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

_whitespace_re = re.compile(r"[ \t]+")
_newlines_re = re.compile(r"\n{2,}")
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _whitespace_re.sub(" ", s)
    s = _newlines_re.sub("\n", s)
    return s


def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def maybe_parallelize(model, device, use_data_parallel=True):
    model = model.to(device)
    if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, fold, best_f1, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}_epoch_{epoch}.pt")
    
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "fold": fold,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_f1": best_f1,
    }
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def load_model_state_compat(model, state_dict, strict=False, log_prefix=""):
    """Load checkpoint state_dict with DP/DDP prefix compatibility.

    This keeps existing checkpoint files usable across single-GPU, DataParallel,
    and DDP inference/training resumes.
    """
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dict")

    model_keys = model.state_dict().keys()
    model_has_module = any(k.startswith("module.") for k in model_keys)
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())

    adapted_state = state_dict
    if ckpt_has_module and not model_has_module:
        adapted_state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    elif model_has_module and not ckpt_has_module:
        adapted_state = {f"module.{k}": v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(adapted_state, strict=strict)

    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing or unexpected:
        print(
            f"[WARN] {log_prefix}load_state_dict mismatch | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    return incompatible


def load_checkpoint(ckpt_path, model, optimizer, scheduler, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Resume should be strict for optimizer/scheduler continuity, while still
    # handling DP/DDP key prefix differences in saved checkpoints.
    load_model_state_compat(model, checkpoint["model_state_dict"], strict=True, log_prefix=f"{os.path.basename(ckpt_path)}: ")
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return model, optimizer, scheduler, checkpoint["epoch"], checkpoint["best_f1"]


def find_latest_ckpt(fold, ckpt_dir):
    pattern = os.path.join(ckpt_dir, f"fold_{fold}_epoch_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getctime)


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_encodings(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []

    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=-1)

        preds.extend(pred.cpu().numpy().tolist())
        trues.extend(labels.cpu().numpy().tolist())

    return f1_score(trues, preds, average="macro"), np.array(preds, dtype=int)


def save_confusion_matrix_as_table(cm, id2label, save_path):
    """Save confusion matrix as CSV table."""
    label_names = [str(id2label[i]) for i in range(len(id2label))]
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.index.name = 'True Label'
    cm_df.to_csv(save_path)


def save_classification_report_as_table(y_true, y_pred, id2label, save_path):
    """Save classification report as CSV table."""
    label_names = [str(id2label[i]) for i in range(len(id2label))]
    report_dict = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(save_path)

