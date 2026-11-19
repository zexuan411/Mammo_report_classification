import os
import re
import copy
import random
import hashlib
import argparse
import gc
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from modules import set_seed, clean_text, stable_hash, maybe_parallelize, save_checkpoint, load_checkpoint, find_latest_ckpt, evaluate, make_encodings, TextDataset, load_model_state_compat, save_confusion_matrix_as_table, save_classification_report_as_table, hard_cascade_from_outputs
from data_segmentation import split_report_by_sections


def prepare_report_text(report: str, text_section: str = "full") -> str:
    """Build model input text. text_section='achados' keeps findings only."""
    if text_section == "achados":
        findings = split_report_by_sections(report).get("findings", "")
        findings = clean_text(findings)
        if findings:
            return findings
        # Fallback if section parser misses Achados.
        return clean_text(report)
    return clean_text(report)

RANDOM_STATE = 42
N_SPLITS = 5
NUM_LABELS = 7  # BI-RADS targets are fixed contiguous ints 0..6

_whitespace_re = re.compile(r"[ \t]+")
_newlines_re = re.compile(r"\n{2,}")


def unwrap_model(model):
    while True:
        if isinstance(model, (torch.nn.DataParallel, DDP)):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        break
    return model


def parse_train_folds(train_folds_str, n_splits):
    """Parse train_folds string like '1,2,3' or '1-3' into a set of fold indices."""
    if not train_folds_str or train_folds_str.strip() == "":
        return set(range(1, n_splits + 1))
    
    folds = set()
    parts = train_folds_str.split(",")
    
    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            start, end = int(start.strip()), int(end.strip())
            folds.update(range(start, end + 1))
        else:
            folds.add(int(part))
    
    # Validate fold indices
    invalid = [f for f in folds if f < 1 or f > n_splits]
    if invalid:
        raise ValueError(f"Invalid fold indices: {invalid}. Must be between 1 and {n_splits}")
    
    return sorted(folds)


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def select_encodings(encodings, indices):
    idx = torch.as_tensor(indices, dtype=torch.long)
    return {k: v.index_select(0, idx) for k, v in encodings.items()}


def resolve_amp_dtype(amp_dtype: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if amp_dtype not in mapping:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
    return mapping[amp_dtype]


def ddp_env_is_ready() -> bool:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    return all(k in os.environ for k in required)


def get_first_trainable_param_dtype(model):
    base_model = unwrap_model(model)
    for p in base_model.parameters():
        if p.requires_grad:
            return p.dtype
    return None


from typing import NamedTuple, Optional


class DualHeadOutput(NamedTuple):
    """Minimal output container compatible with HF SequenceClassifierOutput usage."""

    logits: Optional[torch.Tensor] = None
    ordinal_logits: Optional[torch.Tensor] = None
    binary_logits: Optional[torch.Tensor] = None


class DualHeadSequenceClassifier(torch.nn.Module):

    def __init__(self, model_name, dropout=None, ordinal_ignore_label=None):
        super().__init__()
        self.num_labels = NUM_LABELS
        self.ordinal_ignore_label = (
            None if ordinal_ignore_label is None else int(ordinal_ignore_label)
        )
        # Default: K-1 thresholds; with ignore label: K-2 (exclude non-diagnostic).
        n_ordinal = NUM_LABELS - 2 if self.ordinal_ignore_label is not None else NUM_LABELS - 1
        self.num_ordinal_thresholds = n_ordinal
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        if dropout is None:
            dropout = getattr(self.config, "hidden_dropout_prob", 0.1)
        self.dropout = torch.nn.Dropout(dropout)
        self.ce_classifier = torch.nn.Linear(hidden, NUM_LABELS)
        self.ordinal_classifier = torch.nn.Linear(hidden, n_ordinal)

        # DualHead uses CLS token features, not BertPooler. Freeze it so DDP
        # does not treat pooler weights as unused parameters.
        pooler = getattr(self.encoder, "pooler", None)
        if pooler is not None:
            pooler.requires_grad_(False)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        encoder_kwargs = {}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **encoder_kwargs,
        )
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = self.ce_classifier(cls)
        ordinal_logits = self.ordinal_classifier(cls)
        return DualHeadOutput(logits=logits, ordinal_logits=ordinal_logits)

    def export_ce_sequence_classifier(self, model_name):
        """Build a standard HF sequence classifier with encoder + CE head weights."""
        export_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
        )
        base = getattr(export_model, "base_model", None)
        if base is None:
            raise RuntimeError("Could not locate base encoder on export model")
        base.load_state_dict(self.encoder.state_dict(), strict=False)
        if hasattr(export_model, "classifier") and isinstance(export_model.classifier, torch.nn.Linear):
            if (
                export_model.classifier.weight.shape == self.ce_classifier.weight.shape
                and export_model.classifier.bias.shape == self.ce_classifier.bias.shape
            ):
                export_model.classifier.weight.data.copy_(self.ce_classifier.weight.data)
                export_model.classifier.bias.data.copy_(self.ce_classifier.bias.data)
        return export_model


class HierarchicalBiradsClassifier(torch.nn.Module):
    """
    Shared encoder + binary head (label 0 vs rest) + ordinal head (classes 1..K-1).
    Decode with hard cascade (binary gate then ordinal); no soft K-way logits.
    """

    def __init__(self, model_name, dropout=None, zero_label=0):
        super().__init__()
        self.num_labels = NUM_LABELS
        self.zero_label = int(zero_label)
        if self.zero_label != 0:
            raise ValueError("HierarchicalBiradsClassifier currently supports zero_label=0 only")
        self.num_ordinal_thresholds = NUM_LABELS - 2
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        if dropout is None:
            dropout = getattr(self.config, "hidden_dropout_prob", 0.1)
        self.dropout = torch.nn.Dropout(dropout)
        self.binary_classifier = torch.nn.Linear(hidden, 1)
        self.ordinal_classifier = torch.nn.Linear(hidden, self.num_ordinal_thresholds)
        pooler = getattr(self.encoder, "pooler", None)
        if pooler is not None:
            pooler.requires_grad_(False)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        encoder_kwargs = {}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **encoder_kwargs,
        )
        cls = self.dropout(outputs.last_hidden_state[:, 0])
        binary_logits = self.binary_classifier(cls).squeeze(-1)
        ordinal_logits = self.ordinal_classifier(cls)
        return DualHeadOutput(
            ordinal_logits=ordinal_logits,
            binary_logits=binary_logits,
        )

    def export_ce_sequence_classifier(self, model_name):
        """Export encoder into a standard K-way HF classifier (head re-init)."""
        export_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
        )
        base = getattr(export_model, "base_model", None)
        if base is None:
            raise RuntimeError("Could not locate base encoder on export model")
        base.load_state_dict(self.encoder.state_dict(), strict=False)
        return export_model


def build_classification_model(
    model_name,
    dual_head=False,
    ordinal_ignore_label=None,
    hierarchical=False,
    zero_label=0,
):
    if hierarchical:
        return HierarchicalBiradsClassifier(
            model_name,
            zero_label=zero_label,
        )
    if dual_head:
        return DualHeadSequenceClassifier(
            model_name,
            ordinal_ignore_label=ordinal_ignore_label,
        )
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
    )


def ordinal_threshold_loss(
    ordinal_logits,
    labels,
    num_classes,
    class_weights=None,
    ordinal_ignore_label=None,
):
    """
    Ordinal BCE on dedicated threshold logits.
    For remapped class k, target is k leading ones then zeros.
    If ordinal_ignore_label is set, that class is excluded from the ordinal
    chain and the remaining labels are collapsed around it.
    ordinal_logits is expected to already have the matching width:
      K-1 normally, or K-2 when one label is ignored.
    """
    ordinal_logits = ordinal_logits.float()
    labels = labels.long()
    original_labels = labels
    n_thresholds = ordinal_logits.size(1)

    if ordinal_ignore_label is not None:
        ordinal_ignore_label = int(ordinal_ignore_label)
        keep_mask = labels != ordinal_ignore_label
        if not torch.any(keep_mask):
            # Exclusive graph hook for all-ignored batches (do NOT also add this
            # when BCE runs, or DDP marks ordinal params ready twice).
            return (ordinal_logits * 0.0).sum()

        ordinal_logits = ordinal_logits[keep_mask]
        original_labels = labels[keep_mask]
        labels = original_labels - (original_labels > ordinal_ignore_label).long()
        # Diagnostic ordinal chain has one fewer class than the full CE head.
        num_classes = num_classes - 1
    expected_thresholds = max(num_classes - 1, 0)
    if n_thresholds != expected_thresholds:
        raise ValueError(
            f"ordinal_logits has {n_thresholds} thresholds, expected {expected_thresholds} "
            f"for num_classes={num_classes} (ordinal_ignore_label={ordinal_ignore_label})"
        )
    if n_thresholds == 0:
        return (ordinal_logits * 0.0).sum()

    batch_size = ordinal_logits.size(0)
    device = ordinal_logits.device
    dtype = ordinal_logits.dtype

    ordinal_targets = torch.zeros(batch_size, n_thresholds, device=device, dtype=dtype)
    label_list = labels.detach().cpu().tolist()
    for i, label in enumerate(label_list):
        label = int(label)
        if label > 0:
            ordinal_targets[i, :label] = 1.0

    bce_loss = F.binary_cross_entropy_with_logits(
        ordinal_logits, ordinal_targets, reduction="none"
    )
    if class_weights is not None:
        weights = class_weights[original_labels].to(dtype)
        bce_loss = bce_loss * weights.unsqueeze(1)
    return bce_loss.mean()


def ordinal_loss(logits, labels, class_weights=None, ordinal_ignore_label=None):
    """
    Ordinal loss for ordinal regression on shared K-way logits.
    Uses the first K-1 logits as thresholds.
    """
    logits = logits.float()
    num_classes = logits.size(1)
    threshold_logits = logits[:, :-1]
    if ordinal_ignore_label == 0:
        # BI-RADS 0 means insufficient information, not the lowest diagnosis.
        # Keep it out of the shared-logit ordinal thresholds.
        threshold_logits = logits[:, 1:-1]
    return ordinal_threshold_loss(
        threshold_logits,
        labels,
        num_classes=num_classes,
        class_weights=class_weights,
        ordinal_ignore_label=ordinal_ignore_label,
    )


def focal_loss(logits, labels, class_weights=None, gamma=2.0):
    """
    Focal loss for multi-class classification.
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: [batch_size, num_classes]
        labels: [batch_size] with values 0, 1, ..., num_classes-1
        class_weights: [num_classes] optional class weights (alpha)
        gamma: focusing parameter; higher values down-weight easy examples

    Returns:
        scalar loss
    """
    logits = logits.float()
    weight = class_weights.float() if class_weights is not None else None
    ce_loss = F.cross_entropy(logits, labels, weight=weight, reduction="none")
    pt = torch.exp(-ce_loss)
    return (((1.0 - pt) ** gamma) * ce_loss).mean()


def cross_entropy_loss(logits, labels, class_weights=None, **kwargs):
    return F.cross_entropy(
        logits.float(),
        labels,
        weight=class_weights.float() if class_weights is not None else None,
    )


def dual_head_mixed_loss(
    ce_logits,
    ordinal_logits,
    labels,
    class_weights=None,
    mix_lambda=0.9,
    use_focal=False,
    gamma=2.0,
    ordinal_ignore_label=None,
):
    """Dual-head mix: L = λ * L_CE/focal + (1-λ) * L_ordinal."""
    mix_lambda = float(mix_lambda)
    mix_lambda = min(max(mix_lambda, 0.0), 1.0)
    primary = (
        focal_loss(ce_logits, labels, class_weights, gamma=gamma)
        if use_focal
        else cross_entropy_loss(ce_logits, labels, class_weights)
    )
    ord_loss = ordinal_threshold_loss(
        ordinal_logits,
        labels,
        num_classes=ce_logits.size(1),
        class_weights=class_weights,
        ordinal_ignore_label=ordinal_ignore_label,
    )
    return mix_lambda * primary + (1.0 - mix_lambda) * ord_loss


def hierarchical_birads_loss(
    binary_logits,
    ordinal_logits,
    labels,
    num_classes,
    class_weights=None,
    mix_lambda=0.5,
    zero_label=0,
):
    """
    Hierarchical BI-RADS loss:
      L = λ * BCE(is_zero) + (1-λ) * ordinal(classes != zero_label)
    """
    mix_lambda = float(mix_lambda)
    mix_lambda = min(max(mix_lambda, 0.0), 1.0)
    labels = labels.long()
    zero_label = int(zero_label)
    if binary_logits.ndim > 1:
        binary_logits = binary_logits.squeeze(-1)

    is_zero = (labels == zero_label).to(dtype=binary_logits.dtype)
    sample_w = None
    if class_weights is not None:
        # Weight binary positives/negatives by the corresponding class weights.
        sample_w = torch.where(
            labels == zero_label,
            class_weights[zero_label].to(binary_logits.dtype),
            class_weights[labels].to(binary_logits.dtype),
        )
    bin_loss = F.binary_cross_entropy_with_logits(
        binary_logits.float(), is_zero, weight=sample_w, reduction="mean"
    )

    ord_loss = ordinal_threshold_loss(
        ordinal_logits,
        labels,
        num_classes=num_classes,
        class_weights=class_weights,
        ordinal_ignore_label=zero_label,
    )
    return mix_lambda * bin_loss + (1.0 - mix_lambda) * ord_loss


def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    lr,
    epochs,
    warmup_ratio,
    class_weights,
    loss_fn=cross_entropy_loss,
    fold=1,
    ckpt_dir=None,
    resume=True,
    use_amp=True,
    grad_accum_steps=1,
    val_interval=1,
    use_compile=False,
    use_ddp=False,
    run_validation=True,
    amp_dtype=torch.float16,
    use_grad_scaler=True,
    loss_fn_kwargs=None,
    use_ordinal_decode=False,
    use_dual_head=False,
    use_hierarchical=False,
    use_hard_cascade=False,
    cascade_tau=0.5,
    hier_zero_label=0,
):
    # Optional: compile model for improved performance (requires PyTorch 2.0+)
    # Compile the unwrapped module before DDP when possible; skip if already wrapped.
    if use_compile and hasattr(torch, "compile") and not isinstance(model, (torch.nn.DataParallel, DDP)):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    if loss_fn_kwargs is None:
        loss_fn_kwargs = {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # GradScaler is only safe when grads are float16-compatible for unscale kernels.
    param_dtype = get_first_trainable_param_dtype(model)
    grad_scaler_enabled = bool(use_grad_scaler)
    if grad_scaler_enabled and param_dtype == torch.bfloat16:
        grad_scaler_enabled = False

    scaler = GradScaler('cuda', enabled=grad_scaler_enabled, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)
    best_f1 = -1.0
    best_epoch = 1
    best_state = None
    start_epoch = 1

    if resume and ckpt_dir:
        latest_ckpt = find_latest_ckpt(fold, ckpt_dir)
        if latest_ckpt:
            model, optimizer, scheduler, start_epoch, best_f1 = load_checkpoint(
                latest_ckpt, model, optimizer, scheduler, device
            )
            best_epoch = start_epoch
            start_epoch += 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if use_ddp and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        accumulation_steps = 0

        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if use_hierarchical:
                    loss = hierarchical_birads_loss(
                        outputs.binary_logits.float(),
                        outputs.ordinal_logits.float(),
                        labels,
                        num_classes=NUM_LABELS,
                        class_weights=class_weights,
                        **loss_fn_kwargs,
                    )
                elif use_dual_head:
                    loss = dual_head_mixed_loss(
                        outputs.logits.float(),
                        outputs.ordinal_logits.float(),
                        labels,
                        class_weights=class_weights,
                        **loss_fn_kwargs,
                    )
                else:
                    loss = loss_fn(
                        outputs.logits.float(), labels, class_weights, **loss_fn_kwargs
                    )
                loss = loss / grad_accum_steps

            if grad_scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulation_steps += 1
            running_loss += loss.item() * grad_accum_steps

            if accumulation_steps % grad_accum_steps == 0:
                if grad_scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Keep DDP ranks aligned: non-zero ranks must not race into the next
        # collective while rank-0 is still validating / checkpointing.
        if use_ddp and dist.is_available() and dist.is_initialized():
            dist.barrier()

        if run_validation and epoch % val_interval == 0:
            val_f1, _, _, _ = evaluate(
                model,
                val_loader,
                device,
                use_ordinal=use_ordinal_decode,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                return_both_decodes=True,
                use_hard_cascade=use_hard_cascade,
                cascade_tau=cascade_tau,
                hier_zero_label=hier_zero_label,
            )
            avg_loss = running_loss / max(1, len(train_loader))
            print(f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} | val F1: {val_f1:.5f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                best_state = copy.deepcopy(unwrap_model(model).state_dict())
                if ckpt_dir:
                    save_checkpoint(model, optimizer, scheduler, epoch, fold, best_f1, ckpt_dir)
        elif run_validation:
            avg_loss = running_loss / max(1, len(train_loader))
            print(f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f}")
            if ckpt_dir:
                save_checkpoint(model, optimizer, scheduler, epoch, fold, best_f1, ckpt_dir)

    if use_ddp and dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Load best weights into the unwrapped module so this cannot start a new
    # DDP collective while other ranks have already left the training loop.
    if best_state is not None:
        load_model_state_compat(unwrap_model(model), best_state, strict=False)
    return model, best_f1, best_epoch

def train_full_model(
    model,
    train_loader,
    device,
    lr,
    epochs,
    warmup_ratio,
    class_weights,
    loss_fn=cross_entropy_loss,
    use_amp=True,
    amp_dtype=torch.float16,
    use_grad_scaler=True,
    loss_fn_kwargs=None,
):
    if loss_fn_kwargs is None:
        loss_fn_kwargs = {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    param_dtype = get_first_trainable_param_dtype(model)
    grad_scaler_enabled = bool(use_grad_scaler)
    if grad_scaler_enabled and param_dtype == torch.bfloat16:
        grad_scaler_enabled = False

    scaler = GradScaler('cuda', enabled=grad_scaler_enabled, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(
                    outputs.logits.float(), labels, class_weights, **loss_fn_kwargs
                )

            optimizer.zero_grad()
            if grad_scaler_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Final train epoch {epoch}/{epochs} | loss: {avg_loss:.4f}")

    return model

def main(args):
    set_seed(RANDOM_STATE)
    
    # Initialize DDP if requested
    use_ddp = args.use_ddp and torch.cuda.is_available()
    if use_ddp and not ddp_env_is_ready():
        print("[WARN] --use_ddp set but RANK/WORLD_SIZE/LOCAL_RANK missing; falling back to single process.")
        use_ddp = False

    if use_ddp:
        # Default NCCL timeout is 10 minutes; fold-end export/eval on rank0 can exceed that
        # while other ranks wait, so raise it explicitly (env NCCL_TIMEOUT alone is ignored).
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_process = is_main_process()
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    use_amp = args.use_amp and amp_dtype != torch.float32
    use_grad_scaler = use_amp and amp_dtype == torch.float16

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)

    assert {"ID", "report", "target"}.issubset(train.columns), train.columns
    assert {"ID", "report"}.issubset(test.columns), test.columns

    text_section = getattr(args, "text_section", "full")
    # Keep GroupKFold groups on full cleaned reports for comparable CV splits.
    train["group"] = train["report"].apply(lambda s: stable_hash(clean_text(s)))
    train["text"] = train["report"].apply(lambda s: prepare_report_text(s, text_section))
    test["text"] = test["report"].apply(lambda s: prepare_report_text(s, text_section))

    y = train["target"].astype(int).values
    id2label = {i: i for i in range(NUM_LABELS)}
    X_text = train["text"].values
    groups = train["group"].values

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    all_train_enc = make_encodings(tokenizer, X_text.tolist(), args.max_length)


    # ============ Check if checkpoints exist ============
    ckpt_exists = []
    for fold in range(1, args.n_splits + 1):
        pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
        ckpts = glob.glob(pattern)
        if ckpts:
            ckpt_exists.append(fold)
    
    if len(ckpt_exists) > 4:
        if use_ddp and not main_process:
            return
        # Load checkpoints and generate submission
        test_enc = make_encodings(tokenizer, test["text"].tolist(), args.max_length)
        test_ds = TextDataset(test_enc, labels=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
        
        all_fold_preds = []
        cascade_tau = float(getattr(args, "cascade_tau", 0.5))
        hier_zero_label = int(getattr(args, "hier_zero_label", 0) or 0)
        for fold in range(1, args.n_splits + 1):
            pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
            ckpts = glob.glob(pattern)
            
            if not ckpts:
                continue
            
            # Find best checkpoint
            best_ckpt = None
            best_f1 = -1.0
            best_epoch = -1
            
            for ckpt_path in ckpts:
                ck = torch.load(ckpt_path, map_location="cpu")
                f1 = float(ck.get("best_f1", -1.0))
                ep = int(ck.get("epoch", -1))
                
                if f1 > best_f1 or (f1 == best_f1 and ep > best_epoch):
                    best_f1 = f1
                    best_epoch = ep
                    best_ckpt = ckpt_path
            
            # Load model and checkpoint
            ck = torch.load(best_ckpt, map_location=device)
            state = ck["model_state_dict"]
            is_hier_ckpt = any(
                k == "binary_classifier.weight" or k.endswith(".binary_classifier.weight")
                for k in state.keys()
            )
            is_dual_ckpt = (not is_hier_ckpt) and any(
                k == "ce_classifier.weight"
                or k.endswith(".ce_classifier.weight")
                or k == "ordinal_classifier.weight"
                or k.endswith(".ordinal_classifier.weight")
                for k in state.keys()
            )
            model = build_classification_model(
                args.model_name,
                dual_head=is_dual_ckpt,
                hierarchical=is_hier_ckpt,
                ordinal_ignore_label=None if args.ordinal_ignore_label < 0 else int(args.ordinal_ignore_label),
                zero_label=0 if args.hier_zero_label is None else int(args.hier_zero_label),
            ).to(device)
            load_model_state_compat(model, state, strict=False, log_prefix=f"{os.path.basename(best_ckpt)}: ")
            model.eval()

            fold_preds = []
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    for batch in test_loader:
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        if is_hier_ckpt:
                            pred = hard_cascade_from_outputs(
                                outputs, tau=cascade_tau, zero_label=hier_zero_label
                            )
                        else:
                            pred = torch.argmax(outputs.logits, dim=-1)
                        fold_preds.append(pred.cpu().numpy())

            all_fold_preds.append(np.concatenate(fold_preds, axis=0))
            torch.cuda.empty_cache()

        # Majority vote across folds
        stacked = np.stack(all_fold_preds, axis=0)
        pred_ids = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=NUM_LABELS).argmax(),
            axis=0,
            arr=stacked,
        )
        pred_test = [id2label[int(i)] for i in pred_ids]
        
        submission = pd.DataFrame({
            "ID": test["ID"],
            "target": pred_test,
        })

        submission.to_csv(args.sub_path, index=False)
        print("Saved:", args.sub_path)
        return

    counts = np.bincount(y, minlength=NUM_LABELS)
    class_weights_np = len(y) / (NUM_LABELS * np.maximum(counts, 1))
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    # Select loss function / mix mode
    loss_fn = cross_entropy_loss
    loss_fn_kwargs = {}
    use_dual_head = False
    use_hierarchical = False
    use_ordinal_decode = False
    use_hard_cascade = False
    cascade_tau = 0.5
    ordinal_ignore_label = None if args.ordinal_ignore_label < 0 else int(args.ordinal_ignore_label)
    hier_zero_label = int(args.hier_zero_label)

    if args.loss_type == "ordinal":
        loss_fn = ordinal_loss
        use_ordinal_decode = True
        loss_fn_kwargs = {"ordinal_ignore_label": ordinal_ignore_label}
    elif args.loss_type == "focal":
        loss_fn = focal_loss
        loss_fn_kwargs = {"gamma": args.focal_gamma}
    elif args.loss_type == "hierarchical":
        use_hierarchical = True
        use_hard_cascade = True
        mix_lambda = float(args.mix_lambda)
        cascade_tau = float(args.cascade_tau)
        # Hierarchical already excludes zero from ordinal; keep ignore aligned.
        ordinal_ignore_label = hier_zero_label
        loss_fn = hierarchical_birads_loss
        loss_fn_kwargs = {
            "mix_lambda": mix_lambda,
            "zero_label": hier_zero_label,
        }
    elif args.loss_type == "mixed":
        use_dual_head = True
        mix_lambda = float(args.mix_lambda)
        # λ=0: only ordinal head is trained → ordinal-threshold decode, not CE argmax.
        use_ordinal_decode = mix_lambda <= 0.0 and ordinal_ignore_label is None
        loss_fn = dual_head_mixed_loss
        loss_fn_kwargs = {
            "mix_lambda": mix_lambda,
            "use_focal": bool(args.loss_weight_focal > 0),
            "gamma": args.focal_gamma,
            "ordinal_ignore_label": ordinal_ignore_label,
        }

    gkf = GroupKFold(n_splits=args.n_splits)
    oof_pred = np.empty(len(train), dtype=int)
    fold_scores = []

    # Parse which folds to train
    try:
        train_folds = parse_train_folds(args.train_folds, args.n_splits)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_text, y, groups=groups), 1):
        if fold not in train_folds:
            continue

        if main_process:
            print(f"\nFold {fold}/{args.n_splits}")

        y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
        y_va = torch.tensor(y[va_idx], dtype=torch.long)

        tr_enc = select_encodings(all_train_enc, tr_idx)
        va_enc = select_encodings(all_train_enc, va_idx)

        tr_ds = TextDataset(tr_enc, y_tr)
        va_ds = TextDataset(va_enc, y_va)

        if use_ddp:
            tr_sampler = DistributedSampler(
                tr_ds,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=RANDOM_STATE + fold,
                drop_last=True,
            )
            tr_loader = DataLoader(
                tr_ds,
                batch_size=args.batch_size,
                sampler=tr_sampler,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
                drop_last=True,
            )
            if main_process:
                va_loader = DataLoader(
                    va_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=args.num_workers > 0,
                )
            else:
                va_loader = None
        else:
            tr_loader = DataLoader(
                tr_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
            )
            va_loader = DataLoader(
                va_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
            )

        model = build_classification_model(
            args.model_name,
            dual_head=use_dual_head,
            hierarchical=use_hierarchical,
            ordinal_ignore_label=ordinal_ignore_label if use_dual_head else None,
            zero_label=hier_zero_label,
        )

        # DataParallel only when not using DDP; DDP wraps below for true multi-GPU sync.
        model = maybe_parallelize(
            model,
            device,
            use_data_parallel=(not use_ddp and not args.no_data_parallel),
        )
        if use_ddp:
            # Ordinal head is always graph-connected (BCE or exclusive zero loss
            # on all-ignored batches), so unused-parameter search is not needed.
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )

        model, best_f1, best_epoch = train_fold(
            model=model,
            train_loader=tr_loader,
            val_loader=va_loader,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            class_weights=class_weights,
            loss_fn=loss_fn,
            fold=fold,
            ckpt_dir=args.ckpt_dir,
            resume=args.resume,
            use_amp=use_amp,
            grad_accum_steps=args.grad_accum_steps,
            val_interval=args.val_interval,
            use_compile=args.use_compile,
            use_ddp=use_ddp,
            run_validation=main_process,
            amp_dtype=amp_dtype,
            use_grad_scaler=use_grad_scaler,
            loss_fn_kwargs=loss_fn_kwargs,
            use_ordinal_decode=use_ordinal_decode,
            use_dual_head=use_dual_head,
            use_hierarchical=use_hierarchical,
            use_hard_cascade=use_hard_cascade,
            cascade_tau=cascade_tau,
            hier_zero_label=hier_zero_label,
        )

        if use_ddp:
            # All ranks reach here together (train_fold ends with a barrier).
            # Destroy before rank-0 post I/O so nobody waits inside NCCL.
            dist.destroy_process_group()

        post_done_flag = os.path.join(args.ckpt_dir, f".fold_{fold}_post_done")
        if main_process:
            if os.path.exists(post_done_flag):
                os.remove(post_done_flag)

            val_f1, pred_va, _, _ = evaluate(
                unwrap_model(model),
                va_loader,
                device,
                use_ordinal=use_ordinal_decode,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                return_both_decodes=True,
                use_hard_cascade=use_hard_cascade,
                cascade_tau=cascade_tau,
                hier_zero_label=hier_zero_label,
            )
            oof_pred[va_idx] = pred_va
            fold_scores.append(val_f1)

            print(f"Fold {fold} best F1: {best_f1:.5f} @ epoch {best_epoch}")

            best_ckpt_name = f"fold_{fold}_epoch_{best_epoch}.pt"
            for ckpt_path in glob.glob(os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")):
                if os.path.basename(ckpt_path) == best_ckpt_name:
                    continue
                try:
                    os.remove(ckpt_path)
                except OSError:
                    pass

            with open(post_done_flag, "w", encoding="utf-8") as f:
                f.write("ok\n")
                f.flush()
                os.fsync(f.fileno())
        elif args.train_folds and "," not in args.train_folds and "-" not in args.train_folds:
            # Single-fold job: non-zero ranks can exit after destroy.
            pass
        else:
            while not os.path.exists(post_done_flag):
                time.sleep(2.0)

        del model, tr_loader, va_loader, tr_ds, va_ds
        if "tr_sampler" in locals():
            del tr_sampler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Re-init only when more DDP folds remain in this process.
        remaining = [f for f in train_folds if f > fold]
        if use_ddp and remaining:
            while not main_process and not os.path.exists(post_done_flag):
                time.sleep(2.0)
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
            torch.cuda.set_device(local_rank)
            if main_process and os.path.exists(post_done_flag):
                os.remove(post_done_flag)

    if not main_process:
        return

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    oof_score = f1_score(y, oof_pred, average="macro")
    print(f"\nCV mean: {cv_mean:.5f} | CV std: {cv_std:.5f} | OOF F1: {oof_score:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT baseline for SPR2026 report classification (FAST + Checkpoints)")
    # parser.add_argument("--train_path", type=str, default="/kaggle/input/competitions/spr-2026-mammography-report-classification/train.csv")
    parser.add_argument("--train_path", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/train_new.csv")
    # parser.add_argument("--train_path", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/train_new_en.csv")
    parser.add_argument("--test_path", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/test.csv")
    parser.add_argument("--sub_path", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/submission.csv")
    parser.add_argument("--save_path", type=str, default="submission.csv")

    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-cased")
    # parser.add_argument("--model_name", type=str, default="Simonlee711/Clinical_ModernBERT")
    # parser.add_argument("--model_name", type=str, default="pucpr/biobertpt-all")
    # parser.add_argument("--model_name", type=str, default="neuralmind/bert-base-portuguese-cased")
    # parser.add_argument("--model_name", type=str, default="PORTULAN/albertina-900m-portuguese-ptpt-encoder")

    # parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_data_parallel", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=0,
        help="Final training epochs. <=0 means auto from CV best epochs.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    
    # Loss function options
    parser.add_argument("--loss_type", type=str, default="cross_entropy",
                        choices=["cross_entropy", "ordinal", "focal", "mixed", "hierarchical"],
                        help="Loss function type: 'cross_entropy', 'ordinal', 'focal', 'mixed', or 'hierarchical'")
    parser.add_argument(
        "--mix_lambda",
        type=float,
        default=0.9,
        help=(
            "Mix weight λ. For mixed: L=λ*CE+(1-λ)*ordinal (dual-head). "
            "For hierarchical: L=λ*BCE(is_zero)+(1-λ)*ordinal(diagnostic)."
        ),
    )
    parser.add_argument("--loss_weight_focal", type=float, default=0.0,
                        help="If >0 with mixed loss, use focal as the dual-head primary loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focusing parameter gamma for focal loss (default: 2.0)")
    parser.add_argument(
        "--ordinal_ignore_label",
        type=int,
        default=0,
        help=(
            "Label to exclude from ordinal loss because it is non-diagnostic "
            "(default: 0 for insufficient information; use -1 to disable)"
        ),
    )
    parser.add_argument(
        "--hier_zero_label",
        type=int,
        default=0,
        help="For loss_type=hierarchical: which label is the non-diagnostic/zero class (default: 0)",
    )
    parser.add_argument(
        "--cascade_tau",
        type=float,
        default=0.5,
        help="Hard-cascade threshold on p0=sigmoid(binary). Predict 0 if p0>=tau.",
    )
    
    # Speedup options
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision training for faster throughput on Tensor Core GPUs")
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Precision type for mixed precision training")
    parser.add_argument("--use_compile", action="store_true", default=False, help="Use torch.compile for optimization (PyTorch 2.0+)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    
    # Checkpoint options
    parser.add_argument("--ckpt_dir", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/results/google-bert-attention", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from latest checkpoint")
    parser.add_argument("--no_resume", action="store_false", dest="resume", help="Do not resume from checkpoint; start fresh training")
    parser.add_argument("--train_folds", type=str, default="", help="Specify which folds to train (e.g., '1,2,3' or '1-3'). Empty means train all folds.")
    parser.add_argument(
        "--text_section",
        type=str,
        default="full",
        choices=["full", "achados"],
        help="Input text: 'full' report or only Achados/findings section",
    )
    
    parser.add_argument("--use_ddp", action="store_true", default=True, help="Use DistributedDataParallel (DDP) for multi-GPU training (recommended for 2+ GPUs)")

    # args, _ = parser.parse_known_args()
    args = parser.parse_args()
    try:
        main(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

# torchrun --nproc_per_node=2 bertori.py --use_ddp --use_amp 