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
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
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
    # s = str(s).strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # s = re.sub(r"[?\\/]{1} ?", "", s)
    # s = re.sub(r"([.,:;!?])\1+", r"\1", s)
    # s = re.sub(r"[®]", "", s)
    s = _whitespace_re.sub(" ", s)
    s = _newlines_re.sub("\n", s)
    return s



def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ============ Distributed Training Utilities ============
def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed(is_distributed: bool) -> None:
    """Clean up distributed training."""
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

def rank0_print(rank: int, *args, **kwargs):
    """Print only from rank 0 process."""
    if rank == 0:
        print(*args, **kwargs)


def maybe_parallelize(model, device, use_data_parallel=True):
    model = model.to(device)
    if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, fold, best_f1, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}_epoch_{epoch}.pt")

    # Strip DP/DDP wrapper so checkpoints stay portable across single-GPU / DP / DDP.
    while isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    model_state = model.state_dict()

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
def evaluate(
    model,
    loader,
    device,
    use_ordinal=False,
    use_amp=False,
    amp_dtype=torch.float16,
    return_both_decodes=False,
    use_hard_cascade=False,
    cascade_tau=0.5,
    hier_zero_label=0,
):
    """Evaluate macro F1.

    By default returns (f1, preds) for the selected decode mode.
    If return_both_decodes=True, returns
    (f1_primary, preds_primary, f1_argmax, f1_ordinal).
    """
    model.eval()
    preds_argmax = []
    preds_ordinal = []
    preds_hard = []
    trues = []

    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            pred_argmax = torch.argmax(logits, dim=-1)

            # Dual-head models expose a dedicated ordinal head; otherwise use first K-1 logits.
            if getattr(outputs, "ordinal_logits", None) is not None:
                ordinal_logits = outputs.ordinal_logits
                max_class = logits.size(1) - 1
            else:
                ordinal_logits = logits[:, :-1]
                max_class = logits.size(1) - 1
            ordinal_probs = torch.sigmoid(ordinal_logits)
            pred_ordinal = torch.sum(ordinal_probs >= 0.5, dim=1).long()
            pred_ordinal = torch.clamp(pred_ordinal, 0, max_class)

            if use_hard_cascade and getattr(outputs, "binary_logits", None) is not None:
                pred_hard = hard_cascade_from_outputs(
                    outputs, tau=cascade_tau, zero_label=hier_zero_label
                )
            else:
                pred_hard = pred_argmax

        preds_argmax.extend(pred_argmax.cpu().numpy().tolist())
        preds_ordinal.extend(pred_ordinal.cpu().numpy().tolist())
        preds_hard.extend(pred_hard.cpu().numpy().tolist())
        trues.extend(labels.cpu().numpy().tolist())

    f1_argmax = f1_score(trues, preds_argmax, average="macro")
    f1_ordinal = f1_score(trues, preds_ordinal, average="macro")
    f1_hard = f1_score(trues, preds_hard, average="macro")
    if use_hard_cascade:
        f1_primary, preds_primary = f1_hard, np.array(preds_hard, dtype=int)
    elif use_ordinal:
        f1_primary, preds_primary = f1_ordinal, np.array(preds_ordinal, dtype=int)
    else:
        f1_primary, preds_primary = f1_argmax, np.array(preds_argmax, dtype=int)

    if return_both_decodes:
        return f1_primary, preds_primary, f1_argmax, f1_ordinal
    return f1_primary, preds_primary


@torch.no_grad()
def extract_ce_logits(outputs):
    """Support both plain logit tensors and DualHeadOutput-like objects."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def logits_to_ordinal_class(logits_or_ordinal, num_classes=None):
    """
    Decode ordinal threshold logits to a class index.
    If a full K-way tensor is passed, use the first K-1 columns.
    """
    if num_classes is None:
        # shared K-way logits -> thresholds are first K-1
        ordinal_logits = logits_or_ordinal[:, :-1]
        max_class = logits_or_ordinal.size(1) - 1
    else:
        ordinal_logits = logits_or_ordinal
        max_class = num_classes - 1
    ordinal_probs = torch.sigmoid(ordinal_logits)
    pred = torch.sum(ordinal_probs >= 0.5, dim=1).long()
    return torch.clamp(pred, 0, max_class)


def hard_cascade_from_outputs(outputs, tau=0.5, zero_label=0):
    """
    Hard cascade decode for hierarchical BI-RADS heads:
      if sigmoid(binary) >= tau -> zero_label
      else -> ordinal remapped class mapped back around zero_label
    Assumes zero_label=0 and ordinal head has (K-2) thresholds.
    """
    binary_logits = getattr(outputs, "binary_logits", None)
    ordinal_logits = getattr(outputs, "ordinal_logits", None)
    if binary_logits is None or ordinal_logits is None:
        raise ValueError("hard cascade requires binary_logits and ordinal_logits")
    if int(zero_label) != 0:
        raise ValueError("hard_cascade_from_outputs currently supports zero_label=0 only")
    if binary_logits.ndim > 1:
        binary_logits = binary_logits.squeeze(-1)
    p0 = torch.sigmoid(binary_logits.float())
    n_diag = ordinal_logits.size(1) + 1  # remapped diagnostic classes
    ord_remapped = logits_to_ordinal_class(ordinal_logits, num_classes=n_diag)
    ord_label = ord_remapped + 1  # map 0..K-2 -> labels 1..K-1
    return torch.where(p0 >= float(tau), torch.zeros_like(ord_label), ord_label)


def distributed_evaluate(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
    is_distributed,
    rank,
    world_size,
    use_ordinal=False,
    use_hard_cascade=False,
    cascade_tau=0.5,
    hier_zero_label=0,
):
    """Evaluate on all ranks by partitioning dataset and aggregating predictions."""

    def _predict_batch(outputs):
        if use_hard_cascade:
            return hard_cascade_from_outputs(
                outputs, tau=cascade_tau, zero_label=hier_zero_label
            )
        ce_logits = extract_ce_logits(outputs)
        if use_ordinal:
            ordinal_logits = getattr(outputs, "ordinal_logits", None)
            if ordinal_logits is not None:
                return logits_to_ordinal_class(ordinal_logits, num_classes=ce_logits.size(1))
            return logits_to_ordinal_class(ce_logits)
        return torch.argmax(ce_logits, dim=-1)

    # Unwrap DDP: eval must not run reducer collectives (uneven shard sizes desync ranks).
    eval_model = model.module if hasattr(model, "module") else model
    # DataLoader workers + CUDA/NCCL under DDP are a common hang source.
    eval_workers = 0 if is_distributed else num_workers

    if not is_distributed:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=eval_workers > 0,
        )
        preds = []
        trues = []
        eval_model.eval()
        with torch.no_grad():
            for batch in loader:
                labels = batch.get("labels")
                batch_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = eval_model(
                    clinical_input_ids=batch_device["clinical_input_ids"],
                    clinical_attention_mask=batch_device["clinical_attention_mask"],
                    findings_input_ids=batch_device["findings_input_ids"],
                    findings_attention_mask=batch_device["findings_attention_mask"],
                    comparative_input_ids=batch_device["comparative_input_ids"],
                    comparative_attention_mask=batch_device["comparative_attention_mask"],
                )
                pred = _predict_batch(outputs)
                preds.extend(pred.cpu().numpy().tolist())
                if labels is not None:
                    trues.extend(labels.cpu().numpy().tolist())

        if trues:
            val_f1 = float(f1_score(trues, preds, average="macro"))
            return val_f1, np.array(preds)
        return 0.0, np.array(preds)

    eval_model.eval()
    local_indices = np.arange(len(dataset))[rank::world_size]
    local_subset = Subset(dataset, local_indices.tolist())
    local_loader = DataLoader(
        local_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=torch.cuda.is_available(),
    )

    local_preds = []
    local_trues = []

    with torch.no_grad():
        for batch in local_loader:
            labels = batch.get("labels")
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = eval_model(
                clinical_input_ids=batch_device["clinical_input_ids"],
                clinical_attention_mask=batch_device["clinical_attention_mask"],
                findings_input_ids=batch_device["findings_input_ids"],
                findings_attention_mask=batch_device["findings_attention_mask"],
                comparative_input_ids=batch_device["comparative_input_ids"],
                comparative_attention_mask=batch_device["comparative_attention_mask"],
            )
            pred = _predict_batch(outputs)

            local_preds.extend(pred.cpu().numpy().tolist())
            if labels is not None:
                local_trues.extend(labels.cpu().numpy().tolist())

    dist.barrier()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, (local_indices.tolist(), local_preds, local_trues))

    pred_full = np.array([], dtype=int)
    val_f1 = 0.0
    if rank == 0:
        pred_full = np.empty(len(dataset), dtype=int)
        true_full = np.empty(len(dataset), dtype=int)
        for idxs, preds, trues in gathered:
            if idxs:
                pred_full[idxs] = np.array(preds)
                true_full[idxs] = np.array(trues)

        if true_full.size > 0:
            val_f1 = float(f1_score(true_full, pred_full, average="macro"))

    f1_tensor = torch.tensor([val_f1], dtype=torch.float32, device=device)
    dist.broadcast(f1_tensor, src=0)
    return float(f1_tensor.item()), pred_full

def save_confusion_matrix_as_table(cm, id2label, save_path):
    """Save confusion matrix as CSV table."""
    label_names = [str(id2label[i]) for i in range(len(id2label))]
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.index.name = 'True Label'
    cm_df.to_csv(save_path)


def save_classification_report_as_table(y_true, y_pred, id2label, save_path):
    """Save classification report as CSV table."""
    label_ids = list(range(len(id2label)))
    label_names = [str(id2label[i]) for i in label_ids]
    report_dict = classification_report(
        y_true, y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(save_path)



def compute_classification_loss(logits, labels, class_weights=None, use_focal_loss=False, focal_gamma=2.0):
    """
    Compute classification loss with optional focal loss.
    """
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    
    if not use_focal_loss:
        return ce_loss.mean()
    
    pt = torch.exp(-ce_loss)
    focal_weight = (1.0 - pt).pow(focal_gamma)
    focal_loss = focal_weight * ce_loss
    
    return focal_loss.mean()


def clean_and_isolate_text(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    match = re.search(r'achados:(.*?)(análise comparativa:|$)', s, flags=re.DOTALL)
    if match: s = match.group(1).strip()
    s = re.sub(r'\b(?:cm|mm)\b', ' \1 ', s)
    s = re.sub(r'[0-9]+,[0-9]+', 'NUM', s)
    s = re.sub(r'[0-9]+', 'NUM', s)
    return s





def _build_attention_token_labels_and_indices(tokenizer, cli_ids, cli_mask, find_ids, find_mask, comp_ids, comp_mask):
    """
    Build labels and exact index mapping into the concatenated sequence:
    [G_CLS] + clinical(max_length) + findings(max_length) + comparative(max_length)

    Only non-PAD token positions are included.
    """
    labels = ["[G_CLS]"]
    keep_indices = [0]

    def _append_segment(input_ids, attention_mask, prefix, base_offset):
        nonlocal labels, keep_indices
        valid_pos = torch.nonzero(attention_mask > 0, as_tuple=False).squeeze(-1).tolist()
        if not isinstance(valid_pos, list):
            valid_pos = [int(valid_pos)]
        if len(valid_pos) == 0:
            return

        token_ids = input_ids[valid_pos].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        labels.extend([f"{prefix}:{tok}" for tok in tokens])
        keep_indices.extend([base_offset + int(p) for p in valid_pos])

    # +1 offset because index 0 is global CLS prepended before concatenated segment tokens.
    # Segment lengths may differ (e.g. clinical=40, findings=160, comparative=40).
    cli_len = int(cli_ids.numel())
    find_len = int(find_ids.numel())
    _append_segment(cli_ids, cli_mask, "CLI", base_offset=1)
    _append_segment(find_ids, find_mask, "FIND", base_offset=1 + cli_len)
    _append_segment(comp_ids, comp_mask, "COMP", base_offset=1 + cli_len + find_len)

    return labels, keep_indices


def _merge_wordpiece_labels_and_attention(labels, attn_mean):
    """
    Merge consecutive WordPiece continuation tokens (##xxx) for visualization.
    Attention matrix is block-averaged accordingly.
    """
    if len(labels) <= 1:
        return labels, attn_mean

    merged_labels = []
    group_ids = []
    current_group = -1

    for label in labels:
        if ":" not in label:
            current_group += 1
            merged_labels.append(label)
            group_ids.append(current_group)
            continue

        prefix, tok = label.split(":", 1)
        can_merge = (
            tok.startswith("##")
            and len(merged_labels) > 0
            and merged_labels[-1].startswith(prefix + ":")
        )

        if can_merge:
            merged_labels[-1] = merged_labels[-1] + tok[2:]
            group_ids.append(current_group)
        else:
            current_group += 1
            if tok.startswith("##"):
                tok = tok[2:]
            merged_labels.append(f"{prefix}:{tok}")
            group_ids.append(current_group)

    num_groups = max(group_ids) + 1
    if num_groups == len(labels):
        return labels, attn_mean

    # Block-average original attention to merged-token level.
    merged_attn = attn_mean.new_zeros((num_groups, num_groups))
    group_to_positions = [[] for _ in range(num_groups)]
    for pos, gid in enumerate(group_ids):
        group_to_positions[gid].append(pos)

    for i in range(num_groups):
        row_idx = torch.tensor(group_to_positions[i], device=attn_mean.device, dtype=torch.long)
        for j in range(num_groups):
            col_idx = torch.tensor(group_to_positions[j], device=attn_mean.device, dtype=torch.long)
            merged_attn[i, j] = attn_mean.index_select(0, row_idx).index_select(1, col_idx).mean()

    return merged_labels, merged_attn


@torch.no_grad()
def save_attention_heatmap_for_sample(
    model,
    tokenizer,
    clinical_text,
    findings_text,
    comparative_text,
    max_length,
    device,
    save_path,
    max_tokens=120,
    merge_wordpieces=True,
):
    """Extract last-layer attention (first sample) and save heatmap with token labels."""
    import matplotlib.pyplot as plt
    from data_segmentation import make_segmented_encodings

    model.eval()

    enc_list = make_segmented_encodings(
        tokenizer,
        [clinical_text],
        [findings_text],
        [comparative_text],
        max_length,
    )
    cli_enc, find_enc, comp_enc = enc_list[0]

    cli_ids = cli_enc["input_ids"].unsqueeze(0).to(device)
    cli_mask = cli_enc["attention_mask"].unsqueeze(0).to(device)
    find_ids = find_enc["input_ids"].unsqueeze(0).to(device)
    find_mask = find_enc["attention_mask"].unsqueeze(0).to(device)
    comp_ids = comp_enc["input_ids"].unsqueeze(0).to(device)
    comp_mask = comp_enc["attention_mask"].unsqueeze(0).to(device)

    outputs = model(
        clinical_input_ids=cli_ids,
        clinical_attention_mask=cli_mask,
        findings_input_ids=find_ids,
        findings_attention_mask=find_mask,
        comparative_input_ids=comp_ids,
        comparative_attention_mask=comp_mask,
        return_last_layer_attn=True,
    )

    if not isinstance(outputs, tuple) or len(outputs) < 2:
        raise RuntimeError("Model did not return attention weights. Check return_last_layer_attn flow.")

    _, last_attn = outputs
    if last_attn is None:
        raise RuntimeError("Last layer attention is None.")

    # last_attn expected shape: (batch, heads, seq, seq) with average_attn_weights=False
    attn_first_sample = last_attn[0]  # (heads, seq, seq)
    attn_mean = attn_first_sample.mean(dim=0)  # (seq, seq)

    token_labels, keep_indices = _build_attention_token_labels_and_indices(
        tokenizer,
        cli_enc["input_ids"],
        cli_enc["attention_mask"],
        find_enc["input_ids"],
        find_enc["attention_mask"],
        comp_enc["input_ids"],
        comp_enc["attention_mask"],
    )

    seq_len = attn_mean.size(0)
    valid_pairs = [(idx, label) for idx, label in zip(keep_indices, token_labels) if 0 <= idx < seq_len]
    if len(valid_pairs) == 0:
        raise RuntimeError("No valid non-PAD attention indices found for visualization.")

    keep_indices = [p[0] for p in valid_pairs]
    token_labels = [p[1] for p in valid_pairs]

    if max_tokens and len(token_labels) > max_tokens:
        keep_indices = keep_indices[:max_tokens]
        token_labels = token_labels[:max_tokens]

    index_keep_t = torch.tensor(keep_indices, device=attn_mean.device, dtype=torch.long)
    attn_mean = attn_mean.index_select(0, index_keep_t).index_select(1, index_keep_t)

    merged_labels_for_txt = token_labels  # Keep original for txt (before merge)
    if merge_wordpieces:
        token_labels, attn_mean = _merge_wordpiece_labels_and_attention(token_labels, attn_mean)

    attn_mean_cpu = attn_mean.detach().cpu()
    attn_np = attn_mean_cpu.numpy()

    n = len(token_labels)
    fig_side = max(8, min(26, int(0.28 * n) + 6))
    plt.figure(figsize=(fig_side, fig_side))

    # Use robust color scaling to avoid near-uniform heatmap when values are in a narrow band.
    vmin = float(np.percentile(attn_np, 2))
    vmax = float(np.percentile(attn_np, 98))
    if vmax <= vmin:
        vmin, vmax = None, None

    plt.imshow(attn_np, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar()

    if n <= 80:
        ticks = np.arange(n)
    else:
        step = max(1, n // 40)
        ticks = np.arange(0, n, step)

    plt.xticks(ticks, [token_labels[i] for i in ticks], rotation=90, fontsize=7)
    plt.yticks(ticks, [token_labels[i] for i in ticks], fontsize=7)
    plt.title("Last-layer mean attention ([G_CLS] + non-PAD report tokens)")
    plt.tight_layout()

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close()

    # ===== Save txt report with token labels and attention values =====
    txt_path = save_path.replace('.png', '_attention_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("ATTENTION VISUALIZATION REPORT\n")
        f.write("=" * 100 + "\n\n")

        # Original texts
        f.write("ORIGINAL REPORT SEGMENTS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Clinical Indication:\n{clinical_text}\n\n")
        f.write(f"Findings:\n{findings_text}\n\n")
        f.write(f"Comparative Analysis:\n{comparative_text}\n")
        f.write("-" * 100 + "\n\n")

        # Reconstructed text from tokens
        f.write("RECONSTRUCTED TEXT FROM TOKENS:\n")
        f.write("-" * 100 + "\n")
        reconstructed = _reconstruct_text_from_labels(token_labels)
        f.write(reconstructed + "\n")
        f.write("-" * 100 + "\n\n")

        # Token-level attention analysis
        f.write("TOKEN-LEVEL ATTENTION ANALYSIS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Index':<6} {'Token/Label':<30} {'Avg Attention':<15} {'Max Attention':<15}\n")
        f.write("-" * 100 + "\n")

        for i, label in enumerate(token_labels):
            avg_attn = float(attn_np[i, :].mean())
            max_attn = float(attn_np[i, :].max())
            f.write(f"{i:<6} {label:<30} {avg_attn:<15.6f} {max_attn:<15.6f}\n")

        f.write("-" * 100 + "\n\n")

        # Attention matrix summary
        f.write("ATTENTION MATRIX STATISTICS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Matrix shape: {attn_np.shape}\n")
        f.write(f"Min attention: {float(attn_np.min()):.6f}\n")
        f.write(f"Max attention: {float(attn_np.max()):.6f}\n")
        f.write(f"Mean attention: {float(attn_np.mean()):.6f}\n")
        f.write(f"Std attention: {float(attn_np.std()):.6f}\n")
        f.write("-" * 100 + "\n\n")

        # Top attention patterns (CLS attention to tokens)
        if "[G_CLS]" in token_labels:
            cls_idx = token_labels.index("[G_CLS]")
            cls_attention = attn_np[cls_idx, :]
            top_attn_idx = np.argsort(cls_attention)[::-1][:10]
            f.write("[G_CLS] TOKEN TOP ATTENTION TARGETS:\n")
            f.write("-" * 100 + "\n")
            for rank, idx in enumerate(top_attn_idx, 1):
                f.write(f"Rank {rank}: {token_labels[idx]:<30} attention={cls_attention[idx]:.6f}\n")
            f.write("-" * 100 + "\n\n")
        
        # Append serialized attention matrix for regeneration
        _append_attention_matrix_to_txt(f, token_labels, attn_np)



def _reconstruct_text_from_labels(token_labels):
    """
    Reconstruct readable text from token labels by removing prefixes and merging subword tokens.
    Properly handles space insertion between regular tokens and subword tokens (##).
    """
    segments = {}  # {segment_name: [(token, is_subword)]}

    for label in token_labels:
        if label == "[G_CLS]":
            continue

        if ":" in label:
            prefix, token = label.split(":", 1)
            if prefix not in segments:
                segments[prefix] = []
            
            # Check if this is a subword token
            is_subword = token.startswith("##")
            if is_subword:
                token = token[2:]  # Remove ## prefix
            
            # Skip special tokens but track them for proper spacing
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                segments[prefix].append((token, is_subword))

    # Reconstruct text for each segment
    reconstructed_lines = []
    for seg_name in ["CLI", "FIND", "COMP"]:
        if seg_name in segments:
            seg_label = {
                "CLI": "Clinical Indication",
                "FIND": "Findings",
                "COMP": "Comparative Analysis",
            }[seg_name]

            tokens_info = segments[seg_name]
            # Build text with proper spacing
            text_parts = []
            for token, is_subword in tokens_info:
                if is_subword:
                    # Subword tokens are directly appended (no space)
                    if text_parts:
                        text_parts[-1] += token
                else:
                    # Regular tokens are added as separate parts (space will separate them)
                    text_parts.append(token)

            text = " ".join(text_parts)
            reconstructed_lines.append(f"{seg_label}: {text}")

    return "\n".join(reconstructed_lines)


def _append_attention_matrix_to_txt(f, token_labels, attn_np):
    """
    Append serialized attention matrix to txt file for later reconstruction.
    Saves in a parseable format that can be used to regenerate PNG.
    """
    import json
    
    f.write("=" * 100 + "\n")
    f.write("SERIALIZED ATTENTION MATRIX (for PNG regeneration)\n")
    f.write("=" * 100 + "\n\n")
    
    # Save metadata
    f.write("METADATA:\n")
    f.write("-" * 100 + "\n")
    f.write(f"Token count: {len(token_labels)}\n")
    f.write(f"Matrix shape: {attn_np.shape}\n")
    f.write(f"Data type: float32\n")
    f.write("-" * 100 + "\n\n")
    
    # Save token labels as JSON
    f.write("TOKEN_LABELS:\n")
    f.write("-" * 100 + "\n")
    token_json = json.dumps(token_labels, ensure_ascii=False, indent=2)
    f.write(token_json + "\n")
    f.write("-" * 100 + "\n\n")
    
    # Save attention matrix as CSV
    f.write("ATTENTION_MATRIX_CSV:\n")
    f.write("-" * 100 + "\n")
    f.write("# Format: comma-separated values, one row per line\n")
    f.write("# To reconstruct: np.loadtxt(StringIO(csv_block), delimiter=',')\n\n")
    
    for i, row in enumerate(attn_np):
        row_str = ",".join([f"{v:.6e}" for v in row])
        f.write(row_str + "\n")
    
    f.write("-" * 100 + "\n\n")


def regenerate_attention_heatmap_from_txt(txt_path, output_png_path, max_tokens=120):
    """
    Regenerate attention heatmap from saved txt report.
    Reads serialized attention matrix and token labels from txt.
    """
    import json
    from io import StringIO
    import matplotlib.pyplot as plt
    
    rank0_print(0, f"[INFO] Regenerating heatmap from txt: {txt_path}")
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract TOKEN_LABELS section
        token_start = content.find("TOKEN_LABELS:\n") + len("TOKEN_LABELS:\n")
        token_end = content.find("-" * 100 + "\n\n", token_start)
        token_section = content[token_start:token_end].strip()
        token_labels = json.loads(token_section)
        
        # Extract ATTENTION_MATRIX_CSV section
        attn_start = content.find("ATTENTION_MATRIX_CSV:\n") + len("ATTENTION_MATRIX_CSV:\n")
        attn_start = content.find("\n", attn_start) + 1  # Skip comment line
        attn_start = content.find("\n", attn_start) + 1  # Skip blank line
        attn_end = content.find("-" * 100 + "\n\n", attn_start)
        attn_section = content[attn_start:attn_end].strip()
        
        # Parse CSV
        attn_np = np.loadtxt(StringIO(attn_section), delimiter=',')
        
        # Limit tokens if specified
        if max_tokens and len(token_labels) > max_tokens:
            attn_np = attn_np[:max_tokens, :max_tokens]
            token_labels = token_labels[:max_tokens]
        
        # Generate heatmap
        n = len(token_labels)
        fig_side = max(8, min(26, int(0.28 * n) + 6))
        plt.figure(figsize=(fig_side, fig_side))
        
        vmin = float(np.percentile(attn_np, 2))
        vmax = float(np.percentile(attn_np, 98))
        if vmax <= vmin:
            vmin, vmax = None, None
        
        plt.imshow(attn_np, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap='viridis')
        plt.colorbar()
        
        if n <= 80:
            ticks = np.arange(n)
        else:
            step = max(1, n // 40)
            ticks = np.arange(0, n, step)
        
        plt.xticks(ticks, [token_labels[i] for i in ticks], rotation=90, fontsize=7)
        plt.yticks(ticks, [token_labels[i] for i in ticks], fontsize=7)
        plt.title("Last-layer mean attention ([G_CLS] + non-PAD report tokens)")
        plt.tight_layout()
        
        plt.savefig(output_png_path, dpi=220)
        plt.close()
        
        rank0_print(0, f"[INFO] Successfully regenerated: {output_png_path}")
        return True
        
    except Exception as e:
        rank0_print(0, f"[WARN] Failed to regenerate heatmap from txt: {e}")
        return False


def maybe_extract_attention_after_run(
    args,
    tokenizer,
    train_df,
    test_df,
    num_labels,
    device,
    rank,
    preferred_ckpt_path=None,
    preferred_fold=None,
):
    """Optionally extract and save attention heatmap from the latest available fold checkpoint."""
    if not args.extract_attention_after_train or rank != 0:
        return

    from data_segmentation import TokenLevelSegmentedBert, split_report_by_sections

    selected_fold = None
    selected_ckpt = None

    if preferred_ckpt_path and os.path.exists(preferred_ckpt_path):
        selected_ckpt = preferred_ckpt_path
        selected_fold = preferred_fold
    else:
        for fold in range(1, args.n_splits + 1):
            pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
            ckpts = glob.glob(pattern)
            if ckpts:
                latest_ckpt = max(ckpts, key=os.path.getctime)
                if selected_ckpt is None or os.path.getctime(latest_ckpt) > os.path.getctime(selected_ckpt):
                    selected_ckpt = latest_ckpt
                    selected_fold = fold

    if selected_ckpt is None:
        rank0_print(rank, "[WARN] Attention extraction skipped: no fold checkpoint found.")
        return

    if args.attention_source == "train":
        if len(train_df) == 0:
            rank0_print(rank, "[WARN] Attention extraction skipped: train dataframe is empty.")
            return
        sample_idx = max(0, min(args.attention_sample_index, len(train_df) - 1))
        row = train_df.iloc[sample_idx]
        cli_text = "" if pd.isna(row.get("clinical_indication", "")) else str(row.get("clinical_indication", ""))
        find_text = "" if pd.isna(row.get("findings", "")) else str(row.get("findings", ""))
        comp_text = "" if pd.isna(row.get("comparative_analysis", "")) else str(row.get("comparative_analysis", ""))
    else:
        if len(test_df) == 0:
            rank0_print(rank, "[WARN] Attention extraction skipped: test dataframe is empty.")
            return
        sample_idx = max(0, min(args.attention_sample_index, len(test_df) - 1))
        row = test_df.iloc[sample_idx]
        report_text = "" if pd.isna(row.get("report", "")) else str(row.get("report", ""))
        sections = split_report_by_sections(report_text)
        cli_text = sections.get('clinical_indication', '')
        find_text = sections.get('findings', '')
        comp_text = sections.get('comparative_analysis', '')

    vis_model = TokenLevelSegmentedBert(
        args.model_name,
        num_labels=num_labels
    )
    ckpt_data = torch.load(selected_ckpt, map_location=device)
    vis_model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
    vis_model = vis_model.to(device)

    if args.attention_save_path:
        save_path = args.attention_save_path
    else:
        save_path = os.path.join(
            args.ckpt_dir,
            f"attention_last_layer_fold{selected_fold}_{args.attention_source}_idx{sample_idx}.png",
        )

    attn_max_length = getattr(args, "max_length", None)
    if attn_max_length is None:
        attn_max_length = max(
            int(getattr(args, "clinical_max_length", 256)),
            int(getattr(args, "findings_max_length", 256)),
            int(getattr(args, "comparative_max_length", 256)),
        )
    save_attention_heatmap_for_sample(
        model=vis_model,
        tokenizer=tokenizer,
        clinical_text=cli_text,
        findings_text=find_text,
        comparative_text=comp_text,
        max_length=attn_max_length,
        device=device,
        save_path=save_path,
        max_tokens=args.attention_max_tokens,
        merge_wordpieces=args.attention_merge_wordpieces,
    )
    rank0_print(
        rank,
        f"[INFO] Saved attention heatmap from fold {selected_fold}, sample idx {sample_idx} ({args.attention_source}) to: {save_path}",
    )
