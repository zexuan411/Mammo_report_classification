# Segmented BERT with Cross-Attention fusion + Learnable CLS-pool gate (v2, 2026-08-11)
import os
import re
import sys
import copy
import gc
import random
import hashlib
import argparse
import math
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, DistributedSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from modules import (
    set_seed, clean_text, stable_hash, compute_classification_loss,
    maybe_parallelize, save_checkpoint, load_checkpoint, find_latest_ckpt, setup_distributed, cleanup_distributed, rank0_print,
    distributed_evaluate, load_model_state_compat,
    extract_ce_logits, logits_to_ordinal_class, hard_cascade_from_outputs,
)
from data_segmentation import split_report_by_sections, DualHeadOutput, SegmentedTextDataset
from segmented_masked_model import make_segmented_encodings_per_segment
from bertori import (
    ordinal_loss,
    dual_head_mixed_loss,
    hierarchical_birads_loss,
    combine_zero_vs_ordinal_logits,
    focal_loss,
)

# Per-segment token budgets (clinical / findings / comparative).
DEFAULT_SEGMENT_MAX_LENGTHS = (40, 160, 40)

NUM_LABELS = 7  # BI-RADS targets are fixed contiguous ints 0..6


class CrossAttentionSegmentedBert(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels=NUM_LABELS,
        hidden_size=768,
        proj_size=384,
        dropout_rate=0.1,
        cross_attn_layers=1,
        nhead=4,
        dual_head=False,
        hierarchical=False,
        hier_zero_label=0,
        no_segment_fusion=False,
    ):
        super().__init__()
        if int(num_labels) != NUM_LABELS:
            raise ValueError(f"num_labels must be {NUM_LABELS}, got {num_labels}")
        if hierarchical and dual_head:
            raise ValueError("hierarchical and dual_head are mutually exclusive")
        if hierarchical and int(hier_zero_label) != 0:
            raise ValueError("hierarchical currently supports hier_zero_label=0 only")

        self.num_labels = NUM_LABELS
        self.dual_head = bool(dual_head)
        self.hierarchical = bool(hierarchical)
        self.hier_zero_label = int(hier_zero_label)
        self.no_segment_fusion = bool(no_segment_fusion)

        # --- BERT backbone (shared across 3 segments) ---
        self.bert = AutoModel.from_pretrained(model_name)
        # Only last_hidden_state is used; drop unused pooler so DDP does not
        # require find_unused_parameters=True on the CE-only path.
        if getattr(self.bert, "pooler", None) is not None:
            self.bert.pooler = None
        pad_id = getattr(self.bert.config, "pad_token_id", 0)
        self.pad_token_id = 0 if pad_id is None else int(pad_id)

        # --- Projection from BERT hidden to smaller dim ---
        self.proj = nn.Linear(hidden_size, proj_size)

        # --- Segment type embedding (0=clinical, 1=findings, 2=comparative) ---
        self.segment_embed = nn.Embedding(3, proj_size)

        # --- Segment gate (softmax over 3 segments) ---
        self.segment_gate = nn.Linear(proj_size, 1)
        nn.init.xavier_uniform_(self.segment_gate.weight)
        nn.init.zeros_(self.segment_gate.bias)

        # --- Learnable global CLS query for cross-attention ---
        self.global_cls = nn.Parameter(
            torch.empty(1, 1, proj_size).normal_(std=1.0 / math.sqrt(proj_size))
        )

        # --- Cross-attention: CLS as query, segment tokens as key/value ---
        cross_layer = nn.TransformerDecoderLayer(
            d_model=proj_size,
            nhead=nhead,
            dim_feedforward=proj_size * 2,
            dropout=dropout_rate,
            activation="relu",
            batch_first=False,        # (seq, batch, dim)
        )
        self.cross_attn = nn.TransformerDecoder(
            cross_layer,
            num_layers=cross_attn_layers,
        )

        # --- Attention pooling over segment tokens (independent of CLS) ---
        self.attn_pool = nn.Linear(proj_size, 1)
        nn.init.xavier_uniform_(self.attn_pool.weight)
        nn.init.zeros_(self.attn_pool.bias)

        # --- Learnable gate to fuse CLS output and attention-pooled tokens ---
        self.cls_pool_gate = nn.Linear(proj_size * 2, proj_size)

        # --- Classifier(s) ---
        self.dropout = nn.Dropout(dropout_rate)
        if self.hierarchical:
            # Binary: is_zero vs rest; ordinal: diagnostic classes 1..K-1 (K-2 thresholds)
            self.classifier = None
            self.binary_classifier = nn.Linear(proj_size, 1)
            self.ordinal_classifier = nn.Linear(proj_size, NUM_LABELS - 2)
        else:
            self.binary_classifier = None
            self.classifier = nn.Linear(proj_size, NUM_LABELS)
            self.ordinal_classifier = (
                nn.Linear(proj_size, NUM_LABELS - 1) if self.dual_head else None
            )

    # --------------------------------------------------------------------- #
    def _project_segment(self, hidden, input_ids, seg_id):
        """Linear proj (+ optional segment embed) → (batch, seq_len, proj_size)."""
        proj = self.proj(hidden)
        if not self.no_segment_fusion:
            seg_ids = torch.full_like(input_ids, seg_id)
            proj = proj + self.segment_embed(seg_ids)
        return proj

    # --------------------------------------------------------------------- #
    def _bert_encode_segments(self, seg_ids, seg_masks):
        """
        One fused BERT forward for three segments.

        Segments may have different sequence lengths (e.g. 40/160/40). When lengths
        differ, pad to the max length, run one batched forward, then slice back.
        """
        lengths = [ids.size(1) for ids in seg_ids]
        batch_size = seg_ids[0].size(0)

        if lengths[0] == lengths[1] == lengths[2]:
            all_hidden = self.bert(
                torch.cat(list(seg_ids), dim=0),
                attention_mask=torch.cat(list(seg_masks), dim=0),
            ).last_hidden_state
            return list(all_hidden.chunk(3, dim=0))

        max_len = max(lengths)
        padded_ids = [
            F.pad(ids, (0, max_len - ids.size(1)), value=self.pad_token_id)
            for ids in seg_ids
        ]
        padded_masks = [
            F.pad(mask, (0, max_len - mask.size(1)), value=0) for mask in seg_masks
        ]
        fused = self.bert(
            torch.cat(padded_ids, dim=0),
            attention_mask=torch.cat(padded_masks, dim=0),
        ).last_hidden_state
        chunks = fused.split(batch_size, dim=0)
        return [chunk[:, :length] for chunk, length in zip(chunks, lengths)]

    # --------------------------------------------------------------------- #
    def forward(
        self,
        clinical_input_ids, clinical_attention_mask,
        findings_input_ids, findings_attention_mask,
        comparative_input_ids, comparative_attention_mask,
    ):
        batch_size = clinical_input_ids.size(0)

        # ===== 1. One batched BERT forward for all 3 segments =====
        cli_h, find_h, comp_h = self._bert_encode_segments(
            (clinical_input_ids, findings_input_ids, comparative_input_ids),
            (clinical_attention_mask, findings_attention_mask, comparative_attention_mask),
        )

        cli_proj = self._project_segment(cli_h, clinical_input_ids, 0)
        find_proj = self._project_segment(find_h, findings_input_ids, 1)
        comp_proj = self._project_segment(comp_h, comparative_input_ids, 2)
        cli_mask, find_mask, comp_mask = (
            clinical_attention_mask,
            findings_attention_mask,
            comparative_attention_mask,
        )

        def _masked_mean(proj, mask):
            # proj: (B, L, D), mask: (B, L) — empty segments (all-zero mask) -> zeros
            m = mask.unsqueeze(-1).to(dtype=proj.dtype)
            denom = m.sum(dim=1).clamp(min=1e-6)
            return (proj * m).sum(dim=1) / denom

        if self.no_segment_fusion:
            # Ablation: mean of available segment CLS tokens (skip fully-masked segs)
            vecs = []
            for proj, mask in (
                (cli_proj, cli_mask),
                (find_proj, find_mask),
                (comp_proj, comp_mask),
            ):
                if mask.sum() > 0:
                    vecs.append(proj[:, 0])
            final_repr = (
                torch.stack(vecs, dim=0).mean(dim=0)
                if vecs
                else find_proj[:, 0]
            )
        else:
            # ===== 2. Segment gating (softmax; empty segs get -inf score) =====
            cli_vec = _masked_mean(cli_proj, cli_mask)
            find_vec = _masked_mean(find_proj, find_mask)
            comp_vec = _masked_mean(comp_proj, comp_mask)

            segment_scores = torch.cat([
                self.segment_gate(cli_vec),
                self.segment_gate(find_vec),
                self.segment_gate(comp_vec),
            ], dim=1)
            seg_valid = torch.stack(
                [
                    cli_mask.sum(dim=1) > 0,
                    find_mask.sum(dim=1) > 0,
                    comp_mask.sum(dim=1) > 0,
                ],
                dim=1,
            )
            # Use dtype min (fp16 max finite ~65504); -1e9 overflows float16 under AMP.
            neg = torch.finfo(segment_scores.dtype).min
            segment_scores = segment_scores.masked_fill(~seg_valid, neg)
            segment_weights = torch.softmax(segment_scores, dim=1)

            w1 = segment_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
            w2 = segment_weights[:, 1].unsqueeze(-1).unsqueeze(-1)
            w3 = segment_weights[:, 2].unsqueeze(-1).unsqueeze(-1)

            cli_proj = cli_proj * w1
            find_proj = find_proj * w2
            comp_proj = comp_proj * w3

            # ===== 3. Concatenate segment tokens & masks =====
            # (batch, total_seq, proj_size)
            kv_tokens = torch.cat([cli_proj, find_proj, comp_proj], dim=1)
            kv_mask = torch.cat([cli_mask, find_mask, comp_mask], dim=1)
            # key_padding_mask: True where padded (mask==0)
            kv_pad_mask = (kv_mask == 0)

            # ===== 4. Cross-attention: CLS query attends to segment tokens =====
            # TransformerDecoder expects (seq, batch, dim) when batch_first=False
            query = self.global_cls.expand(-1, batch_size, -1)  # (1, batch, proj_size)
            memory = kv_tokens.permute(1, 0, 2)                 # (total_seq, batch, proj_size)

            cls_output = self.cross_attn(
                tgt=query,
                memory=memory,
                memory_key_padding_mask=kv_pad_mask,
            )  # (1, batch, proj_size)
            cls_output = cls_output.squeeze(0)  # (batch, proj_size)

            # ===== 5. Attention pooling over segment tokens =====
            # memory: (total_seq, batch, proj_size)
            attn_scores = self.attn_pool(memory)               # (total_seq, batch, 1)
            # mask out padding positions with -inf before softmax
            attn_scores = attn_scores.masked_fill(
                kv_pad_mask.T.unsqueeze(-1), float("-inf")
            )
            attn_weights = torch.softmax(attn_scores, dim=0)   # (total_seq, batch, 1)
            attn_pooled = (memory * attn_weights).sum(dim=0)    # (batch, proj_size)

            # ===== 6. Gated fusion of CLS and attention-pooled =====
            gate = torch.sigmoid(
                self.cls_pool_gate(torch.cat([cls_output, attn_pooled], dim=-1))
            )  # (batch, proj_size)
            final_repr = gate * cls_output + (1.0 - gate) * attn_pooled

        # ===== Classifier =====
        dropped = self.dropout(final_repr)
        if self.hierarchical:
            binary_logits = self.binary_classifier(dropped).squeeze(-1)
            ordinal_logits = self.ordinal_classifier(dropped)
            logits = combine_zero_vs_ordinal_logits(binary_logits, ordinal_logits)
            outputs = DualHeadOutput(
                logits=logits,
                ordinal_logits=ordinal_logits,
                binary_logits=binary_logits,
            )
        elif self.dual_head:
            outputs = DualHeadOutput(
                logits=self.classifier(dropped),
                ordinal_logits=self.ordinal_classifier(dropped),
            )
        else:
            outputs = self.classifier(dropped)

        return outputs


import warnings 
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_PATH = "./train_split.csv"  # Use pre-segmented data
# Full-report CSV used only for GroupKFold grouping (must match bertori.py)
GROUP_REPORT_PATH = "/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/train_new.csv"
TEST_PATH = "./test.csv"
SUB_PATH = "./submission.csv"


def resolve_loss_config(args):
    """
    Mirror bertori.py loss selection.
    Returns (loss_type, use_dual_head, use_ordinal_decode, loss_kwargs, log_msg, use_hierarchical).
    """
    loss_type = args.loss_type
    # Backward compatible: --use_focal_loss alone still selects focal when loss_type is CE.
    if loss_type == "cross_entropy" and getattr(args, "use_focal_loss", False):
        loss_type = "focal"

    use_dual_head = False
    use_hierarchical = False
    use_ordinal_decode = False
    loss_kwargs = {
        "focal_gamma": float(args.focal_gamma),
        "mix_lambda": float(args.mix_lambda),
        "weight_focal": float(args.loss_weight_focal),
        "use_focal_primary": bool(args.loss_weight_focal > 0) or loss_type == "focal",
        "label_smoothing": float(getattr(args, "label_smoothing", 0.0) or 0.0),
        "hier_zero_label": int(getattr(args, "hier_zero_label", 0)),
    }

    if loss_type == "ordinal":
        use_ordinal_decode = True
        msg = "[INFO] Using ordinal loss (shared K-way logits, ordinal decode)"
    elif loss_type == "focal":
        msg = f"[INFO] Using FOCAL LOSS with gamma={args.focal_gamma}"
    elif loss_type == "hierarchical":
        use_hierarchical = True
        mix_lambda = float(args.mix_lambda)
        use_ordinal_decode = False
        zero_label = int(loss_kwargs["hier_zero_label"])
        cascade_tau = float(getattr(args, "cascade_tau", 0.5))
        loss_kwargs["cascade_tau"] = cascade_tau
        loss_kwargs["use_hard_cascade"] = True
        msg = (
            f"[INFO] Using hierarchical BI-RADS loss: "
            f"L = {mix_lambda:.3f}*BCE(is_zero={zero_label}) + "
            f"{1.0 - mix_lambda:.3f}*ordinal(diagnostic) "
            f"(select by hard-cascade tau={cascade_tau:g})"
        )
    elif loss_type == "mixed":
        use_dual_head = True
        mix_lambda = float(args.mix_lambda)
        # λ=0 trains only the ordinal head (K-1 thresholds). CE argmax F1 is
        # then the untrained CE classifier. Decode: pred = Σ_k 1[σ(θ_k) ≥ 0.5].
        use_ordinal_decode = mix_lambda <= 0.0
        primary = "focal" if loss_kwargs["use_focal_primary"] and args.loss_weight_focal > 0 else "CE"
        decode_name = (
            "ordinal-threshold F1"
            if use_ordinal_decode
            else "CE-argmax F1"
        )
        msg = (
            f"[INFO] Using dual-head mixed loss: "
            f"λ={mix_lambda:.3f}*{primary} + {1.0 - mix_lambda:.3f}*ordinal "
            f"(select by {decode_name})"
        )
    else:
        msg = "[INFO] Using standard CROSS ENTROPY loss"

    return loss_type, use_dual_head, use_ordinal_decode, loss_kwargs, msg, use_hierarchical


def compute_segmented_loss(outputs, labels, class_weights, loss_type, loss_kwargs):
    ce_logits = extract_ce_logits(outputs)
    label_smoothing = float(loss_kwargs.get("label_smoothing", 0.0) or 0.0)

    def _ce(logits):
        return F.cross_entropy(
            logits.float(),
            labels,
            weight=class_weights.float() if class_weights is not None else None,
            label_smoothing=label_smoothing if label_smoothing > 0 else 0.0,
        )

    if loss_type == "cross_entropy":
        return _ce(ce_logits) if label_smoothing > 0 else compute_classification_loss(
            ce_logits, labels, class_weights=class_weights, use_focal_loss=False
        )
    if loss_type == "focal":
        return compute_classification_loss(
            ce_logits,
            labels,
            class_weights=class_weights,
            use_focal_loss=True,
            focal_gamma=loss_kwargs["focal_gamma"],
        )
    if loss_type == "ordinal":
        return ordinal_loss(ce_logits, labels, class_weights=class_weights)
    if loss_type == "mixed":
        ordinal_logits = getattr(outputs, "ordinal_logits", None)
        if ordinal_logits is None:
            raise ValueError("mixed loss requires dual-head ordinal_logits on outputs")
        mix_lambda = float(loss_kwargs.get("mix_lambda", 0.9))
        mix_lambda = min(max(mix_lambda, 0.0), 1.0)
        use_focal = bool(loss_kwargs.get("weight_focal", 0.0) > 0) or bool(
            loss_kwargs.get("use_focal_primary", False)
        )
        return dual_head_mixed_loss(
            ce_logits,
            ordinal_logits,
            labels,
            class_weights=class_weights,
            mix_lambda=mix_lambda,
            use_focal=use_focal,
            gamma=loss_kwargs["focal_gamma"],
        )
    if loss_type == "hierarchical":
        binary_logits = getattr(outputs, "binary_logits", None)
        ordinal_logits = getattr(outputs, "ordinal_logits", None)
        if binary_logits is None or ordinal_logits is None:
            raise ValueError("hierarchical loss requires binary_logits and ordinal_logits on outputs")
        return hierarchical_birads_loss(
            binary_logits,
            ordinal_logits,
            labels,
            num_classes=ce_logits.size(1),
            class_weights=class_weights,
            mix_lambda=float(loss_kwargs.get("mix_lambda", 0.5)),
            zero_label=int(loss_kwargs.get("hier_zero_label", 0)),
        )
    raise ValueError(f"Unknown loss_type: {loss_type}")


def _resolve_amp_dtype(amp_dtype):
    if isinstance(amp_dtype, torch.dtype):
        return amp_dtype
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(amp_dtype).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
    return mapping[key]


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def predict_from_outputs(
    outputs,
    use_ordinal_decode=False,
    use_hard_cascade=False,
    cascade_tau=0.5,
    hier_zero_label=0,
):
    if use_hard_cascade:
        return hard_cascade_from_outputs(
            outputs, tau=cascade_tau, zero_label=hier_zero_label
        )
    ce_logits = extract_ce_logits(outputs)
    if use_ordinal_decode:
        ordinal_logits = getattr(outputs, "ordinal_logits", None)
        if ordinal_logits is not None:
            return logits_to_ordinal_class(ordinal_logits, num_classes=ce_logits.size(1))
        return logits_to_ordinal_class(ce_logits)
    return torch.argmax(ce_logits, dim=-1)


def state_dict_has_hierarchical(state_dict):
    return any(
        k == "binary_classifier.weight" or k.endswith(".binary_classifier.weight")
        for k in state_dict.keys()
    )


def state_dict_has_dual_head(state_dict):
    if state_dict_has_hierarchical(state_dict):
        return False
    return any(
        k == "ordinal_classifier.weight" or k.endswith(".ordinal_classifier.weight")
        for k in state_dict.keys()
    )


def load_segmented_checkpoint(
    model_name,
    num_labels,
    ckpt_path,
    device,
    dual_head=None,
    hierarchical=None,
    hier_zero_label=0,
    no_segment_fusion=False,
):
    """Load a fold checkpoint; auto-detect dual_head/hierarchical from weights if None."""
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    state = (
        ckpt_data["model_state_dict"]
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data
        else ckpt_data
    )
    if hierarchical is None:
        hierarchical = state_dict_has_hierarchical(state)
    if dual_head is None:
        dual_head = False if hierarchical else state_dict_has_dual_head(state)
    model = CrossAttentionSegmentedBert(
        model_name,
        num_labels=num_labels,
        dual_head=dual_head,
        hierarchical=hierarchical,
        hier_zero_label=hier_zero_label,
        no_segment_fusion=no_segment_fusion,
    )
    load_model_state_compat(
        model, state, strict=False, log_prefix=f"{os.path.basename(ckpt_path)}: "
    )
    return model.to(device), ckpt_data if isinstance(ckpt_data, dict) else {"model_state_dict": state}


def segment_max_lengths(args):
    """
    Resolve (clinical, findings, comparative) token budgets.

    --max_length, if set, forces a uniform length on all three segments (legacy).
    Otherwise use --clinical/findings/comparative_max_length (default 40/160/40).
    """
    if getattr(args, "max_length", None) is not None:
        m = int(args.max_length)
        return (m, m, m)
    return (
        int(args.clinical_max_length),
        int(args.findings_max_length),
        int(args.comparative_max_length),
    )


def encode_three_segments(tokenizer, clinical, findings, comparative, max_lengths):
    """Tokenize segments with per-segment max_length."""
    if isinstance(max_lengths, int):
        max_lengths = (int(max_lengths), int(max_lengths), int(max_lengths))
    return make_segmented_encodings_per_segment(
        tokenizer,
        clinical,
        findings,
        comparative,
        max_lengths=max_lengths,
    )


def assign_group_like_bertori(train: pd.DataFrame, group_report_path: str) -> pd.DataFrame:
    """
    Match bertori.py fold grouping:
      text = clean_text(report); group = stable_hash(text)
    train_split.csv has section columns only, so report is joined from group_report_path by ID.
    """
    train = train.copy()
    if "report" in train.columns:
        train["text"] = train["report"].apply(clean_text)
    else:
        report_df = pd.read_csv(group_report_path)
        assert {"ID", "report"}.issubset(report_df.columns), report_df.columns
        report_df = report_df[["ID", "report"]].drop_duplicates(subset=["ID"], keep="first")
        train = train.merge(report_df, on="ID", how="left", validate="many_to_one")
        if train["report"].isna().any():
            missing = int(train["report"].isna().sum())
            raise ValueError(
                f"{missing} train rows missing report after merge with {group_report_path}"
            )
        train["text"] = train["report"].apply(clean_text)
    train["group"] = train["text"].apply(stable_hash)
    return train


def train_fold(
    model,
    train_loader,
    device,
    lr,
    epochs,
    warmup_ratio,
    class_weights,
    fold=1,
    ckpt_dir=None,
    resume=True,
    use_amp=True,
    grad_accum_steps=1,
    val_interval=1,
    is_distributed=False,
    rank=0,
    world_size=1,
    val_dataset=None,
    val_batch_size=32,
    val_num_workers=0,
    use_focal_loss=False,
    focal_gamma=2.0,
    early_stop_patience=3,
    start_epoch_override=1,
    best_f1_override=-1.0,
    best_epoch_override=0,
    loss_type="cross_entropy",
    loss_kwargs=None,
    use_ordinal_decode=False,
    use_hard_cascade=False,
    cascade_tau=0.5,
    hier_zero_label=0,
    amp_dtype=torch.float16,
):
    if loss_kwargs is None:
        loss_kwargs = {"focal_gamma": focal_gamma}
    # Keep legacy flag in sync if callers still pass use_focal_loss only.
    if use_focal_loss and loss_type == "cross_entropy":
        loss_type = "focal"
        loss_kwargs = {**loss_kwargs, "focal_gamma": focal_gamma}

    use_hard_cascade = bool(use_hard_cascade) or bool(loss_kwargs.get("use_hard_cascade", False))
    cascade_tau = float(loss_kwargs.get("cascade_tau", cascade_tau))
    hier_zero_label = int(loss_kwargs.get("hier_zero_label", hier_zero_label))

    amp_dtype = _resolve_amp_dtype(amp_dtype)
    use_autocast = bool(use_amp) and amp_dtype != torch.float32
    use_grad_scaler = use_autocast and amp_dtype == torch.float16

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler("cuda", enabled=use_grad_scaler)
    best_f1 = best_f1_override
    # Preserve prior best epoch across resume; do NOT reset to 1 (cleanup would
    # otherwise delete the previous best ckpt when the extended run does not improve).
    if best_epoch_override and best_epoch_override > 0:
        best_epoch = int(best_epoch_override)
    elif start_epoch_override > 1:
        best_epoch = int(start_epoch_override) - 1
    else:
        best_epoch = 1
    best_state = None
    start_epoch = start_epoch_override
    early_stop_counter = 0

    # Keep a handle on the prior best weights so end-of-run restore still works
    # when the resumed epochs never beat best_f1.
    if ckpt_dir and best_epoch > 0 and start_epoch > 1:
        prev_best_path = os.path.join(ckpt_dir, f"fold_{fold}_epoch_{best_epoch}.pt")
        if os.path.exists(prev_best_path):
            prev_ckpt = torch.load(prev_best_path, map_location="cpu")
            if isinstance(prev_ckpt, dict) and "model_state_dict" in prev_ckpt:
                best_state = copy.deepcopy(prev_ckpt["model_state_dict"])
                if best_f1 < 0 and "best_f1" in prev_ckpt:
                    best_f1 = float(prev_ckpt["best_f1"])

    if start_epoch > 1:
        rank0_print(
            rank,
            f"    Starting from epoch {start_epoch} (prior best epoch {best_epoch}), "
            f"best_f1={best_f1:.5f}",
        )

    for epoch in range(start_epoch, epochs + 1):
        # Set epoch for DistributedSampler
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        accumulation_steps = 0

        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)

            batch_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k != "labels"
            }

            with autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
                outputs = model(
                    clinical_input_ids=batch_device["clinical_input_ids"],
                    clinical_attention_mask=batch_device["clinical_attention_mask"],
                    findings_input_ids=batch_device["findings_input_ids"],
                    findings_attention_mask=batch_device["findings_attention_mask"],
                    comparative_input_ids=batch_device["comparative_input_ids"],
                    comparative_attention_mask=batch_device["comparative_attention_mask"],
                )
                loss = compute_segmented_loss(
                    outputs,
                    labels,
                    class_weights=class_weights,
                    loss_type=loss_type,
                    loss_kwargs=loss_kwargs,
                )

            loss = loss / grad_accum_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_steps += 1
            running_loss += float(loss.detach()) * grad_accum_steps

            if accumulation_steps % grad_accum_steps == 0:
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        avg_loss = running_loss / max(1, len(train_loader))
        if epoch % val_interval == 0:
            val_f1, pred_va = distributed_evaluate(
                model=model,
                dataset=val_dataset,
                device=device,
                batch_size=val_batch_size,
                num_workers=val_num_workers,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                use_ordinal=use_ordinal_decode,
                use_hard_cascade=use_hard_cascade,
                cascade_tau=cascade_tau,
                hier_zero_label=hier_zero_label,
            )

            if use_hard_cascade:
                decode_tag = f"hard-cascade(tau={cascade_tau:g})"
            elif use_ordinal_decode:
                decode_tag = "ordinal-threshold"
            else:
                decode_tag = "CE-argmax"
            rank0_print(
                rank,
                f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} | "
                f"val macro F1 ({decode_tag}): {val_f1:.5f} ",
            )

            model.train()

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                model_state = unwrap_model(model).state_dict()
                best_state = copy.deepcopy(model_state)
                if ckpt_dir and rank == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}_epoch_{epoch}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_f1': best_f1,
                    }, ckpt_path)
        else:
            rank0_print(rank, f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} (no validation)")

        if is_distributed:
            dist.barrier()

    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)
    
    # Clean up old checkpoints: keep only the true best for this fold.
    # On resume with no improvement, best_epoch still points at the prior best file
    # so we must not fall back to epoch_1 and wipe it.
    if ckpt_dir and rank == 0:
        pattern = os.path.join(ckpt_dir, f"fold_{fold}_epoch_*.pt")
        all_ckpts = glob.glob(pattern)

        def _read_ckpt_f1(path):
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict) and "best_f1" in data:
                return float(data["best_f1"])
            return float("-inf")

        if all_ckpts and best_epoch > 0:
            best_ckpt_name = f"fold_{fold}_epoch_{best_epoch}.pt"
            best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)

            if not os.path.exists(best_ckpt_path):
                # Filename mismatch: keep the on-disk ckpt with the highest stored F1.
                best_ckpt_path = max(all_ckpts, key=_read_ckpt_f1)
                best_ckpt_name = os.path.basename(best_ckpt_path)
                disk_f1 = _read_ckpt_f1(best_ckpt_path)
                if disk_f1 > best_f1:
                    best_f1 = disk_f1

            keep_abs = os.path.abspath(best_ckpt_path)
            for ckpt_path in all_ckpts:
                if os.path.abspath(ckpt_path) == keep_abs:
                    continue
                # Never delete a file whose recorded score is >= the kept best
                # (protects prior-run best if best_epoch was wrong).
                if _read_ckpt_f1(ckpt_path) >= best_f1 - 1e-12:
                    rank0_print(
                        rank,
                        f"    [WARN] Keeping extra ckpt (score not worse than best): "
                        f"{os.path.basename(ckpt_path)}",
                    )
                    continue
                os.remove(ckpt_path)

            rank0_print(rank, f"    Kept best checkpoint: {best_ckpt_name} (F1={best_f1:.5f})")

    # Rank-0 checkpoint deletion can take minutes; wait before the next collective.
    if is_distributed:
        dist.barrier()

    if ckpt_dir and rank == 0:
        done_path = os.path.join(ckpt_dir, f"fold_{fold}.done")
        with open(done_path, "w") as f:
            f.write(f"best_epoch={best_epoch}\nbest_f1={best_f1:.6f}\n")
    if is_distributed:
        dist.barrier()
    
    rank0_print(rank, f"    Training completed for fold {fold}: best_f1={best_f1:.5f} @ epoch {best_epoch}")
    return model, best_f1, best_epoch


def train_full_model(
    model,
    train_loader,
    device,
    lr,
    epochs,
    warmup_ratio,
    class_weights,
    use_focal_loss=False,
    focal_gamma=2.0,
    loss_type="cross_entropy",
    loss_kwargs=None,
):
    if loss_kwargs is None:
        loss_kwargs = {"focal_gamma": focal_gamma}
    if use_focal_loss and loss_type == "cross_entropy":
        loss_type = "focal"
        loss_kwargs = {**loss_kwargs, "focal_gamma": focal_gamma}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Loss log setup ---
    raw_model = unwrap_model(model)
    loss_log_path = getattr(raw_model, "loss_log_path", None)
    if loss_log_path is None and hasattr(raw_model, "ckpt_dir"):
        loss_log_path = os.path.join(raw_model.ckpt_dir, "loss_log.txt")
    if loss_log_path is None:
        loss_log_path = "loss_log.txt"
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w") as f:
            f.write("full\tepoch\tloss\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items() if k != "labels"}

            outputs = model(
                clinical_input_ids=batch_device['clinical_input_ids'],
                clinical_attention_mask=batch_device['clinical_attention_mask'],
                findings_input_ids=batch_device['findings_input_ids'],
                findings_attention_mask=batch_device['findings_attention_mask'],
                comparative_input_ids=batch_device['comparative_input_ids'],
                comparative_attention_mask=batch_device['comparative_attention_mask']
            )
            loss = compute_segmented_loss(
                outputs,
                labels,
                class_weights=class_weights,
                loss_type=loss_type,
                loss_kwargs=loss_kwargs,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Final train epoch {epoch}/{epochs} | loss: {avg_loss:.4f}")

        # --- Write loss log ---
        if loss_log_path:
            with open(loss_log_path, 'a') as f:
                f.write(f"full\t{epoch}\t{avg_loss:.6f}\n")

    return model


def main(args):
    # ============ Initialize Distributed Training ============
    rank, local_rank, world_size, is_distributed = setup_distributed()
    try:
        set_seed(RANDOM_STATE + rank)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank0_print(rank, "Device:", device, "| world_size:", world_size, "| local_rank:", local_rank)
        if torch.cuda.is_available():
            rank0_print(rank, "GPU count:", torch.cuda.device_count())
            torch.backends.cudnn.benchmark = True
        
        loss_type, use_dual_head, use_ordinal_decode, loss_kwargs, loss_msg, use_hierarchical = resolve_loss_config(args)
        rank0_print(rank, loss_msg)
        train = pd.read_csv(args.train_path)
        test = pd.read_csv(args.test_path)

        assert {"ID", "target", "clinical_indication", "findings", "comparative_analysis"}.issubset(train.columns), train.columns
        assert {"ID", "report"}.issubset(test.columns), test.columns

        # Same GroupKFold grouping as bertori.py (hash of cleaned full report text)
        train = assign_group_like_bertori(train, args.group_report_path)

        y = train["target"].astype(int).values
        id2label = {i: i for i in range(NUM_LABELS)}
        num_labels = NUM_LABELS

        # Use pre-segmented columns directly
        X_clinical = train["clinical_indication"].values
        X_findings = train["findings"].values
        X_comparative = train["comparative_analysis"].values
        groups = train["group"].values

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        ckpt_exists = []
        for fold in range(1, args.n_splits + 1):
            pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
            ckpts = glob.glob(pattern)
            if ckpts:
                ckpt_exists.append(fold)
        
        folds_to_train = set()
        if args.train_folds == "none":
            rank0_print(rank, f"[INFO] Found checkpoints for {len(ckpt_exists)}/{args.n_splits} folds. Running inference only (--train_folds=none)...")
            folds_to_train = set()
        elif args.train_folds == "all":
            folds_to_train = set(range(1, args.n_splits + 1))
            rank0_print(rank, f"[INFO] Training all folds (--train_folds=all)...")
        else:
            folds_to_train = set(int(f.strip()) for f in args.train_folds.split(",") if f.strip())
            valid_folds = set(range(1, args.n_splits + 1))
            if not folds_to_train or not folds_to_train.issubset(valid_folds):
                raise ValueError(
                    f"Invalid --train_folds={args.train_folds!r}; "
                    f"expected comma-separated ints in {sorted(valid_folds)}"
                )
            rank0_print(rank, f"[INFO] Training folds: {sorted(folds_to_train)}")

        # If no training needed, proceed directly to inference/submission
        if not folds_to_train:
            if len(ckpt_exists) < args.n_splits:
                rank0_print(rank, f"[WARN] Only {len(ckpt_exists)}/{args.n_splits} folds have checkpoints. Using available checkpoints for submission...")
            
            rank0_print(rank, f"[INFO] Found checkpoints for {len(ckpt_exists)}/{args.n_splits} folds. Generating submission from existing checkpoints...")
            
            if is_distributed:
                dist.barrier()
            
            # Only rank 0 does inference to avoid redundant GPU memory usage
            if rank == 0:
                rank0_print(rank, "\n[INFO] Computing OOF validation metrics from existing checkpoints...")
                oof_pred_inference = np.empty(len(train), dtype=int)
                oof_scores_inference = []
                
                # Initialize fold validation summary file
                fold_validation_summary_path = os.path.join(args.ckpt_dir, "fold_validation_summary.txt")
                os.makedirs(args.ckpt_dir, exist_ok=True)
                with open(fold_validation_summary_path, "w") as f:
                    f.write("=" * 80 + "\n")
                    f.write("FOLD VALIDATION RESULTS SUMMARY\n")
                    f.write("=" * 80 + "\n\n")
                
                # gkf_check = GroupKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
                gkf_check = GroupKFold(n_splits=args.n_splits)
                
                for fold_check, (tr_idx_check, va_idx_check) in enumerate(gkf_check.split(X_clinical, y, groups=groups), 1):
                    pattern = os.path.join(args.ckpt_dir, f"fold_{fold_check}_epoch_*.pt")
                    ckpts = glob.glob(pattern)
                    if not ckpts:
                        rank0_print(rank, f"  [SKIP] Fold {fold_check}: No checkpoint found")
                        continue
                    
                    latest_ckpt = max(ckpts, key=os.path.getctime)
                    
                    va_clinical = X_clinical[va_idx_check].tolist()
                    va_findings = X_findings[va_idx_check].tolist()
                    va_comparative = X_comparative[va_idx_check].tolist()
                    y_va = torch.tensor(y[va_idx_check], dtype=torch.long)
                    
                    va_enc = encode_three_segments(
                        tokenizer, va_clinical, va_findings, va_comparative,
                        segment_max_lengths(args),
                    )
                    va_ds = SegmentedTextDataset(va_enc, y_va)
                    
                    va_model, _ = load_segmented_checkpoint(
                        args.model_name,
                        num_labels,
                        latest_ckpt,
                        device,
                        dual_head=use_dual_head,
                        hierarchical=use_hierarchical,
                        hier_zero_label=int(getattr(args, "hier_zero_label", 0)),
                        no_segment_fusion=args.no_segment_fusion,
                    )
                    
                    val_f1, pred_va = distributed_evaluate(
                        model=va_model,
                        dataset=va_ds,
                        device=device,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        is_distributed=False,
                        rank=0,
                        world_size=1,
                        use_ordinal=use_ordinal_decode,
                        use_hard_cascade=bool(loss_kwargs.get("use_hard_cascade", False)),
                        cascade_tau=float(loss_kwargs.get("cascade_tau", 0.5)),
                        hier_zero_label=int(loss_kwargs.get("hier_zero_label", 0)),
                    )
                    
                    oof_pred_inference[va_idx_check] = pred_va
                    oof_scores_inference.append(val_f1)

                    y_va_np = y[va_idx_check]
                    cm = confusion_matrix(y_va_np, pred_va, labels=list(range(num_labels)))
                    report_str = classification_report(
                        y_va_np, pred_va,
                        target_names=[str(id2label[i]) for i in range(num_labels)],
                    )
                    rank0_print(rank, f"  Fold {fold_check}: F1 = {val_f1:.5f}")
                    with open(fold_validation_summary_path, "a") as f:
                        f.write(f"\nFold {fold_check}: F1 = {val_f1:.5f}\n")
                        f.write(f"Confusion Matrix:\n{cm}\n\n")
                        f.write(f"Classification Report:\n{report_str}\n")
                        f.write("=" * 80 + "\n\n")
                
                # Output OOF metrics if all folds have predictions
                if len(oof_scores_inference) == args.n_splits:
                    cv_mean = float(np.mean(oof_scores_inference))
                    cv_std = float(np.std(oof_scores_inference))
                    oof_score = f1_score(y, oof_pred_inference, average="macro")
                    
                    rank0_print(rank, f"\nCV mean: {cv_mean:.5f} | CV std: {cv_std:.5f}")
                    rank0_print(rank, f"OOF macro F1: {oof_score:.5f}")

                    cm_overall = confusion_matrix(y, oof_pred_inference, labels=list(range(num_labels)))
                    report_overall = classification_report(
                        y, oof_pred_inference,
                        target_names=[str(id2label[i]) for i in range(num_labels)],
                    )
                    with open(fold_validation_summary_path, "a") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write("OVERALL OUT-OF-FOLD METRICS\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"CV mean: {cv_mean:.5f} | CV std: {cv_std:.5f}\n")
                        f.write(f"OOF macro F1: {oof_score:.5f}\n")
                        f.write(f"\nOverall Confusion Matrix:\n{cm_overall}\n\n")
                        f.write(f"Overall Classification Report:\n{report_overall}\n")
                
                # Now proceed with test submission
                test_texts = test["report"].values
                
                # Segment test reports (test.csv doesn't have pre-segmentation)
                test_clinical = []
                test_findings = []
                test_comparative = []
                
                for report in test_texts:
                    sections = split_report_by_sections(report)
                    test_clinical.append(sections['clinical_indication'])
                    test_findings.append(sections['findings'])
                    test_comparative.append(sections['comparative_analysis'])
                
                test_enc = encode_three_segments(
                    tokenizer, test_clinical, test_findings, test_comparative,
                    segment_max_lengths(args),
                )
                test_ds = SegmentedTextDataset(test_enc, labels=None)
                test_loader = DataLoader(
                    test_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                
                all_fold_preds = []
                used_folds = []
                for fold in range(1, args.n_splits + 1):
                    pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
                    ckpts = glob.glob(pattern)
                    if not ckpts:
                        rank0_print(rank, f"  [SKIP] Fold {fold}: No checkpoint found")
                        continue
                    
                    latest_ckpt = max(ckpts, key=os.path.getctime)
                    
                    test_model, _ = load_segmented_checkpoint(
                        args.model_name,
                        num_labels,
                        latest_ckpt,
                        device,
                        dual_head=use_dual_head,
                        hierarchical=use_hierarchical,
                        hier_zero_label=int(getattr(args, "hier_zero_label", 0)),
                        no_segment_fusion=args.no_segment_fusion,
                    )
                    test_model.eval()
                    
                    fold_preds = []
                    with torch.no_grad():
                        for batch in test_loader:
                            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                           for k, v in batch.items()}
                            
                            outputs = test_model(
                                clinical_input_ids=batch_device['clinical_input_ids'],
                                clinical_attention_mask=batch_device['clinical_attention_mask'],
                                findings_input_ids=batch_device['findings_input_ids'],
                                findings_attention_mask=batch_device['findings_attention_mask'],
                                comparative_input_ids=batch_device['comparative_input_ids'],
                                comparative_attention_mask=batch_device['comparative_attention_mask']
                            )
                            pred = predict_from_outputs(
                                outputs,
                                use_ordinal_decode=use_ordinal_decode,
                                use_hard_cascade=bool(loss_kwargs.get("use_hard_cascade", False)),
                                cascade_tau=float(loss_kwargs.get("cascade_tau", 0.5)),
                                hier_zero_label=int(loss_kwargs.get("hier_zero_label", 0)),
                            )
                            fold_preds.extend(pred.cpu().numpy().tolist())
                    
                    all_fold_preds.append(fold_preds)
                    used_folds.append(fold)
                    rank0_print(rank, f"  Fold {fold}: predictions generated ({len(fold_preds)} samples)")
                
                # Majority vote across folds
                if not all_fold_preds:
                    rank0_print(rank, "[ERROR] No fold predictions available!")
                    return
                
                pred_matrix = np.stack(all_fold_preds, axis=0)  # [n_folds, n_samples]
                majority_ids = []
                for i in range(pred_matrix.shape[1]):
                    voted_id = np.argmax(np.bincount(pred_matrix[:, i]))
                    majority_ids.append(voted_id)
                
                pred_test = [id2label[i] for i in majority_ids]
                
                submission = pd.DataFrame({
                    "ID": test["ID"],
                    "target": pred_test,
                })

                save_dir = os.path.dirname(args.sub_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                elif save_dir and not os.path.isdir(save_dir):
                    raise ValueError(f"{save_dir} exists but is not a directory")
                
                submission.to_csv(args.sub_path, index=False)
                rank0_print(rank, f"\n[INFO] Saved submission using {len(used_folds)} folds: {sorted(used_folds)}")
                rank0_print(rank, f"[INFO] Path: {args.sub_path}")
                rank0_print(rank, submission.head())
            
            if is_distributed:
                dist.barrier()
            return

        # ============ Proceed with training ============
        rank0_print(rank, f"[INFO] Starting training for {len(folds_to_train)} folds...")

        counts = np.bincount(y, minlength=num_labels)
        class_weights_np = len(y) / (num_labels * np.maximum(counts, 1))
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

        # gkf = GroupKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
        gkf = GroupKFold(n_splits=args.n_splits)

        oof_pred = np.empty(len(train), dtype=int)
        fold_scores = []
        best_epochs = []

        # Initialize fold validation summary file (rank 0 only — avoid multi-rank truncate races)
        fold_validation_summary_path = os.path.join(args.ckpt_dir, "fold_validation_summary.txt")
        os.makedirs(args.ckpt_dir, exist_ok=True)
        if rank == 0:
            mode = "a" if args.resume and os.path.exists(fold_validation_summary_path) else "w"
            with open(fold_validation_summary_path, mode) as f:
                if mode == "w":
                    f.write("=" * 80 + "\n")
                    f.write("FOLD VALIDATION RESULTS SUMMARY\n")
                    f.write("=" * 80 + "\n\n")
                else:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("RESUMED RUN\n")
                    f.write("=" * 80 + "\n\n")
        if is_distributed:
            dist.barrier()

        # DataLoader workers under DDP are a common hang source (CPU oversubscription).
        loader_workers = 0 if is_distributed else args.num_workers

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_clinical, y, groups=groups), 1):
            # Skip folds not in training list
            if fold not in folds_to_train:
                rank0_print(rank, f"\nFold {fold}/{args.n_splits} - SKIPPED (not in --train_folds)")
                continue

            if is_distributed:
                dist.barrier()
            
            rank0_print(rank, f"\nFold {fold}/{args.n_splits}")
            
            # Check if checkpoint exists for this fold
            pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
            existing_ckpts = glob.glob(pattern)
            if existing_ckpts:
                rank0_print(rank, f"  [INFO] Found {len(existing_ckpts)} existing checkpoint(s) for fold {fold}. Will resume training from checkpoint.")
            else:
                rank0_print(rank, f"  [INFO] No checkpoint found for fold {fold}. Training from scratch.")

            va_clinical = X_clinical[va_idx].tolist()
            va_findings = X_findings[va_idx].tolist()
            va_comparative = X_comparative[va_idx].tolist()
            y_va = torch.tensor(y[va_idx], dtype=torch.long)
            va_enc = encode_three_segments(
                tokenizer, va_clinical, va_findings, va_comparative,
                segment_max_lengths(args),
            )
            va_ds = SegmentedTextDataset(va_enc, y_va)

            model = None
            tr_loader = None
            best_f1 = -1.0
            best_epoch = 1

            tr_clinical = X_clinical[tr_idx].tolist()
            tr_findings = X_findings[tr_idx].tolist()
            tr_comparative = X_comparative[tr_idx].tolist()
            y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
            tr_enc = encode_three_segments(
                tokenizer, tr_clinical, tr_findings, tr_comparative,
                segment_max_lengths(args),
            )
            tr_ds = SegmentedTextDataset(tr_enc, y_tr)

            # Use DistributedSampler for training
            tr_sampler = None
            if is_distributed:
                tr_sampler = DistributedSampler(
                    tr_ds,
                    num_replicas=world_size,
                    rank=rank,
                    drop_last=True,
                )
            
            tr_loader = DataLoader(
                tr_ds,
                batch_size=args.batch_size,
                shuffle=(tr_sampler is None),
                sampler=tr_sampler,
                num_workers=loader_workers,
                pin_memory=torch.cuda.is_available(),
            )

            # Sync after uneven tokenize/build work before DDP collectives.
            if is_distributed:
                dist.barrier()

            model = CrossAttentionSegmentedBert(
                args.model_name,
                num_labels=num_labels,
                dual_head=use_dual_head,
                hierarchical=use_hierarchical,
                hier_zero_label=int(getattr(args, "hier_zero_label", 0)),
                no_segment_fusion=args.no_segment_fusion,
            )

            # Try to load checkpoint BEFORE wrapping with DDP
            start_epoch_ckpt = 1
            best_f1_ckpt = -1.0
            best_epoch_ckpt = 0
            if args.resume and args.ckpt_dir and existing_ckpts:
                latest_ckpt = max(existing_ckpts, key=os.path.getctime)
                rank0_print(rank, f"  [INFO] Loading checkpoint BEFORE DDP wrap: {latest_ckpt}")
                ckpt_data = torch.load(latest_ckpt, map_location="cpu")
                model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
                best_epoch_ckpt = int(ckpt_data.get("epoch", 1))
                start_epoch_ckpt = best_epoch_ckpt + 1
                best_f1_ckpt = ckpt_data.get("best_f1", -1.0)
                rank0_print(
                    rank,
                    f"  ✓ Checkpoint loaded: epoch {best_epoch_ckpt}, best_f1={best_f1_ckpt:.5f}",
                )

            # NOW wrap with DDP
            if is_distributed:
                model = model.to(device)
                # Dual-head + fusion uses every remaining parameter. Unused-param
                # search is only needed for the no-fusion ablation.
                model = DDP(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=bool(args.no_segment_fusion),
                )
            else:
                model = maybe_parallelize(model, device, use_data_parallel=not args.no_data_parallel)

            model, best_f1, best_epoch = train_fold(
                model=model,
                train_loader=tr_loader,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                warmup_ratio=args.warmup_ratio,
                class_weights=class_weights,
                fold=fold,
                ckpt_dir=args.ckpt_dir,
                resume=False,  # Already loaded before DDP wrap
                use_amp=args.use_amp,
                grad_accum_steps=args.grad_accum_steps,
                val_interval=args.val_interval,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                val_dataset=va_ds,
                val_batch_size=args.batch_size,
                val_num_workers=loader_workers,
                use_focal_loss=args.use_focal_loss,
                focal_gamma=args.focal_gamma,
                start_epoch_override=start_epoch_ckpt,
                best_f1_override=best_f1_ckpt,
                best_epoch_override=best_epoch_ckpt,
                early_stop_patience=3,
                loss_type=loss_type,
                loss_kwargs=loss_kwargs,
                use_ordinal_decode=use_ordinal_decode,
                use_hard_cascade=bool(loss_kwargs.get("use_hard_cascade", False)),
                cascade_tau=float(loss_kwargs.get("cascade_tau", 0.5)),
                hier_zero_label=int(loss_kwargs.get("hier_zero_label", 0)),
                amp_dtype=getattr(args, "amp_dtype", "float16"),
            )

            val_f1, pred_va = distributed_evaluate(
                model=model,
                dataset=va_ds,
                device=device,
                batch_size=args.batch_size,
                num_workers=loader_workers,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                use_ordinal=use_ordinal_decode,
                use_hard_cascade=bool(loss_kwargs.get("use_hard_cascade", False)),
                cascade_tau=float(loss_kwargs.get("cascade_tau", 0.5)),
                hier_zero_label=int(loss_kwargs.get("hier_zero_label", 0)),
            )

            if rank == 0:
                oof_pred[va_idx] = pred_va
                cm = confusion_matrix(y[va_idx], pred_va)
                report_str = classification_report(
                    y[va_idx],
                    pred_va,
                    target_names=[str(id2label[i]) for i in range(num_labels)],
                )
                with open(fold_validation_summary_path, "a") as f:
                    f.write(f"\nFold {fold} Confusion Matrix:\n{cm}\n\n")
                    f.write(f"Fold {fold} Classification Report:\n{report_str}\n")
                    f.write("=" * 80 + "\n\n")

            # Synchronize across ranks after evaluation
            if is_distributed:
                dist.barrier()
            
            fold_scores.append(val_f1)
            best_epochs.append(best_epoch)

            rank0_print(
                rank,
                f"Fold {fold} best macro F1: {best_f1:.5f} @ epoch {best_epoch} "
                f"| final eval macro F1: {val_f1:.5f}"
            )

            # Tear down DDP/loaders before the next fold to avoid worker/NCCL bleed-over.
            del model, tr_loader, va_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if is_distributed:
                dist.barrier()

        cv_mean = float(np.mean(fold_scores)) if fold_scores else 0.0
        cv_std = float(np.std(fold_scores)) if fold_scores else 0.0
        oof_score = f1_score(y, oof_pred, average="macro") if len(fold_scores) == args.n_splits else -1.0

        rank0_print(rank, "\nCV mean:", f"{cv_mean:.5f}", "| CV std:", f"{cv_std:.5f}")
        if oof_score >= 0:
            rank0_print(rank, "OOF macro F1:", f"{oof_score:.5f}")

        if is_distributed:
            dist.barrier()

        rank0_print(rank, "\n[INFO] Ensemble test predictions using available checkpoints...")
        test_texts = test["report"].values
        
        # Segment test reports (test.csv doesn't have pre-segmentation)
        test_clinical = []
        test_findings = []
        test_comparative = []
        
        for report in test_texts:
            sections = split_report_by_sections(report)
            test_clinical.append(sections['clinical_indication'])
            test_findings.append(sections['findings'])
            test_comparative.append(sections['comparative_analysis'])
        
        test_enc = encode_three_segments(
            tokenizer, test_clinical, test_findings, test_comparative,
            segment_max_lengths(args),
        )
        test_ds = SegmentedTextDataset(test_enc, labels=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=torch.cuda.is_available(),
        )

        all_fold_preds = []
        used_folds = []
        
        # Only rank 0 does inference
        if rank == 0:
            for fold in range(1, args.n_splits + 1):
                pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
                ckpts = glob.glob(pattern)
                if not ckpts:
                    rank0_print(rank, f"  [SKIP] Fold {fold}: No checkpoint found")
                    continue
                
                latest_ckpt = max(ckpts, key=os.path.getctime)
                
                test_model, _ = load_segmented_checkpoint(
                    args.model_name,
                    num_labels,
                    latest_ckpt,
                    device,
                    dual_head=use_dual_head,
                    hierarchical=use_hierarchical,
                    hier_zero_label=int(getattr(args, "hier_zero_label", 0)),
                    no_segment_fusion=args.no_segment_fusion,
                    )
                test_model.eval()
                
                fold_preds = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                       for k, v in batch.items()}
                        
                        outputs = test_model(
                            clinical_input_ids=batch_device['clinical_input_ids'],
                            clinical_attention_mask=batch_device['clinical_attention_mask'],
                            findings_input_ids=batch_device['findings_input_ids'],
                            findings_attention_mask=batch_device['findings_attention_mask'],
                            comparative_input_ids=batch_device['comparative_input_ids'],
                            comparative_attention_mask=batch_device['comparative_attention_mask']
                        )
                        pred = predict_from_outputs(
                            outputs,
                            use_ordinal_decode=use_ordinal_decode,
                            use_hard_cascade=bool(loss_kwargs.get("use_hard_cascade", False)),
                            cascade_tau=float(loss_kwargs.get("cascade_tau", 0.5)),
                            hier_zero_label=int(loss_kwargs.get("hier_zero_label", 0)),
                        )
                        fold_preds.extend(pred.cpu().numpy().tolist())
                
                all_fold_preds.append(fold_preds)
                used_folds.append(fold)
                rank0_print(rank, f"  Fold {fold}: predictions generated ({len(fold_preds)} samples)")

            # Majority vote across folds
            if not all_fold_preds:
                rank0_print(rank, "[ERROR] No fold predictions available!")
                return
            
            pred_matrix = np.stack(all_fold_preds, axis=0)
            majority_ids = []
            for i in range(pred_matrix.shape[1]):
                voted_id = np.argmax(np.bincount(pred_matrix[:, i]))
                majority_ids.append(voted_id)
            
            pred_test = [id2label[i] for i in majority_ids]
            
            submission = pd.DataFrame({
                "ID": test["ID"],
                "target": pred_test,
            })

            save_dir = os.path.dirname(args.sub_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            elif save_dir and not os.path.isdir(save_dir):
                raise ValueError(f"{save_dir} exists but is not a directory")
            
            submission.to_csv(args.sub_path, index=False)
            rank0_print(rank, f"\n[INFO] Saved submission using {len(used_folds)} folds: {sorted(used_folds)}")
            rank0_print(rank, f"[INFO] Path: {args.sub_path}")
            rank0_print(rank, submission.head())
    
    finally:
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERT baseline with segmented report processing (3-section)"
    )
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument(
        "--group_report_path",
        type=str,
        default=GROUP_REPORT_PATH,
        help="Full-report CSV (ID, report) for GroupKFold grouping; must match bertori.py train_path",
    )
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--sub_path", type=str, default=SUB_PATH)

    parser.add_argument("--model_name", type=str, default="pucpr/biobertpt-all")
    parser.add_argument("--clinical_max_length", type=int, default=DEFAULT_SEGMENT_MAX_LENGTHS[0], help="Max tokens for clinical_indication (default 40)")
    parser.add_argument("--findings_max_length", type=int, default=DEFAULT_SEGMENT_MAX_LENGTHS[1], help="Max tokens for findings (default 160)")
    parser.add_argument("--comparative_max_length", type=int, default=DEFAULT_SEGMENT_MAX_LENGTHS[2], help="Max tokens for comparative_analysis (default 40)")
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Legacy uniform length for all three segments; if set, overrides "
             "--clinical/findings/comparative_max_length",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_data_parallel", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--train_folds",
        type=str,
        default="all",
        help="Which folds to train: 'all', 'none', or comma-separated fold numbers"
    )
    
    # Speedup options
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision (see --amp_dtype)")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Autocast dtype. bfloat16 recommended on A100 (no GradScaler).",
    )
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "ordinal", "focal", "mixed", "hierarchical"],
        help="Loss type. 'mixed' always uses dual-head (CE + ordinal).",
    )
    parser.add_argument(
        "--mix_lambda",
        type=float,
        default=0.9,
        help="Mix weight λ. mixed (dual-head): L=λ*L_primary+(1-λ)*L_ordinal. "
             "hierarchical: L=λ*BCE(is_zero)+(1-λ)*ordinal(diagnostic).",
    )
    parser.add_argument(
        "--hier_zero_label",
        type=int,
        default=0,
        help="For loss_type=hierarchical: non-diagnostic/zero class (default: 0)",
    )
    parser.add_argument(
        "--cascade_tau",
        type=float,
        default=0.5,
        help="Hierarchical hard-cascade threshold on p0=sigmoid(binary). Predict 0 if p0>=tau.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="CE label smoothing (e.g. 0.05). Often better for segment than dual-head ordinal.",
    )
    parser.add_argument(
        "--loss_weight_focal",
        type=float,
        default=0.0,
        help="If >0 with mixed (dual-head), use focal as the primary head instead of CE",
    )
    parser.add_argument("--use_focal_loss", action="store_true", help="Legacy flag: force focal when loss_type=cross_entropy")
    parser.add_argument("--focal_gamma", type=float, default=1.0, help="Focal loss gamma parameter")

    # Checkpoint options
    parser.add_argument("--ckpt_dir", type=str, default="./spr2026/segmented_crossattn_811", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint")

    # Ablation
    parser.add_argument("--no_segment_fusion", action="store_true", default=False,
                        help="Ablation: disable segment-aware fusion (segment embed, gating, transformer, attn pool). "
                             "Uses mean of 3 segment CLS tokens instead.")

    args, _ = parser.parse_known_args()
    main(args)


#   torchrun --nnodes=1 --nproc_per_node=4 bert_segment_crossattention.py --train_folds "4,5"
# Inference only:
#   torchrun --nnodes=1 --nproc_per_node=2 bert_segment_crossattention.py --train_folds none
# Hierarchical BI-RADS (BCE is_zero + ordinal diagnostic), λ=0.5:
#   torchrun --nnodes=1 --nproc_per_node=4 bert_segment_crossattention_811.py \
#     --loss_type hierarchical --mix_lambda 0.5 --hier_zero_label 0 \
#     --ckpt_dir ./spr2026/segmented_crossattn_811_hier05_zero0 \
#     --train_folds all --epochs 50 --use_amp --amp_dtype bfloat16
