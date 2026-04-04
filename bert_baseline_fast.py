import os
import re
import sys
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, DistributedSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from modules import set_seed, clean_text, stable_hash, maybe_parallelize, save_checkpoint, load_checkpoint, find_latest_ckpt, evaluate, make_encodings, TextDataset, load_model_state_compat, save_confusion_matrix_as_table, save_classification_report_as_table

# don't print warnings
import warnings 
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5

# TRAIN_PATH = "./train_en.csv"
# TEST_PATH = "./test_en.csv"
TRAIN_PATH = "./train_new.csv"
TEST_PATH = "./test.csv"
SUB_PATH = "./submission.csv"


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


@torch.no_grad()
def distributed_evaluate(model, dataset, device, batch_size, num_workers, is_distributed, rank, world_size):
    """Evaluate on all ranks by partitioning dataset and aggregating predictions."""
    if not is_distributed:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        return evaluate(model, loader, device)

    model.eval()
    local_indices = np.arange(len(dataset))[rank::world_size]
    local_subset = Subset(dataset, local_indices.tolist())
    local_loader = DataLoader(
        local_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    local_preds = []
    local_trues = []

    for batch in local_loader:
        labels = batch["labels"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=-1)

        local_preds.extend(pred.cpu().numpy().tolist())
        local_trues.extend(labels.cpu().numpy().tolist())

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, (local_indices.tolist(), local_preds, local_trues))

    pred_full = np.array([], dtype=int)
    val_f1 = 0.0
    if rank == 0:
        pred_full = np.empty(len(dataset), dtype=int)
        true_full = np.empty(len(dataset), dtype=int)
        for idxs, preds, trues in gathered:
            if idxs:
                pred_full[idxs] = np.array(preds, dtype=int)
                true_full[idxs] = np.array(trues, dtype=int)
        val_f1 = float(f1_score(true_full, pred_full, average="macro"))

    f1_tensor = torch.tensor([val_f1], dtype=torch.float32, device=device)
    dist.broadcast(f1_tensor, src=0)
    return float(f1_tensor.item()), pred_full


def build_text_column(series, args):
    text = series.apply(clean_text)
    return text


# def compute_classification_loss(logits, labels, class_weights, use_focal_loss=False, focal_gamma=2.0):
def compute_classification_loss(logits, labels, use_focal_loss=False, focal_gamma=2.0):
    # Use standard cross entropy loss (no weighting)
    ce_loss = F.cross_entropy(logits, labels)
    return ce_loss.mean()

# def compute_classification_loss(logits, labels, class_weights, use_focal_loss=False, focal_gamma=2.0):
#     # Always use reduction='none' to allow per-sample weighting
#     ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")

#     # Apply FN weights only to false negatives for each class
#     # fn_weights should be a tensor of shape [num_classes] or None
#     fn_weights = getattr(compute_classification_loss, 'fn_weights', None)
#     if fn_weights is not None:
#         pred_labels = logits.argmax(dim=1)
#         weights = torch.ones_like(ce_loss)
#         for target_class in range(logits.size(1)):
#             fn_mask = (labels == target_class) & (pred_labels != target_class)
#             weights[fn_mask] *= fn_weights[target_class]
#         ce_loss = ce_loss * weights

#     if not use_focal_loss:
#         return ce_loss.mean()

#     # Focal loss for multi-class classification with optional class weighting.
#     pt = torch.exp(-ce_loss)
#     focal_weight = (1.0 - pt).pow(focal_gamma)
#     return (focal_weight * ce_loss).mean()


def train_fold(
    model,
    train_loader,
    val_loader,
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
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # amp_enabled = bool(use_amp and device.type == "cuda")
    # amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    # scaler = GradScaler("cuda", enabled=amp_enabled)
    
    scaler = GradScaler('cuda', enabled=use_amp)
    best_f1 = -1.0
    best_epoch = 1
    best_state = None
    start_epoch = 1

    if resume and ckpt_dir:
        latest_ckpt = find_latest_ckpt(fold, ckpt_dir)
        if latest_ckpt:
            rank0_print(rank, f"    Found checkpoint: {latest_ckpt}, resuming...")
            model, optimizer, scheduler, start_epoch, best_f1 = load_checkpoint(
                latest_ckpt, model, optimizer, scheduler, device
            )
            best_epoch = start_epoch
            start_epoch += 1

    for epoch in range(start_epoch, epochs + 1):
        # Set epoch for DistributedSampler
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        accumulation_steps = 0

        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast('cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            accumulation_steps += 1
            running_loss += loss.item() * grad_accum_steps

            if accumulation_steps % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        
        if epoch % val_interval == 0:
            if val_dataset is not None:
                val_f1, _ = distributed_evaluate(
                    model=model,
                    dataset=val_dataset,
                    device=device,
                    batch_size=val_batch_size,
                    num_workers=val_num_workers,
                    is_distributed=is_distributed,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                val_f1, _ = evaluate(model, val_loader, device)
            
            avg_loss = running_loss / max(1, len(train_loader))
            rank0_print(rank, f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} | val macro F1: {val_f1:.5f}")

            # Save checkpoint only when validation score improves
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                # Get unwrapped model state for DDP
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                best_state = copy.deepcopy(model_state)
                if ckpt_dir and rank == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, fold, best_f1, ckpt_dir)
        else:
            avg_loss = running_loss / max(1, len(train_loader))
            rank0_print(rank, f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f}")
        
        if is_distributed:
            dist.barrier()

    if best_state is not None:
        # Handle DDP wrapper when loading state
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
    
    # Clean up old checkpoints: keep only the best checkpoint for this fold
    if ckpt_dir and rank == 0:
        pattern = os.path.join(ckpt_dir, f"fold_{fold}_epoch_*.pt")
        all_ckpts = glob.glob(pattern)
        
        if all_ckpts and best_epoch > 0:
            best_ckpt_name = f"fold_{fold}_epoch_{best_epoch}.pt"
            best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)
            
            # Delete all checkpoints except the best one
            for ckpt_path in all_ckpts:
                if ckpt_path != best_ckpt_path:
                    try:
                        os.remove(ckpt_path)
                        rank0_print(rank, f"    Deleted old checkpoint: {os.path.basename(ckpt_path)}")
                    except Exception as e:
                        rank0_print(rank, f"    Warning: Failed to delete {os.path.basename(ckpt_path)}: {e}")
            
            rank0_print(rank, f"    Kept best checkpoint: {best_ckpt_name} (F1={best_f1:.5f})")
    
    return model, best_f1, best_epoch

def train_full_model(
    model,
    train_loader,
    device,
    lr,
    epochs,
    warmup_ratio,
    class_weights,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Loss log setup ---
    loss_log_path = None
    if hasattr(model, 'module') and hasattr(model.module, 'loss_log_path'):
        loss_log_path = model.module.loss_log_path
    elif hasattr(model, 'loss_log_path'):
        loss_log_path = model.loss_log_path
    # If not set, try to use ckpt_dir from args if available
    if loss_log_path is None and hasattr(model, 'ckpt_dir'):
        loss_log_path = os.path.join(model.ckpt_dir, 'loss_log.txt')
    # If still None, fallback to default
    if loss_log_path is None:
        loss_log_path = 'loss_log.txt'
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, 'w') as f:
            f.write('full\tepoch\tloss\n')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)

            optimizer.zero_grad()
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

        train = pd.read_csv(args.train_path)
        test = pd.read_csv(args.test_path)

        assert {"ID", "report", "target"}.issubset(train.columns), train.columns
        assert {"ID", "report"}.issubset(test.columns), test.columns

        train["text"] = build_text_column(train["report"], args)
        test["text"] = build_text_column(test["report"], args)

        train["group"] = train["text"].apply(stable_hash)

        labels_sorted = sorted(train["target"].astype(int).unique().tolist())
        label2id = {label: i for i, label in enumerate(labels_sorted)}
        id2label = {i: label for label, i in label2id.items()}

        y = train["target"].astype(int).map(label2id).values
        X_text = train["text"].values
        groups = train["group"].values

        num_labels = len(labels_sorted)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # ============ Check if checkpoints exist ============
        ckpt_exists = []
        for fold in range(1, args.n_splits + 1):
            pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
            ckpts = glob.glob(pattern)
            if ckpts:
                ckpt_exists.append(fold)
        
        # ============ Determine which folds to train ============
        folds_to_train = set()
        if args.train_folds == "none":
            rank0_print(rank, f"[INFO] Found checkpoints for {len(ckpt_exists)}/{args.n_splits} folds. Running inference only (--train_folds=none)...")
            folds_to_train = set()
        elif args.train_folds == "all":
            folds_to_train = set(range(1, args.n_splits + 1))
            rank0_print(rank, f"[INFO] Training all folds (--train_folds=all)...")
        else:
            try:
                folds_to_train = set(int(f.strip()) for f in args.train_folds.split(",") if f.strip())
                if not folds_to_train.issubset(set(range(1, args.n_splits + 1))):
                    raise ValueError(f"Fold numbers must be in range [1, {args.n_splits}]")
                rank0_print(rank, f"[INFO] Training folds: {sorted(folds_to_train)}")
            except Exception as e:
                rank0_print(rank, f"[ERROR] Invalid --train_folds format: {e}")
                return
        
        # If no training needed, proceed directly to inference/submission
        if not folds_to_train:
            if len(ckpt_exists) < args.n_splits:
                rank0_print(rank, f"[WARN] Only {len(ckpt_exists)}/{args.n_splits} folds have checkpoints. Using available checkpoints for submission...")
            
            rank0_print(rank, f"[INFO] Found checkpoints for {len(ckpt_exists)}/{args.n_splits} folds. Generating submission from existing checkpoints...")
            
            if is_distributed:
                dist.barrier()
            
            # Only rank 0 does inference to avoid redundant GPU memory usage
            if rank == 0:
                # First compute OOF validation metrics from existing checkpoints
                rank0_print(rank, "\n[INFO] Computing OOF validation metrics from existing checkpoints...")
                oof_pred_inference = np.empty(len(train), dtype=int)
                oof_scores_inference = []
                
                # Initialize fold validation summary file
                fold_validation_summary_path = os.path.join(args.ckpt_dir, "fold_validation_summary.txt")
                with open(fold_validation_summary_path, "w") as f:
                    f.write("=" * 80 + "\n")
                    f.write("FOLD VALIDATION RESULTS SUMMARY (Inference Mode)\n")
                    f.write("=" * 80 + "\n\n")
                
                gkf_check = GroupKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
                for fold_check, (tr_idx_check, va_idx_check) in enumerate(gkf_check.split(X_text, y, groups=groups), 1):
                    pattern = os.path.join(args.ckpt_dir, f"fold_{fold_check}_epoch_*.pt")
                    ckpts_check = glob.glob(pattern)
                    
                    if not ckpts_check:
                        rank0_print(rank, f"  Fold {fold_check}: No checkpoint found (skipped)")
                        continue
                    
                    # Find best checkpoint
                    best_ckpt_check = None
                    best_f1_check = -1.0
                    
                    for ckpt_path in ckpts_check:
                        ck = torch.load(ckpt_path, map_location="cpu")
                        f1 = float(ck.get("best_f1", -1.0))
                        if f1 > best_f1_check:
                            best_f1_check = f1
                            best_ckpt_check = ckpt_path
                    
                    # Load model and evaluate on validation set
                    model = AutoModelForSequenceClassification.from_pretrained(
                        args.model_name,
                        num_labels=num_labels,
                        ignore_mismatched_sizes=True
                    ).to(device)
                    ck = torch.load(best_ckpt_check, map_location=device)
                    load_model_state_compat(
                        model,
                        ck["model_state_dict"],
                        strict=False,
                        log_prefix=f"fold {fold_check} (inference-oof): ",
                    )
                    model.eval()
                    
                    # Prepare validation data for this fold
                    va_texts = X_text[va_idx_check].tolist()
                    y_va = y[va_idx_check]
                    va_enc = make_encodings(tokenizer, va_texts, args.max_length)
                    va_ds = TextDataset(va_enc, torch.tensor(y_va, dtype=torch.long))
                    va_loader = DataLoader(
                        va_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=args.num_workers > 0,
                    )
                    
                    # Inference on this fold's validation set
                    fold_preds = []
                    with torch.no_grad():
                        for batch in va_loader:
                            input_ids = batch["input_ids"].to(device, non_blocking=True)
                            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                            pred_ids = torch.argmax(logits, dim=-1)
                            fold_preds.append(pred_ids.cpu().numpy())
                    
                    fold_pred = np.concatenate(fold_preds, axis=0)
                    oof_pred_inference[va_idx_check] = fold_pred
                    
                    # Compute fold-level F1
                    fold_f1 = f1_score(y_va, fold_pred, average="macro")
                    oof_scores_inference.append(fold_f1)
                    
                    # Output confusion matrix for this fold
                    try:
                        fold_cm = confusion_matrix(y_va, fold_pred)
                        print(f"\n{'='*80}")
                        print(f"[INFO] Fold {fold_check} Validation Confusion Matrix (from checkpoint):")
                        print(fold_cm)
                        sys.stdout.flush()
                        
                        fold_report = classification_report(
                            y_va, fold_pred,
                            target_names=[str(id2label[i]) for i in range(num_labels)]
                        )
                        print(f"\n[INFO] Fold {fold_check} Classification Report:")
                        print(fold_report)
                        print(f"{'='*80}\n")
                        sys.stdout.flush()
                        
                        # Append to fold validation summary
                        with open(fold_validation_summary_path, "a") as f:
                            f.write(f"Fold {fold_check} Validation Results (from checkpoint {os.path.basename(best_ckpt_check)}):\n")
                            f.write(f"  F1 Score (macro): {fold_f1:.5f}\n")
                            f.write(f"\nConfusion Matrix:\n{fold_cm}\n")
                            f.write(f"\nClassification Report:\n{fold_report}\n")
                            f.write("-" * 80 + "\n\n")
                    except Exception as e:
                        print(f"[ERROR] Failed to compute confusion matrix for fold {fold_check}: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                    
                    rank0_print(rank, f"  Fold {fold_check}: F1={fold_f1:.5f} from {os.path.basename(best_ckpt_check)}")
                
                # Output OOF metrics if all folds have predictions
                if len(oof_scores_inference) == args.n_splits:
                    oof_f1_inference = f1_score(y, oof_pred_inference, average="macro")
                    cv_mean_inference = float(np.mean(oof_scores_inference))
                    cv_std_inference = float(np.std(oof_scores_inference))
                    
                    try:
                        cm_inf = confusion_matrix(y, oof_pred_inference)
                        print(f"\n{'='*80}")
                        print(f"[INFO] OOF Confusion Matrix (from inference mode):")
                        print(cm_inf)
                        sys.stdout.flush()
                        
                        report_inf = classification_report(y, oof_pred_inference, target_names=[str(id2label[i]) for i in range(num_labels)])
                        print(f"\n[INFO] OOF Classification Report:")
                        print(report_inf)
                        print(f"{'='*80}\n")
                        sys.stdout.flush()
                        
                        # Save OOF analysis from inference (text format)
                        oof_analysis_path = os.path.join(args.ckpt_dir, "oof_analysis_inference.txt")
                        with open(oof_analysis_path, "w") as f:
                            f.write(f"OOF macro F1: {oof_f1_inference:.5f}\n")
                            f.write(f"CV mean: {cv_mean_inference:.5f}, CV std: {cv_std_inference:.5f}\n")
                            f.write("\nConfusion Matrix:\n")
                            f.write(str(cm_inf))
                            f.write("\n\nClassification Report:\n")
                            f.write(report_inf)
                        print(f"[INFO] OOF analysis (text) saved to {oof_analysis_path}")
                        
                        # Save confusion matrix and report as CSV tables
                        cm_csv_path = os.path.join(args.ckpt_dir, "oof_confusion_matrix_inference.csv")
                        report_csv_path = os.path.join(args.ckpt_dir, "oof_classification_report_inference.csv")
                        save_confusion_matrix_as_table(cm_inf, id2label, cm_csv_path)
                        save_classification_report_as_table(y, oof_pred_inference, id2label, report_csv_path)
                        print(f"[INFO] Confusion matrix saved to {cm_csv_path}")
                        print(f"[INFO] Classification report saved to {report_csv_path}")
                        
                        # Append OOF results to fold validation summary
                        with open(fold_validation_summary_path, "a") as f:
                            f.write("\n" + "=" * 80 + "\n")
                            f.write("OUT-OF-FOLD (OOF) VALIDATION RESULTS\n")
                            f.write("=" * 80 + "\n\n")
                            f.write(f"CV mean F1 (macro): {cv_mean_inference:.5f}\n")
                            f.write(f"CV std F1 (macro): {cv_std_inference:.5f}\n")
                            f.write(f"OOF macro F1: {oof_f1_inference:.5f}\n")
                            f.write(f"\nConfusion Matrix:\n{cm_inf}\n")
                            f.write(f"\nClassification Report:\n{report_inf}\n")
                            f.write("=" * 80 + "\n")
                        print(f"[INFO] OOF results appended to {fold_validation_summary_path}")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"[ERROR] Failed to compute OOF confusion matrix: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                    
                    rank0_print(rank, f"\n[INFO] Inference-mode CV stats: mean={cv_mean_inference:.5f}, std={cv_std_inference:.5f}, OOF F1={oof_f1_inference:.5f}")
                else:
                    rank0_print(rank, f"[WARN] Only {len(oof_scores_inference)}/{args.n_splits} folds have checkpoints. OOF metrics unavailable.")
                
                # Now proceed with test submission
                # Load checkpoints and generate submission
                test_text = test["text"].tolist()
                test_enc = make_encodings(tokenizer, test_text, args.max_length)
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
                used_folds = []
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
                    
                    rank0_print(rank, f"  Fold {fold}: {os.path.basename(best_ckpt)} (F1={best_f1:.5f}, epoch={best_epoch})")
                    used_folds.append(fold)
                    
                    # Load model and checkpoint
                    model = AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=num_labels,
            ignore_mismatched_sizes=True).to(device)
                    ck = torch.load(best_ckpt, map_location=device)
                    load_model_state_compat(
                        model,
                        ck["model_state_dict"],
                        strict=False,
                        log_prefix=f"fold {fold} (inference-submit): ",
                    )
                    model.eval()
                    
                    # Inference: store per-fold class predictions for majority vote
                    fold_pred_ids = []
                    with torch.no_grad():
                        for batch in test_loader:
                            input_ids = batch["input_ids"].to(device, non_blocking=True)
                            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                            pred_ids = torch.argmax(logits, dim=-1)
                            fold_pred_ids.append(pred_ids.cpu().numpy())
                    
                    all_fold_preds.append(np.concatenate(fold_pred_ids, axis=0))
                
                # Majority vote across folds
                if not all_fold_preds:
                    rank0_print(rank, "[ERROR] No checkpoints found for any fold!")
                    return
                
                pred_matrix = np.stack(all_fold_preds, axis=0)  # [n_folds, n_samples]
                majority_ids = []
                for i in range(pred_matrix.shape[1]):
                    vote_counts = np.bincount(pred_matrix[:, i], minlength=num_labels)
                    majority_ids.append(int(np.argmax(vote_counts)))
                pred_test = [id2label[i] for i in majority_ids]
                
                submission = pd.DataFrame({
                    "ID": test["ID"],
                    "target": pred_test,
                })

                save_dir = os.path.dirname(args.sub_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                elif save_dir and not os.path.isdir(save_dir):
                    raise ValueError(f"{save_dir} exists but is not a directory.")
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

        gkf = GroupKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)

        oof_pred = np.empty(len(train), dtype=int)
        fold_scores = []
        best_epochs = []

        # Initialize fold validation summary file
        fold_validation_summary_path = os.path.join(args.ckpt_dir, "fold_validation_summary.txt")
        with open(fold_validation_summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("FOLD VALIDATION RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_text, y, groups=groups), 1):
            # Skip folds not in training list
            if fold not in folds_to_train:
                rank0_print(rank, f"\nFold {fold}/{args.n_splits} - SKIPPED (not in --train_folds)")
                continue
            
            rank0_print(rank, f"\nFold {fold}/{args.n_splits}")
            
            # Check if checkpoint exists for this fold (to load and continue training)
            pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
            existing_ckpts = glob.glob(pattern)
            if existing_ckpts:
                rank0_print(rank, f"  [INFO] Found {len(existing_ckpts)} existing checkpoint(s) for fold {fold}. Will resume training from checkpoint.")
            else:
                rank0_print(rank, f"  [INFO] No checkpoint found for fold {fold}. Training from scratch.")

            tr_texts = X_text[tr_idx].tolist()
            va_texts = X_text[va_idx].tolist()

            y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
            y_va = torch.tensor(y[va_idx], dtype=torch.long)

            tr_enc = make_encodings(tokenizer, tr_texts, args.max_length)
            va_enc = make_encodings(tokenizer, va_texts, args.max_length)

            tr_ds = TextDataset(tr_enc, y_tr)
            va_ds = TextDataset(va_enc, y_va)

            # Use DistributedSampler for training
            tr_sampler = None
            if is_distributed:
                tr_sampler = DistributedSampler(
                    tr_ds,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=RANDOM_STATE + fold,
                    drop_last=True,
                )
            
            tr_loader = DataLoader(
                tr_ds,
                batch_size=args.batch_size,
                shuffle=(tr_sampler is None),
                sampler=tr_sampler,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
            )
            
            # Validation loader: DO NOT use DistributedSampler
            # Only rank 0 will evaluate during training; other ranks skip
            va_loader = DataLoader(
                va_ds,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=None,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=num_labels,
            )
            
            # Wrap with DDP if distributed, otherwise use maybe_parallelize
            if is_distributed:
                model = model.to(device)
                model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            else:
                model = maybe_parallelize(model, device, use_data_parallel=not args.no_data_parallel)

            model, best_f1, best_epoch = train_fold(
                model=model,
                train_loader=tr_loader,
                val_loader=va_loader,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                warmup_ratio=args.warmup_ratio,
                class_weights=class_weights,
                fold=fold,
                ckpt_dir=args.ckpt_dir,
                resume=args.resume,
                use_amp=args.use_amp,
                grad_accum_steps=args.grad_accum_steps,
                val_interval=args.val_interval,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                val_dataset=va_ds,
                val_batch_size=args.batch_size,
                val_num_workers=args.num_workers,
            )

            val_f1, pred_va = distributed_evaluate(
                model=model,
                dataset=va_ds,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
            )

            if rank == 0:
                oof_pred[va_idx] = pred_va
                
                # Output confusion matrix and classification report for this fold
                try:
                    y_va_np = y_va.cpu().numpy() if isinstance(y_va, torch.Tensor) else y_va.numpy()
                    fold_cm = confusion_matrix(y_va_np, pred_va)
                    
                    print(f"\n{'='*80}")
                    print(f"[INFO] Fold {fold} Confusion Matrix:")
                    print(fold_cm)
                    sys.stdout.flush()
                    
                    fold_report = classification_report(
                        y_va_np, pred_va, 
                        target_names=[str(id2label[i]) for i in range(num_labels)]
                    )
                    print(f"\n[INFO] Fold {fold} Classification Report:")
                    print(fold_report)
                    print(f"{'='*80}\n")
                    sys.stdout.flush()
                    
                    # Save fold-level analysis (text format)
                    fold_analysis_path = os.path.join(args.ckpt_dir, f"fold_{fold}_analysis.txt")
                    with open(fold_analysis_path, "w") as f:
                        f.write(f"Fold {fold} Validation Analysis\n")
                        f.write(f"F1 Score (macro): {val_f1:.5f}\n")
                        f.write(f"\nConfusion Matrix:\n")
                        f.write(str(fold_cm))
                        f.write(f"\n\nClassification Report:\n")
                        f.write(fold_report)
                    
                    # Append to fold validation summary
                    with open(fold_validation_summary_path, "a") as f:
                        f.write(f"Fold {fold} Validation Results:\n")
                        f.write(f"  F1 Score (macro): {val_f1:.5f}\n")
                        f.write(f"  Best Epoch: {best_epoch}\n")
                        f.write(f"  Best F1 (during training): {best_f1:.5f}\n")
                        f.write(f"\nConfusion Matrix:\n{fold_cm}\n")
                        f.write(f"\nClassification Report:\n{fold_report}\n")
                        f.write("-" * 80 + "\n\n")
                    
                    # Save confusion matrix and report as CSV tables
                    fold_cm_csv_path = os.path.join(args.ckpt_dir, f"fold_{fold}_confusion_matrix.csv")
                    fold_report_csv_path = os.path.join(args.ckpt_dir, f"fold_{fold}_classification_report.csv")
                    save_confusion_matrix_as_table(fold_cm, id2label, fold_cm_csv_path)
                    save_classification_report_as_table(y_va_np, pred_va, id2label, fold_report_csv_path)
                    
                    print(f"[INFO] Fold {fold} analysis (text) saved to {fold_analysis_path}")
                    print(f"[INFO] Fold {fold} confusion matrix saved to {fold_cm_csv_path}")
                    print(f"[INFO] Fold {fold} classification report saved to {fold_report_csv_path}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[ERROR] Failed to compute confusion matrix for fold {fold}: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
            
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
        cv_mean = float(np.mean(fold_scores)) if fold_scores else 0.0
        cv_std = float(np.std(fold_scores)) if fold_scores else 0.0
        oof_score = f1_score(y, oof_pred, average="macro") if len(fold_scores) == args.n_splits else -1.0
        auto_final_epochs = max(1, int(round(float(np.mean(best_epochs))))) if best_epochs else 1

        rank0_print(rank, "\nCV mean:", f"{cv_mean:.5f}", "| CV std:", f"{cv_std:.5f}")
        if oof_score >= 0:
            rank0_print(rank, "OOF macro F1:", f"{oof_score:.5f}")
            
            # Output confusion matrix and classification report
            try:
                cm = confusion_matrix(y, oof_pred)
                print(f"\n{'='*80}")
                print(f"[INFO] OOF Confusion Matrix:")
                print(cm)
                sys.stdout.flush()
                
                report = classification_report(y, oof_pred, target_names=[str(id2label[i]) for i in range(num_labels)])
                print(f"\n[INFO] OOF Classification Report:")
                print(report)
                print(f"{'='*80}\n")
                sys.stdout.flush()
                
                # Save confusion matrix and report to file (text format)
                oof_analysis_path = os.path.join(args.ckpt_dir, "oof_analysis.txt")
                with open(oof_analysis_path, "w") as f:
                    f.write(f"OOF macro F1: {oof_score:.5f}\n")
                    f.write(f"CV mean: {cv_mean:.5f}, CV std: {cv_std:.5f}\n")
                    f.write("\nConfusion Matrix:\n")
                    f.write(str(cm))
                    f.write("\n\nClassification Report:\n")
                    f.write(report)
                
                # Save confusion matrix and report as CSV tables
                oof_cm_csv_path = os.path.join(args.ckpt_dir, "oof_confusion_matrix.csv")
                oof_report_csv_path = os.path.join(args.ckpt_dir, "oof_classification_report.csv")
                save_confusion_matrix_as_table(cm, id2label, oof_cm_csv_path)
                save_classification_report_as_table(y, oof_pred, id2label, oof_report_csv_path)
                
                print(f"[INFO] OOF analysis (text) saved to {oof_analysis_path}")
                print(f"[INFO] OOF confusion matrix saved to {oof_cm_csv_path}")
                print(f"[INFO] OOF classification report saved to {oof_report_csv_path}")
                
                # Append OOF results to fold validation summary
                with open(fold_validation_summary_path, "a") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("OUT-OF-FOLD (OOF) VALIDATION RESULTS\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"CV mean F1 (macro): {cv_mean:.5f}\n")
                    f.write(f"CV std F1 (macro): {cv_std:.5f}\n")
                    f.write(f"OOF macro F1: {oof_score:.5f}\n")
                    f.write(f"\nConfusion Matrix:\n{cm}\n")
                    f.write(f"\nClassification Report:\n{report}\n")
                    f.write("=" * 80 + "\n")
                print(f"[INFO] OOF results appended to {fold_validation_summary_path}")
                sys.stdout.flush()
            except Exception as e:
                print(f"[ERROR] Failed to compute OOF confusion matrix: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        else:
            rank0_print(rank, "[INFO] OOF score unavailable (partial folds trained)")
        rank0_print(rank, "Best epochs per fold:", best_epochs)

        if is_distributed:
            dist.barrier()

        rank0_print(rank, "\n[INFO] Ensemble test predictions using available checkpoints...")
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
        used_folds = []
        
        # Only rank 0 does inference to save GPU memory
        if rank == 0:
            for fold in range(1, args.n_splits + 1):
                pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
                ckpts = glob.glob(pattern)

                if not ckpts:
                    rank0_print(rank, f"  Fold {fold}: NO CHECKPOINT (skipped)")
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

                status = "(TRAINED)" if fold in folds_to_train else "(FROM PREVIOUS)"
                rank0_print(rank, f"  Fold {fold}: {os.path.basename(best_ckpt)} (F1={best_f1:.5f}, epoch={best_epoch}) {status}")
                used_folds.append(fold)

                # Load model and checkpoint
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name,
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True
                ).to(device)
                ck = torch.load(best_ckpt, map_location=device)
                load_model_state_compat(
                    model,
                    ck["model_state_dict"],
                    strict=False,
                    log_prefix=f"fold {fold} (ensemble-submit): ",
                )
                model.eval()

                # Inference: store per-fold class predictions for majority vote
                fold_pred_ids = []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                        pred_ids = torch.argmax(logits, dim=-1)
                        fold_pred_ids.append(pred_ids.cpu().numpy())

                all_fold_preds.append(np.concatenate(fold_pred_ids, axis=0))

            # Majority vote across folds
            if not all_fold_preds:
                rank0_print(rank, "[ERROR] No fold predictions collected from any fold!")
                return
            
            pred_matrix = np.stack(all_fold_preds, axis=0)  # [n_folds, n_samples]
            majority_ids = []
            for i in range(pred_matrix.shape[1]):
                vote_counts = np.bincount(pred_matrix[:, i], minlength=num_labels)
                majority_ids.append(int(np.argmax(vote_counts)))
            pred_test = [id2label[i] for i in majority_ids]
            
            submission = pd.DataFrame({
                "ID": test["ID"],
                "target": pred_test,
            })

            save_dir = os.path.dirname(args.sub_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            elif save_dir and not os.path.isdir(save_dir):
                raise ValueError(f"{save_dir} exists but is not a directory.")
            submission.to_csv(args.sub_path, index=False)
            rank0_print(rank, f"\n[INFO] Saved submission using {len(used_folds)} folds: {sorted(used_folds)}")
            rank0_print(rank, f"[INFO] Path: {args.sub_path}")
            rank0_print(rank, submission.head())
    
    finally:
        cleanup_distributed(is_distributed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT baseline for SPR2026 report classification (FAST + Checkpoints + Distributed)")
    parser.add_argument(
        "--fn_weights",
        type=str,
        default=None,
        help="Comma-separated per-class FN weights (e.g. 1,2,1,1 for 4 classes). Multiplies class_weights for each class."
    )
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--sub_path", type=str, default=SUB_PATH)

    # parser.add_argument("--model_name", type=str, default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--model_name", type=str, default="pucpr/biobertpt-all")
    # parser.add_argument("--model_name", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/mlm_pretrain_ckpt/checkpoint-5906")
    # parser.add_argument("--model_name", type=str, default="google-bert/bert-base-cased") # neuralmind/bert-base-portuguese-cased
    # parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT",
                        # help="Model name. Medical options: emilyalsentzer/distilbert-base-medical, allenai/scibert_scivocab_cased, "
                        #      "dmis-lab/biobert-base-cased-v1.1, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_data_parallel", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    # parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=0,
        help="Final training epochs. <=0 means auto from CV best epochs.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--train_folds",
        type=str,
        default="none",
        help="Which folds to train: 'all' (all folds), 'none' (inference only), or comma-separated fold numbers (e.g., '1,3,5'). "
             "When 'none', uses existing checkpoints for submission. When specific folds, trains only those folds from their checkpoints."
    )
    
    # Speedup options
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision FP16")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss instead of cross entropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (only when --use_focal_loss)")
    
    # Checkpoint options
    parser.add_argument("--ckpt_dir", type=str, default="spr2026/pretrained_biobertpt", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint")
    parser.add_argument("--save_final_model", action="store_true", default=True, help="Save final model weights")

    args, _ = parser.parse_known_args()
    main(args)

# ============ TORCHRUN USAGE ============
# Single node with 2 GPUs:
#   torchrun --nnodes=1 --nproc_per_node=2 bert_baseline_fast.py --train_folds "1,3"
#
# Inference only (run on any GPU setup):
#   torchrun --nnodes=1 --nproc_per_node=1 bert_baseline_fast.py --train_folds none
# ======================================
