# use the splited data to train the segmented bert with cross attention
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    maybe_parallelize, save_checkpoint, load_checkpoint, find_latest_ckpt, setup_distributed,cleanup_distributed, rank0_print,
    distributed_evaluate, make_encodings, TextDataset, load_model_state_compat, save_confusion_matrix_as_table, save_classification_report_as_table, 
    save_attention_heatmap_for_sample, maybe_extract_attention_after_run,
)
from data_segmentation import split_report_by_sections, TokenLevelSegmentedBert, make_segmented_encodings, SegmentedTextDataset

import warnings 
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_PATH = "./train_split.csv"  # Use pre-segmented data
TEST_PATH = "./test.csv"
SUB_PATH = "./submission.csv"


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
    use_focal_loss=False,
    focal_gamma=2.0,
    early_stop_patience=3,
    start_epoch_override=1,
    best_f1_override=-1.0
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler('cuda', enabled=use_amp)
    best_f1 = best_f1_override
    best_epoch = 1
    best_state = None
    start_epoch = start_epoch_override
    early_stop_counter = 0

    if start_epoch > 1:
        rank0_print(rank, f"    Starting from epoch {start_epoch} (checkpoint epoch {start_epoch-1}), best_f1={best_f1:.5f}")

    for epoch in range(start_epoch, epochs + 1):
        # Set epoch for DistributedSampler
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        running_constraint_loss = 0.0
        running_constraint_active = 0
        accumulation_steps = 0


        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items() if k != "labels"}

            with autocast('cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(
                    clinical_input_ids=batch_device['clinical_input_ids'],
                    clinical_attention_mask=batch_device['clinical_attention_mask'],
                    findings_input_ids=batch_device['findings_input_ids'],
                    findings_attention_mask=batch_device['findings_attention_mask'],
                    comparative_input_ids=batch_device['comparative_input_ids'],
                    comparative_attention_mask=batch_device['comparative_attention_mask']
                )

                loss = compute_classification_loss(
                    outputs,
                    labels,
                    class_weights=class_weights,
                    use_focal_loss=use_focal_loss,
                    focal_gamma=focal_gamma
                )

                extra_constraint_loss = outputs.new_tensor(0.0)
                running_constraint_loss += extra_constraint_loss.item()
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
                val_f1, pred_va = distributed_evaluate(
                    model=model,
                    dataset=val_dataset,
                    device=device,
                    batch_size=val_batch_size,
                    num_workers=val_num_workers,
                    is_distributed=is_distributed,
                    rank=rank,
                    world_size=world_size,
                )
                # Compute prediction distribution for debugging
                
            else:
                val_f1, pred_va = distributed_evaluate(
                    model=model,
                    dataset=None,
                    device=device,
                    batch_size=val_batch_size,
                    num_workers=val_num_workers,
                    is_distributed=is_distributed,
                    rank=rank,
                    world_size=world_size,
                )
            
            avg_loss = running_loss / max(1, len(train_loader))
            rank0_print(rank, f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} | val macro F1: {val_f1:.5f} ")
            
            # Return to training mode after evaluation
            model.train()

            # Save checkpoint only when validation score improves
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                # Get unwrapped model state for DDP
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
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
            avg_loss = running_loss / max(1, len(train_loader))
            if constraint_is_on:
                base_model = model.module if hasattr(model, 'module') else model
                constraint_w_val = float(base_model.get_carcinoma_constraint_w().detach().item())
                avg_constraint = running_constraint_loss / max(1, len(train_loader))
                rank0_print(
                    rank,
                    f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} (no validation) "
                    f"| carcinoma_w: {constraint_w_val:.6f} | carcinoma_loss: {avg_constraint:.6f} "
                    f"| carcinoma_samples: {running_constraint_active}"
                )
            else:
                rank0_print(rank, f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} (no validation)")
        
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
                    except:
                        pass
            
            rank0_print(rank, f"    Kept best checkpoint: {best_ckpt_name} (F1={best_f1:.5f})")
    
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
    if loss_log_path is None and hasattr(model, 'ckpt_dir'):
        loss_log_path = os.path.join(model.ckpt_dir, 'loss_log.txt')
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
            loss = compute_classification_loss(
                outputs,
                labels,
                class_weights=class_weights,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma
            )

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
    # ============ Handle txt-to-PNG regeneration mode ============
    if args.regenerate_attention_from_txt:
        """Quick mode: regenerate PNG from saved txt without training"""
        txt_path = args.regenerate_attention_from_txt
        output_png = txt_path.replace('_attention_report.txt', '.png')
        
        print(f"[INFO] Regenerate PNG mode activated")
        print(f"[INPUT] txt: {txt_path}")
        print(f"[OUTPUT] png: {output_png}")
        
        success = regenerate_attention_heatmap_from_txt(
            txt_path=txt_path,
            output_png_path=output_png,
            max_tokens=args.attention_max_tokens
        )
        
        if success:
            print(f"[SUCCESS] PNG regenerated successfully!")
        else:
            print(f"[FAILED] Could not regenerate PNG from txt")
        return

    # ============ Initialize Distributed Training ============
    rank, local_rank, world_size, is_distributed = setup_distributed()
    try:
        set_seed(RANDOM_STATE + rank)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank0_print(rank, "Device:", device, "| world_size:", world_size, "| local_rank:", local_rank)
        if torch.cuda.is_available():
            rank0_print(rank, "GPU count:", torch.cuda.device_count())
            torch.backends.cudnn.benchmark = True
        
        # Log focal loss configuration
        if args.use_focal_loss:
            rank0_print(rank, f"[INFO] Using FOCAL LOSS with gamma={args.focal_gamma}")
        else:
            rank0_print(rank, "[INFO] Using standard CROSS ENTROPY loss")

        train = pd.read_csv(args.train_path)
        test = pd.read_csv(args.test_path)

        assert {"ID", "target", "clinical_indication", "findings", "comparative_analysis"}.issubset(train.columns), train.columns
        assert {"ID", "report"}.issubset(test.columns), test.columns

        train["group"] = train["ID"].apply(stable_hash)

        labels_sorted = sorted(train["target"].astype(int).unique().tolist())
        label2id = {label: i for i, label in enumerate(labels_sorted)}
        id2label = {i: label for label, i in label2id.items()}


        y = train["target"].astype(int).map(label2id).values
        
        # Use pre-segmented columns directly
        X_clinical = train["clinical_indication"].values
        X_findings = train["findings"].values
        X_comparative = train["comparative_analysis"].values
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
                    raise ValueError(f"Invalid fold numbers")
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
                
                # Create misclassification summary file
                misclass_summary_path = os.path.join(args.ckpt_dir, "oof_misclassification_summary.txt")
                with open(misclass_summary_path, "w") as f:
                    f.write("=" * 100 + "\n")
                    f.write("OUT-OF-FOLD MISCLASSIFICATION SUMMARY\n")
                    f.write("=" * 100 + "\n\n")
                
                # Create detailed misclassification report file
                misclass_details_path = os.path.join(args.ckpt_dir, "oof_misclassification_details.txt")
                with open(misclass_details_path, "w") as f:
                    f.write("=" * 100 + "\n")
                    f.write("OUT-OF-FOLD MISCLASSIFICATION DETAILS\n")
                    f.write("=" * 100 + "\n\n")
                
                # Track all misclassified samples across all folds
                all_misclass_samples = []
                
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
                    
                    va_enc = make_segmented_encodings(tokenizer, va_clinical, va_findings, va_comparative, args.max_length)
                    va_ds = SegmentedTextDataset(va_enc, y_va)
                    
                    va_model = TokenLevelSegmentedBert(
                        args.model_name,
                        num_labels=num_labels
                    )
                    ckpt_data = torch.load(latest_ckpt, map_location=device)
                    va_model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
                    va_model = va_model.to(device)
                    
                    val_f1, pred_va = distributed_evaluate(
                        model=va_model,
                        dataset=va_ds,
                        device=device,
                        batch_size=args.batch_size,
                        num_workers=0,
                        is_distributed=False,
                        rank=0,
                        world_size=1,
                    )
                    
                    oof_pred_inference[va_idx_check] = pred_va
                    oof_scores_inference.append(val_f1)
                    
                    # ===== Misclassification Analysis =====
                    y_va_np = y[va_idx_check]
                    misclass_mask = (y_va_np != pred_va)
                    misclass_indices = np.where(misclass_mask)[0]
                    misclass_count = np.sum(misclass_mask)
                    accuracy = np.sum(~misclass_mask) / len(y_va_np)
                    
                    # Build confusion matrix
                    cm = confusion_matrix(y_va_np, pred_va, labels=list(range(num_labels)))
                    report_str = classification_report(y_va_np, pred_va, 
                                                      target_names=[str(id2label[i]) for i in range(num_labels)])
                    
                    rank0_print(rank, f"  Fold {fold_check}: F1 = {val_f1:.5f} | Accuracy = {accuracy:.5f} | Misclass = {misclass_count}/{len(y_va_np)}")
                    
                    with open(fold_validation_summary_path, "a") as f:
                        f.write(f"\nFold {fold_check}: F1 = {val_f1:.5f}\n")
                        f.write(f"Accuracy: {accuracy:.5f} | Misclassified: {misclass_count}/{len(y_va_np)}\n")
                        f.write(f"Confusion Matrix:\n{cm}\n\n")
                        f.write(f"Classification Report:\n{report_str}\n")
                        f.write("=" * 80 + "\n\n")
                    
                    # Save misclassification statistics for this fold
                    with open(misclass_summary_path, "a") as f:
                        f.write(f"\n{'='*100}\n")
                        f.write(f"FOLD {fold_check} MISCLASSIFICATION ANALYSIS\n")
                        f.write(f"{'='*100}\n")
                        f.write(f"Total samples: {len(y_va_np)}\n")
                        f.write(f"Misclassified: {misclass_count}\n")
                        f.write(f"Accuracy: {accuracy:.5f}\n\n")
                        
                        # Error statistics by true class
                        for true_label in range(num_labels):
                            true_mask = (y_va_np == true_label)
                            true_count = np.sum(true_mask)
                            if true_count > 0:
                                # Count errors for this true label
                                errors_for_true = np.sum(misclass_mask & true_mask)
                                error_rate = errors_for_true / true_count
                                f.write(f"  Label {id2label[true_label]}: {true_count} samples, {errors_for_true} errors ({error_rate:.2%})\n")
                                
                                # Show which labels they were predicted as
                                pred_for_true = pred_va[true_mask]
                                for pred_label in range(num_labels):
                                    if pred_label != true_label:
                                        pred_count = np.sum(pred_for_true == pred_label)
                                        if pred_count > 0:
                                            f.write(f"      -> Predicted as {id2label[pred_label]}: {pred_count}\n")
                    
                    # Save detailed misclassification samples for this fold
                    train_ids = train.iloc[va_idx_check]["ID"].values
                    with open(misclass_details_path, "a") as f:
                        f.write(f"\n{'='*100}\n")
                        f.write(f"FOLD {fold_check} MISCLASSIFIED SAMPLES DETAILS\n")
                        f.write(f"{'='*100}\n\n")
                        
                        if misclass_count == 0:
                            f.write("No misclassified samples in this fold.\n\n")
                        else:
                            for idx, local_idx in enumerate(misclass_indices):
                                global_idx = va_idx_check[local_idx]
                                sample_id = train_ids[local_idx]
                                true_label = y_va_np[local_idx]
                                pred_label = pred_va[local_idx]
                                
                                f.write(f"\n--- Sample {idx + 1} ---\n")
                                f.write(f"ID: {sample_id}\n")
                                f.write(f"True Label: {id2label[true_label]}\n")
                                f.write(f"Predicted Label: {id2label[pred_label]}\n")
                                f.write(f"\nClinical Indication:\n{train.iloc[global_idx]['clinical_indication']}\n\n")
                                f.write(f"Findings:\n{train.iloc[global_idx]['findings']}\n\n")
                                f.write(f"Comparative Analysis:\n{train.iloc[global_idx]['comparative_analysis']}\n")
                                f.write("-" * 100 + "\n")
                    
                    all_misclass_samples.extend([(fold_check, i, y_va_np[i], pred_va[i]) for i in misclass_indices])
                
                # Output OOF metrics if all folds have predictions
                if len(oof_scores_inference) == args.n_splits:
                    cv_mean = float(np.mean(oof_scores_inference))
                    cv_std = float(np.std(oof_scores_inference))
                    oof_score = f1_score(y, oof_pred_inference, average="macro")
                    
                    rank0_print(rank, f"\nCV mean: {cv_mean:.5f} | CV std: {cv_std:.5f}")
                    rank0_print(rank, f"OOF macro F1: {oof_score:.5f}")
                    
                    # Overall OOF misclassification statistics
                    oof_accuracy = np.sum(y == oof_pred_inference) / len(y)
                    oof_misclass_count = np.sum(y != oof_pred_inference)
                    
                    rank0_print(rank, f"OOF Accuracy: {oof_accuracy:.5f} | Total Misclassified: {oof_misclass_count}/{len(y)}")
                    
                    with open(fold_validation_summary_path, "a") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"OVERALL OUT-OF-FOLD METRICS\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"CV mean: {cv_mean:.5f} | CV std: {cv_std:.5f}\n")
                        f.write(f"OOF macro F1: {oof_score:.5f}\n")
                        f.write(f"OOF Accuracy: {oof_accuracy:.5f}\n")
                        f.write(f"Total Misclassified: {oof_misclass_count}/{len(y)}\n")
                        
                        # Overall confusion matrix and classification report
                        cm_overall = confusion_matrix(y, oof_pred_inference, labels=list(range(num_labels)))
                        report_overall = classification_report(y, oof_pred_inference,
                                                              target_names=[str(id2label[i]) for i in range(num_labels)])
                        f.write(f"\nOverall Confusion Matrix:\n{cm_overall}\n\n")
                        f.write(f"Overall Classification Report:\n{report_overall}\n")
                    
                    # Write overall misclassification summary
                    with open(misclass_summary_path, "a") as f:
                        f.write(f"\n{'='*100}\n")
                        f.write(f"OVERALL OUT-OF-FOLD MISCLASSIFICATION SUMMARY\n")
                        f.write(f"{'='*100}\n")
                        f.write(f"Total samples: {len(y)}\n")
                        f.write(f"Total misclassified: {oof_misclass_count}\n")
                        f.write(f"Overall accuracy: {oof_accuracy:.5f}\n")
                        f.write(f"Overall macro F1: {oof_score:.5f}\n\n")
                        
                        f.write("Error Statistics by True Class:\n")
                        for true_label in range(num_labels):
                            true_mask = (y == true_label)
                            true_count = np.sum(true_mask)
                            if true_count > 0:
                                errors_for_true = np.sum((y != oof_pred_inference) & true_mask)
                                error_rate = errors_for_true / true_count
                                f.write(f"  Label {id2label[true_label]}: {true_count} samples, {errors_for_true} errors ({error_rate:.2%})\n")
                                
                                # Show which labels they were predicted as
                                pred_for_true = oof_pred_inference[true_mask]
                                for pred_label in range(num_labels):
                                    if pred_label != true_label:
                                        pred_count = np.sum(pred_for_true == pred_label)
                                        if pred_count > 0:
                                            f.write(f"      -> Predicted as {id2label[pred_label]}: {pred_count}\n")
                    
                    rank0_print(rank, f"\n[INFO] Misclassification summary saved to: {misclass_summary_path}")
                    rank0_print(rank, f"[INFO] Misclassification details saved to: {misclass_details_path}")
                    
                    with open(fold_validation_summary_path, "a") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Misclassification summary: {misclass_summary_path}\n")
                        f.write(f"Misclassification details: {misclass_details_path}\n")
                
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
                
                test_enc = make_segmented_encodings(tokenizer, test_clinical, test_findings, test_comparative, args.max_length)
                test_ds = SegmentedTextDataset(test_enc, labels=None)
                test_loader = DataLoader(
                    test_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
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
                    
                    test_model = TokenLevelSegmentedBert(
                        args.model_name,
                        num_labels=num_labels
                    )
                    ckpt_data = torch.load(latest_ckpt, map_location=device)
                    test_model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
                    test_model = test_model.to(device)
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
                            pred = torch.argmax(outputs, dim=-1)
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

                maybe_extract_attention_after_run(
                    args=args,
                    tokenizer=tokenizer,
                    train_df=train,
                    test_df=test,
                    num_labels=num_labels,
                    device=device,
                    rank=rank,
                )
            
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

        # Initialize fold validation summary file
        fold_validation_summary_path = os.path.join(args.ckpt_dir, "fold_validation_summary.txt")
        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(fold_validation_summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("FOLD VALIDATION RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_clinical, y, groups=groups), 1):
            # Skip folds not in training list
            if fold not in folds_to_train:
                rank0_print(rank, f"\nFold {fold}/{args.n_splits} - SKIPPED (not in --train_folds)")
                continue
            
            rank0_print(rank, f"\nFold {fold}/{args.n_splits}")
            
            # Check if checkpoint exists for this fold
            pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
            existing_ckpts = glob.glob(pattern)
            if existing_ckpts:
                rank0_print(rank, f"  [INFO] Found {len(existing_ckpts)} existing checkpoint(s) for fold {fold}. Will resume training from checkpoint.")
            else:
                rank0_print(rank, f"  [INFO] No checkpoint found for fold {fold}. Training from scratch.")

            tr_clinical = X_clinical[tr_idx].tolist()
            tr_findings = X_findings[tr_idx].tolist()
            tr_comparative = X_comparative[tr_idx].tolist()
            
            va_clinical = X_clinical[va_idx].tolist()
            va_findings = X_findings[va_idx].tolist()
            va_comparative = X_comparative[va_idx].tolist()

            y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
            y_va = torch.tensor(y[va_idx], dtype=torch.long)

            tr_enc = make_segmented_encodings(tokenizer, tr_clinical, tr_findings, tr_comparative, args.max_length)
            va_enc = make_segmented_encodings(tokenizer, va_clinical, va_findings, va_comparative, args.max_length)

            tr_ds = SegmentedTextDataset(tr_enc, y_tr)
            va_ds = SegmentedTextDataset(va_enc, y_va)

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
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            
            # Validation loader
            va_loader = DataLoader(
                va_ds,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=None,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

            model = TokenLevelSegmentedBert(
                args.model_name,
                num_labels=num_labels
            )
            
            # Try to load checkpoint BEFORE wrapping with DDP
            start_epoch_ckpt = 1
            best_f1_ckpt = -1.0
            if args.resume and args.ckpt_dir:
                pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
                existing_ckpts = glob.glob(pattern)
                if existing_ckpts:
                    latest_ckpt = max(existing_ckpts, key=os.path.getctime)
                    rank0_print(rank, f"  [INFO] Loading checkpoint BEFORE DDP wrap: {latest_ckpt}")
                    try:
                        ckpt_data = torch.load(latest_ckpt, map_location='cpu')
                        model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
                        start_epoch_ckpt = ckpt_data.get('epoch', 1) + 1
                        best_f1_ckpt = ckpt_data.get('best_f1', -1.0)
                        rank0_print(rank, f"  ✓ Checkpoint loaded: epoch {start_epoch_ckpt-1}, best_f1={best_f1_ckpt:.5f}")
                    except Exception as e:
                        rank0_print(rank, f"  ✗ Failed to load checkpoint: {e}")
            
            # NOW wrap with DDP
            if is_distributed:
                model = model.to(device)
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
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
                resume=False,  # Already loaded before DDP wrap
                use_amp=args.use_amp,
                grad_accum_steps=args.grad_accum_steps,
                val_interval=args.val_interval,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                val_dataset=va_ds,
                val_batch_size=args.batch_size,
                val_num_workers=0,
                use_focal_loss=args.use_focal_loss,
                focal_gamma=args.focal_gamma,
                start_epoch_override=start_epoch_ckpt,
                best_f1_override=best_f1_ckpt,
                early_stop_patience=3,
            )

            # Extract attention for every trained fold (regardless of previous folds)
            if args.extract_attention_after_train and rank == 0:
                preferred_ckpt = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_{best_epoch}.pt")
                if not os.path.exists(preferred_ckpt):
                    pattern = os.path.join(args.ckpt_dir, f"fold_{fold}_epoch_*.pt")
                    fold_ckpts = glob.glob(pattern)
                    if fold_ckpts:
                        preferred_ckpt = max(fold_ckpts, key=os.path.getctime)
                maybe_extract_attention_after_run(
                    args=args,
                    tokenizer=tokenizer,
                    train_df=train,
                    test_df=test,
                    num_labels=num_labels,
                    device=device,
                    rank=rank,
                    preferred_ckpt_path=preferred_ckpt if os.path.exists(preferred_ckpt) else None,
                    preferred_fold=fold,
                )

            val_f1, pred_va = distributed_evaluate(
                model=model,
                dataset=va_ds,
                device=device,
                batch_size=args.batch_size,
                num_workers=0,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
            )

            if rank == 0:
                oof_pred[va_idx] = pred_va
                
                # Output confusion matrix and classification report for this fold
                try:
                    cm = confusion_matrix(y[va_idx], pred_va)
                    report_str = classification_report(
                        y[va_idx], pred_va, 
                        target_names=[str(id2label[i]) for i in range(num_labels)]
                    )
                    
                    with open(fold_validation_summary_path, "a") as f:
                        f.write(f"\nFold {fold} Confusion Matrix:\n{cm}\n\n")
                        f.write(f"Fold {fold} Classification Report:\n{report_str}\n")
                        f.write("=" * 80 + "\n\n")
                except Exception as e:
                    rank0_print(rank, f"  [WARN] Could not save confusion matrix/report: {e}")
            
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
        
        test_enc = make_segmented_encodings(tokenizer, test_clinical, test_findings, test_comparative, args.max_length)
        test_ds = SegmentedTextDataset(test_enc, labels=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
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
                
                test_model = TokenLevelSegmentedBert(
                    args.model_name,
                    num_labels=num_labels
                )
                ckpt_data = torch.load(latest_ckpt, map_location=device)
                test_model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
                test_model = test_model.to(device)
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
                        pred = torch.argmax(outputs, dim=-1)
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
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--sub_path", type=str, default=SUB_PATH)

    parser.add_argument("--model_name", type=str, default="pucpr/biobertpt-all")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_data_parallel", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--train_folds",
        type=str,
        default="all",
        help="Which folds to train: 'all', 'none', or comma-separated fold numbers"
    )
    
    # Speedup options
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision FP16")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss")
    parser.add_argument("--focal_gamma", type=float, default=1.0, help="Focal loss gamma parameter")

    # Optional attention extraction and visualization
    parser.add_argument("--extract_attention_after_train", action="store_true", help="After train/inference, extract last-layer attention and save heatmap for each fold")
    parser.add_argument("--attention_source", type=str, default="train", choices=["train", "test"], help="Sample source for attention visualization")
    parser.add_argument("--attention_sample_index", type=int, default=0, help="Sample index in selected source dataframe")
    parser.add_argument("--attention_max_tokens", type=int, default=120, help="Maximum number of tokens to display on attention heatmap")
    parser.add_argument("--attention_merge_wordpieces", action=argparse.BooleanOptionalAction, default=True, help="Merge consecutive ## WordPiece tokens for readable labels")
    parser.add_argument("--attention_save_path", type=str, default="", help="Output image path for attention heatmap; default is ckpt_dir auto path")
    
    # Regenerate attention from txt
    parser.add_argument("--regenerate_attention_from_txt", type=str, default="", help="Path to txt file to regenerate PNG heatmap from")
    
    # Checkpoint options
    parser.add_argument("--ckpt_dir", type=str, default="spr2026/segmented_model_attention0", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint")

    args, _ = parser.parse_known_args()
    main(args)


#   torchrun --nnodes=1 --nproc_per_node=4 bert_segment_crossattention.py --train_folds "4,5"
# Inference only:
#   torchrun --nnodes=1 --nproc_per_node=2 bert_segment_crossattention.py --train_folds none
