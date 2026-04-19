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
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from modules import set_seed, clean_text, stable_hash, maybe_parallelize, save_checkpoint, load_checkpoint, find_latest_ckpt, evaluate, make_encodings, TextDataset, load_model_state_compat, save_confusion_matrix_as_table, save_classification_report_as_table

RANDOM_STATE = 42
N_SPLITS = 5

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


def export_attention_ready_model(model, tokenizer, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    model_to_save = unwrap_model(model)
    previous_output_attentions = getattr(model_to_save.config, "output_attentions", False)
    previous_attn_impl = getattr(model_to_save.config, "attn_implementation", None)

    try:
        # Some models forbid output_attentions=True when attn_implementation is sdpa.
        # First, change attn_implementation to eager BEFORE setting output_attentions
        if previous_attn_impl == "sdpa":
            model_to_save.config.attn_implementation = "eager"
        
        # Now try to set output_attentions
        try:
            model_to_save.config.output_attentions = True
        except ValueError as e:
            # If setting output_attentions fails, log and skip it
            if "output_attentions" in str(e):
                print(f"[WARN] Could not enable output_attentions: {e}")
                model_to_save.config.output_attentions = False
            else:
                raise
        
        model_to_save.save_pretrained(export_dir)
    finally:
        # Restore original config
        model_to_save.config.output_attentions = previous_output_attentions
        if previous_attn_impl is not None:
            model_to_save.config.attn_implementation = previous_attn_impl

    tokenizer.save_pretrained(export_dir)
    return export_dir


def parse_train_folds(train_folds_str, n_splits):
    """Parse train_folds string like '1,2,3' or '1-3' into a set of fold indices."""
    if not train_folds_str or train_folds_str.strip() == "":
        return set(range(1, n_splits + 1))
    
    folds = set()
    parts = train_folds_str.split(",")
    
    for part in parts:
        part = part.strip()
        if "-" in part:
            # Range like '1-3'
            start, end = part.split("-")
            start, end = int(start.strip()), int(end.strip())
            folds.update(range(start, end + 1))
        else:
            # Single fold like '2'
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


def maybe_add_report_special_tokens(tokenizer, enable=False):
    if not enable:
        return 0

    special_tokens_dict = {
        "additional_special_tokens": [
            "<indicação>",
            "<achados>",
            "<análise comparativa>",
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    return num_added_toks


def ordinal_loss(logits, labels, class_weights=None):
    """
    Ordinal loss for ordinal regression.
    Assumes class 0 < class 1 < class 2 < ...
    
    Uses K-1 binary classifiers where K is number of classes.
    For class k, the prediction should have k binary classifiers output 1.
    
    Args:
        logits: [batch_size, num_classes]
        labels: [batch_size] with values 0, 1, ..., num_classes-1
        class_weights: [num_classes] optional class weights
    
    Returns:
        scalar loss
    """
    logits = logits.float()
    num_classes = logits.size(1)
    batch_size = logits.size(0)
    
    # Create ordinal targets: for class k, target is [1, 1, ..., 1, 0, 0, ..., 0]
    # where there are k ones followed by (K-1-k) zeros
    ordinal_targets = torch.zeros(batch_size, num_classes - 1, device=logits.device, dtype=logits.dtype)
    for i in range(batch_size):
        label = labels[i].item()
        if label > 0:
            ordinal_targets[i, :label] = 1.0
    
    # Use first K-1 logits for ordinal prediction
    ordinal_logits = logits[:, :-1]
    
    # Binary cross entropy loss for each ordinal level
    bce_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets, reduction='none')
    
    # Apply class weights if provided (weight by the actual class label)
    if class_weights is not None:
        weights = class_weights[labels].to(logits.dtype)  # [batch_size]
        # Expand and apply: each sample's loss is weighted by its class weight
        bce_loss = bce_loss * weights.unsqueeze(1)  # [batch_size, K-1]
    
    return bce_loss.mean()

def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    lr,
    epochs,
    warmup_ratio,
    class_weights,
    loss_fn=None,
    fold=1,
    ckpt_dir=None,
    resume=True,
    use_amp=True,
    grad_accum_steps=1,
    val_interval=1,
    use_gradient_checkpointing=True,
    use_compile=False,
    use_ddp=False,
    run_validation=True,
    amp_dtype=torch.float16,
    use_grad_scaler=True,
):
    # Enable gradient checkpointing to reduce memory usage
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Optional: compile model for improved performance (requires PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"    Warning: Model compilation failed: {e}")
    
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
        if run_validation:
            print("    [INFO] Disabling GradScaler because model parameters are bfloat16")

    scaler = GradScaler('cuda', enabled=grad_scaler_enabled, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)
    best_f1 = -1.0
    best_epoch = 1
    best_state = None
    start_epoch = 1

    if resume and ckpt_dir:
        latest_ckpt = find_latest_ckpt(fold, ckpt_dir)
        if latest_ckpt:
            print(f"    Found checkpoint: {latest_ckpt}, resuming...")
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
                loss_logits = outputs.logits.float()
                if loss_fn is not None:
                    loss = loss_fn(loss_logits, labels, class_weights)
                else:
                    loss = F.cross_entropy(loss_logits, labels, weight=class_weights.float())
                loss = loss / grad_accum_steps

            if grad_scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulation_steps += 1
            running_loss += loss.item() * grad_accum_steps

            if accumulation_steps % grad_accum_steps == 0:
                if grad_scaler_enabled:
                    try:
                        scaler.unscale_(optimizer)
                    except RuntimeError as e:
                        if "BFloat16" in str(e) or "not implemented" in str(e):
                            pass  # Skip unscale for BFloat16
                        else:
                            raise
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if grad_scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        if run_validation and epoch % val_interval == 0:
            val_f1, _ = evaluate(
                model,
                val_loader,
                device,
                use_ordinal=(loss_fn is not None),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            avg_loss = running_loss / max(1, len(train_loader))
            print(f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} | val macro F1: {val_f1:.5f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                if ckpt_dir:
                    save_checkpoint(model, optimizer, scheduler, epoch, fold, best_f1, ckpt_dir)
        else:
            avg_loss = running_loss / max(1, len(train_loader))
            if run_validation:
                print(f"    Epoch {epoch}/{epochs} | loss: {avg_loss:.4f}")
            if ckpt_dir and run_validation:
                save_checkpoint(model, optimizer, scheduler, epoch, fold, best_f1, ckpt_dir)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, best_epoch

def train_full_model(
    model,
    train_loader,
    device,
    lr,
    epochs,
    warmup_ratio,
    class_weights,
    loss_fn=None,
    use_gradient_checkpointing=True,
    use_amp=True,
    amp_dtype=torch.float16,
    use_grad_scaler=True,
):
    # Enable gradge checkpointing
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
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
        print("    [INFO] Disabling GradScaler because model parameters are bfloat16")

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
                loss_logits = outputs.logits.float()
                if loss_fn is not None:
                    loss = loss_fn(loss_logits, labels, class_weights)
                else:
                    loss = F.cross_entropy(loss_logits, labels, weight=class_weights.float())

            optimizer.zero_grad()
            if grad_scaler_enabled:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        print("[WARN] --use_ddp is enabled but distributed env vars are missing (RANK/WORLD_SIZE/LOCAL_RANK).")
        print("[WARN] Falling back to single-process mode. Use torchrun to launch DDP.")
        use_ddp = False

    if use_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"[DDP] Rank {rank}/{world_size} - Local rank {local_rank} - Device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)

    main_process = is_main_process()
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    use_amp = args.use_amp and amp_dtype != torch.float32
    use_grad_scaler = use_amp and amp_dtype == torch.float16

    if main_process:
        print(f"[INFO] AMP enabled: {use_amp} | amp_dtype: {args.amp_dtype} | grad_scaler: {use_grad_scaler}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        # GPU optimizations
        torch.backends.cudnn.benchmark = True  # Use fastest CUDA algorithms
        torch.backends.cudnn.deterministic = False  # Trade off determinism for speed
        torch.cuda.empty_cache()  # Clear cache before starting
        
        # Print GPU info
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)

    assert {"ID", "report", "target"}.issubset(train.columns), train.columns
    assert {"ID", "report"}.issubset(test.columns), test.columns

    train["text"] = train["report"].apply(clean_text)
    test["text"] = test["report"].apply(clean_text)

    train["group"] = train["text"].apply(stable_hash)

    labels_sorted = sorted(train["target"].astype(int).unique().tolist())
    label2id = {label: i for i, label in enumerate(labels_sorted)}
    id2label = {i: label for label, i in label2id.items()}

    y = train["target"].astype(int).map(label2id).values
    X_text = train["text"].values
    groups = train["group"].values

    num_labels = len(labels_sorted)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    num_added_toks = maybe_add_report_special_tokens(
        tokenizer,
        enable=args.inject_special_tokens,
    )
    if main_process and args.inject_special_tokens:
        print(f"[INFO] Added {num_added_toks} special tokens to tokenizer")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model_save_root = args.model_save_dir or os.path.join(args.ckpt_dir, "attention_ready_model")

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
        print(f"[INFO] Found checkpoints for {len(ckpt_exists)}/{args.n_splits} folds. Using available checkpoints for submission...")
        
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
        
        all_logits = []
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
            
            print(f"  Fold {fold}: {os.path.basename(best_ckpt)} (F1={best_f1:.5f}, epoch={best_epoch})")
            
            # Load model and checkpoint
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=num_labels,
                ignore_mismatched_sizes=True).to(device)
            if num_added_toks > 0:
                model.resize_token_embeddings(len(tokenizer))
            ck = torch.load(best_ckpt, map_location=device)
            
            # ck["model_state_dict"]["classifier.weight"] = ck["model_state_dict"]["classifier.weight"][:2, :]
            # ck["model_state_dict"]["classifier.bias"] = ck["model_state_dict"]["classifier.bias"][:2]

            model.load_state_dict(ck["model_state_dict"], strict=False)
            model.eval()

            export_dir = os.path.join(model_save_root, f"fold_{fold}")
            export_attention_ready_model(model, tokenizer, export_dir)
            print(f"  Saved attention-ready model to {export_dir}")
            
            # Inference with GPU optimization
            fold_logits = []
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    for batch in test_loader:
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                        fold_logits.append(logits.cpu().numpy())
            
            all_logits.append(np.concatenate(fold_logits, axis=0))
            # Clear GPU cache to free up memory
            torch.cuda.empty_cache()
        
        # Average logits across folds
        mean_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
        pred_ids = np.argmax(mean_logits, axis=1)
        pred_test = [id2label[int(i)] for i in pred_ids]
        
        submission = pd.DataFrame({
            "ID": test["ID"],
            "target": pred_test,
        })

        submission.to_csv(args.sub_path, index=False)
        print("Saved:", args.sub_path)
        print(submission.head())
        return

    # ============ No checkpoints found, proceed with training ============
    if main_process:
        print(f"[INFO] No complete checkpoints found (found {len(ckpt_exists)}/{args.n_splits}). Starting training...")

    counts = np.bincount(y, minlength=num_labels)
    class_weights_np = len(y) / (num_labels * np.maximum(counts, 1))
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    # Select loss function
    loss_fn = None
    if args.loss_type == "ordinal":
        loss_fn = ordinal_loss
        if main_process:
            print(f"[INFO] Using ordinal loss function")
    else:
        if main_process:
            print(f"[INFO] Using cross entropy loss function")

    gkf = GroupKFold(n_splits=args.n_splits)
    oof_pred = np.empty(len(train), dtype=int)
    fold_scores = []
    best_epochs = []
    
    # Parse which folds to train
    try:
        train_folds = parse_train_folds(args.train_folds, args.n_splits)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if len(train_folds) < args.n_splits and main_process:
        print(f"[INFO] Training only folds: {train_folds}")

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_text, y, groups=groups), 1):
        if fold not in train_folds:
            if main_process:
                print(f"\nFold {fold}/{args.n_splits} - SKIPPED")
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
            )
            tr_loader = DataLoader(
                tr_ds,
                batch_size=args.batch_size,
                sampler=tr_sampler,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
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

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
        )
        if num_added_toks > 0:
            model.resize_token_embeddings(len(tokenizer))
        # When using DDP, don't wrap with DataParallel; DDP handles parallelization
        model = maybe_parallelize(model, device, use_data_parallel=(not use_ddp and not args.no_data_parallel))

        if main_process:
            model_param_dtype = get_first_trainable_param_dtype(model)
            print(f"[INFO] Fold {fold} model param dtype: {model_param_dtype}")

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
            use_gradient_checkpointing=args.gradient_checkpointing,
            use_compile=args.use_compile,
            use_ddp=use_ddp,
            run_validation=main_process,
            amp_dtype=amp_dtype,
            use_grad_scaler=use_grad_scaler,
        )

        if use_ddp:
            dist.barrier()

        if main_process:
            export_dir = os.path.join(model_save_root, f"fold_{fold}")
            export_attention_ready_model(model, tokenizer, export_dir)
            print(f"Saved attention-ready model to {export_dir}")

            val_f1, pred_va = evaluate(
                model,
                va_loader,
                device,
                use_ordinal=(loss_fn is not None),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            oof_pred[va_idx] = pred_va
            fold_scores.append(val_f1)
            best_epochs.append(best_epoch)

            print(
                f"Fold {fold} best macro F1: {best_f1:.5f} @ epoch {best_epoch} "
                f"| final eval macro F1: {val_f1:.5f}"
            )

        if use_ddp:
            dist.barrier()

    if not main_process:
        return

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    oof_score = f1_score(y, oof_pred, average="macro")
    auto_final_epochs = max(1, int(round(float(np.mean(best_epochs)))))

    print("\nCV mean:", f"{cv_mean:.5f}", "| CV std:", f"{cv_std:.5f}")
    print("OOF macro F1:", f"{oof_score:.5f}")
    print("Best epochs per fold:", best_epochs)
    print("Suggested final_epochs (mean rounded):", auto_final_epochs)

    final_epochs = args.final_epochs if args.final_epochs > 0 else auto_final_epochs
    print("Use final_epochs:", final_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT baseline for SPR2026 report classification (FAST + Checkpoints)")
    # parser.add_argument("--train_path", type=str, default="/kaggle/input/competitions/spr-2026-mammography-report-classification/train.csv")
    parser.add_argument("--train_path", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/train_tokenizer.csv")
    parser.add_argument("--test_path", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/test.csv")
    parser.add_argument("--sub_path", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/submission.csv")
    parser.add_argument("--save_path", type=str, default="submission.csv")
    parser.add_argument("--model_name", type=str, default="pucpr/biobertpt-all")
    # parser.add_argument("--model_name", type=str, default="PORTULAN/albertina-900m-portuguese-ptpt-encoder")
    parser.add_argument(
        "--inject_special_tokens",
        action="store_true",
        default=False,
        help="Inject report section tokens (<indicação>, <achados>, <análise comparativa>) into tokenizer",
    )
    # parser.add_argument("--model_dir", type=str, default="/kaggle/input/models/cccccc411/neuralmindbert-base-portuguese-cased/transformers/default/1/bert-base-portuguese-cased")
    # parser.add_argument("--model_dir", type=str, default="/kaggle/input/models/cccccc411/pucprbiobertpt-all/transformers/default/1/biobertpt-all")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_data_parallel", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=0,
        help="Final training epochs. <=0 means auto from CV best epochs.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    
    # Loss function options
    parser.add_argument("--loss_type", type=str, default="cross_entropy", choices=["cross_entropy", "ordinal"],
                        help="Loss function type: 'cross_entropy' or 'ordinal'")
    
    # Speedup options
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use mixed precision training for faster throughput on Tensor Core GPUs")
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Precision type for mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory")
    parser.add_argument("--use_compile", action="store_true", default=False, help="Use torch.compile for optimization (PyTorch 2.0+)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    
    # Checkpoint options
    parser.add_argument("--ckpt_dir", type=str, default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/spr2026/checkpoint-tokenizer", help="Directory to save checkpoints")
    parser.add_argument("--model_save_dir", type=str, default="", help="Directory to save attention-ready Hugging Face models. Empty means use <ckpt_dir>/attention_ready_model.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint")
    parser.add_argument("--no_resume", action="store_false", dest="resume", help="Do not resume from checkpoint; start fresh training")
    parser.add_argument("--train_folds", type=str, default="", help="Specify which folds to train (e.g., '1,2,3' or '1-3'). Empty means train all folds.")
    
    parser.add_argument("--use_ddp", action="store_true", default=True, help="Use DistributedDataParallel (DDP) for multi-GPU training (recommended for 2+ GPUs)")
    parser.add_argument("--gpu_ids", type=str, default="", help="GPU IDs to use (e.g., '0,1,2' or '0-3'). Empty means use all GPUs.")

    # args, _ = parser.parse_known_args()
    args = parser.parse_args()
    try:
        main(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

# torchrun --nproc_per_node=2 bert_inject_tokens.py --use_ddp --use_amp --inject_special_tokens