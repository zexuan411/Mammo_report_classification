import torch.optim as optim
from sentence_transformers import SentenceTransformer, losses, InputExample
import os
import re
import argparse
import hashlib
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_PATH = "./train_new.csv"
TEST_PATH = "./test.csv"
SUB_PATH = "./rag403"

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

def resolve_artifact_dir(sub_path: str) -> str:
    if os.path.isdir(sub_path):
        return sub_path
    _, ext = os.path.splitext(sub_path)
    if ext:
        return os.path.dirname(sub_path) or "."
    return sub_path

@dataclass
class RetrievalResult:
    indices: np.ndarray
    scores: np.ndarray

class SimpleEncoder:
    def __init__(self, model_name: str, device: torch.device, batch_size: int = 64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
        try:
            self._st_model = SentenceTransformer(model_name, device=str(device))
            self._use_sentence_transformers = True
        except Exception:
            self._st_model = None
            self._use_sentence_transformers = False
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._hf_model = AutoModel.from_pretrained(model_name).to(device)
        self._hf_model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if self._use_sentence_transformers and self._st_model is not None:
            embs = self._st_model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=True,
            )
            return embs.astype(np.float32)
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            toks = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self._hf_model(**toks)
            token_embs = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).float()
            pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

class Retriever:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self._faiss = None
        self._index = None
        self._base_embs = None
        try:
            import faiss
            self._faiss = faiss
        except Exception:
            self._faiss = None
    def fit(self, base_embs: np.ndarray) -> None:
        if self.normalize:
            norms = np.linalg.norm(base_embs, axis=1, keepdims=True) + 1e-12
            base_embs = base_embs / norms
        self._base_embs = base_embs.astype(np.float32)
        if self._faiss is not None:
            dim = self._base_embs.shape[1]
            index = self._faiss.IndexFlatIP(dim)
            index.add(self._base_embs)
            self._index = index
    def search(self, query_embs: np.ndarray, top_k: int) -> RetrievalResult:
        if self.normalize:
            norms = np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-12
            query_embs = query_embs / norms
        query_embs = query_embs.astype(np.float32)
        k = min(top_k, self._base_embs.shape[0])
        if k <= 0:
            raise ValueError("top_k must be >= 1 and base embeddings must be non-empty")
        if self._index is not None:
            scores, indices = self._index.search(query_embs, k)
            return RetrievalResult(indices=indices, scores=scores)
        q = torch.from_numpy(query_embs)
        b = torch.from_numpy(self._base_embs)
        sim = q @ b.T
        scores, indices = torch.topk(sim, k=k, dim=1)
        return RetrievalResult(indices=indices.numpy(), scores=scores.numpy())

def weighted_vote(
    retrieved_indices: np.ndarray,
    retrieved_scores: np.ndarray,
    base_labels: np.ndarray,
    num_labels: int,
    temperature: float,
    min_sim_clip: float,
) -> np.ndarray:
    probs = []
    for idxs, sims in zip(retrieved_indices, retrieved_scores):
        sims = np.maximum(sims, min_sim_clip)
        w = np.exp(sims / max(temperature, 1e-6))
        p = np.zeros(num_labels, dtype=np.float32)
        for i, weight in zip(idxs, w):
            p[int(base_labels[i])] += float(weight)
        if p.sum() <= 0:
            p += 1.0 / num_labels
        else:
            p /= p.sum()
        probs.append(p)
    return np.stack(probs, axis=0)

def setup_distributed():
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
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

def rank0_print(rank: int, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

def main(args):
    rank, local_rank, world_size, is_distributed = setup_distributed()
    try:
        set_seed(RANDOM_STATE + rank)
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        rank0_print(rank, "Device:", device, "| world_size:", world_size)

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
        X_text = train["text"].tolist()
        groups = train["group"].values
        num_labels = len(labels_sorted)
        gkf = GroupKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)

        fold_scores = []
        oof_pred = np.zeros(len(train), dtype=np.int64)
        val_fold_preds = np.zeros((len(train), num_labels), dtype=np.float32)
        test_fold_preds = []
        ckpt_root = resolve_artifact_dir(args.sub_path)
        if rank == 0:
            os.makedirs(ckpt_root, exist_ok=True)
        if is_distributed:
            dist.barrier()

        skip_folds = set()
        if args.skip_folds:
            skip_folds = set(int(f.strip()) for f in args.skip_folds.split(",") if f.strip())
        
        rank0_print(rank, "[INFO] Start cross-validation with DDP fold training...")
        if skip_folds:
            rank0_print(rank, f"[INFO] Skipping folds: {sorted(skip_folds)}")
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_text, y, groups=groups), 1):
            rank0_print(rank, f"\nFold {fold}/{args.n_splits}")
            X_tr = [X_text[i] for i in tr_idx]
            y_tr = y[tr_idx]
            
            fold_ckpt_dir = os.path.join(ckpt_root, f"rag_encoder_fold{fold}.pt")
            has_fold_ckpt = os.path.isdir(fold_ckpt_dir) and os.path.exists(os.path.join(fold_ckpt_dir, "modules.json"))
            
            # Check for any checkpoint (best > last > fold)
            best_ckpt_dir = os.path.join(ckpt_root, f"rag_encoder_fold{fold}_best.pt")
            last_ckpt_dir = os.path.join(ckpt_root, f"rag_encoder_fold{fold}_last.pt")
            has_best_ckpt = os.path.isdir(best_ckpt_dir) and os.path.exists(os.path.join(best_ckpt_dir, "modules.json"))
            has_last_ckpt = os.path.isdir(last_ckpt_dir) and os.path.exists(os.path.join(last_ckpt_dir, "modules.json"))
            
            checkpoint_exists = has_best_ckpt or has_last_ckpt or has_fold_ckpt
            
            # Determine if we should skip training based on checkpoint and mode
            should_skip_training_due_to_ckpt = checkpoint_exists and args.ckpt_mode == "use"
            skip_fold_training = fold in skip_folds or should_skip_training_due_to_ckpt
            
            if skip_fold_training and not checkpoint_exists:
                rank0_print(rank, f"  [WARN] Fold {fold} skipped training and no checkpoint found. Skipping fold entirely.")
                if is_distributed:
                    dist.barrier()
                continue
            
            if should_skip_training_due_to_ckpt:
                rank0_print(rank, f"  [INFO] Checkpoint exists for fold {fold} (mode={args.ckpt_mode}). Skipping training and loading for evaluation...")
            elif checkpoint_exists and args.ckpt_mode == "resume":
                rank0_print(rank, f"  [INFO] Checkpoint exists for fold {fold} (mode={args.ckpt_mode}). Will resume training from best checkpoint...")
            
            if not skip_fold_training:
                train_examples = []
                for i in range(len(X_tr)):
                    anchor = X_tr[i]
                    pos_idx = np.where(y_tr == y_tr[i])[0]
                    if len(pos_idx) > 1:
                        pos = X_tr[random.choice([j for j in pos_idx if j != i])]
                        train_examples.append(InputExample(texts=[anchor, pos]))

                # Prioritize checkpoint loading: best.pt > last.pt > fold.pt > args.embed_model
                model_source = args.embed_model
                checkpoint_for_resume = None
                if has_best_ckpt:
                    model_source = best_ckpt_dir
                    checkpoint_for_resume = best_ckpt_dir
                    if args.ckpt_mode == "resume":
                        rank0_print(rank, f"  [INFO] Resuming from best checkpoint: {best_ckpt_dir}")
                    else:
                        rank0_print(rank, f"  [INFO] Loading best checkpoint for continued training: {best_ckpt_dir}")
                elif has_last_ckpt:
                    model_source = last_ckpt_dir
                    rank0_print(rank, f"  [INFO] Loading last checkpoint for continued training: {last_ckpt_dir}")
                elif has_fold_ckpt:
                    model_source = fold_ckpt_dir
                    rank0_print(rank, f"  [INFO] Loading fold checkpoint for continued training: {fold_ckpt_dir}")
                else:
                    rank0_print(rank, f"  [INFO] No checkpoint found, initializing from base model: {args.embed_model}")

                st_model = SentenceTransformer(model_source, device=str(device))
                train_sampler = None
                if is_distributed:
                    train_sampler = DistributedSampler(
                        train_examples,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=True,
                        seed=RANDOM_STATE + fold,
                    )
                train_dataloader = DataLoader(
                    train_examples,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    batch_size=args.embed_batch_size,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    drop_last=is_distributed,
                )
                train_loss = losses.MultipleNegativesRankingLoss(st_model)
                st_model.train()
                
                best_f1 = -1.0
                last_ckpt_dir = os.path.join(ckpt_root, f"rag_encoder_fold{fold}_last.pt")
                best_ckpt_dir = os.path.join(ckpt_root, f"rag_encoder_fold{fold}_best.pt")
                
                for epoch in range(args.train_epochs):
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)
                    st_model.fit(
                        train_objectives=[(train_dataloader, train_loss)],
                        epochs=1,
                        warmup_steps=0,
                        show_progress_bar=(rank == 0),
                        output_path=None,
                    )
                    
                    if rank == 0:
                        st_model.save(last_ckpt_dir)
                    if is_distributed:
                        dist.barrier()
                    
                    if rank == 0:
                        eval_model = SentenceTransformer(last_ckpt_dir, device=str(device))
                        eval_model.eval()
                        train_embs = eval_model.encode(X_tr, batch_size=args.embed_batch_size, convert_to_numpy=True, normalize_embeddings=True)
                        val_embs = eval_model.encode([X_text[i] for i in va_idx], batch_size=args.embed_batch_size, convert_to_numpy=True, normalize_embeddings=True)
                        
                        retriever = Retriever(normalize=True)
                        retriever.fit(train_embs)
                        rr = retriever.search(val_embs, top_k=args.top_k)
                        rag_proba = weighted_vote(
                            retrieved_indices=rr.indices,
                            retrieved_scores=rr.scores,
                            base_labels=y_tr,
                            num_labels=num_labels,
                            temperature=args.temperature,
                            min_sim_clip=args.min_sim_clip,
                        )
                        pred = np.argmax(rag_proba, axis=1)
                        f1 = f1_score(y[va_idx], pred, average="macro")
                        rank0_print(rank, f"  Epoch {epoch+1}/{args.train_epochs} - val F1: {f1:.5f}")
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            eval_model.save(best_ckpt_dir)
                            rank0_print(rank, f"  [BEST] F1 improved to {f1:.5f}, saved best model.")
                
                if is_distributed:
                    dist.barrier()

            if rank == 0:
                best_ckpt_dir = os.path.join(ckpt_root, f"rag_encoder_fold{fold}_best.pt")
                last_ckpt_dir = os.path.join(ckpt_root, f"rag_encoder_fold{fold}_last.pt")
                
                final_ckpt_to_use = None
                if os.path.isdir(best_ckpt_dir):
                    final_ckpt_to_use = best_ckpt_dir
                    rank0_print(rank, f"  [INFO] Using best checkpoint for evaluation: {best_ckpt_dir}")
                elif os.path.isdir(last_ckpt_dir):
                    final_ckpt_to_use = last_ckpt_dir
                    rank0_print(rank, f"  [INFO] Using last checkpoint for evaluation: {last_ckpt_dir}")
                elif os.path.isdir(fold_ckpt_dir):
                    final_ckpt_to_use = fold_ckpt_dir
                    rank0_print(rank, f"  [INFO] Using fold checkpoint for evaluation: {fold_ckpt_dir}")
                else:
                    raise RuntimeError(f"No checkpoint found for fold {fold}")
                
                eval_model = SentenceTransformer(final_ckpt_to_use, device=str(device))
                eval_model.eval()
                train_embs = eval_model.encode(X_tr, batch_size=args.embed_batch_size, convert_to_numpy=True, normalize_embeddings=True)
                val_embs = eval_model.encode([X_text[i] for i in va_idx], batch_size=args.embed_batch_size, convert_to_numpy=True, normalize_embeddings=True)

                retriever = Retriever(normalize=True)
                retriever.fit(train_embs)
                rr = retriever.search(val_embs, top_k=args.top_k)
                rag_proba = weighted_vote(
                    retrieved_indices=rr.indices,
                    retrieved_scores=rr.scores,
                    base_labels=y_tr,
                    num_labels=num_labels,
                    temperature=args.temperature,
                    min_sim_clip=args.min_sim_clip,
                )
                pred = np.argmax(rag_proba, axis=1)
                oof_pred[va_idx] = pred
                val_fold_preds[va_idx] = rag_proba
                f1 = f1_score(y[va_idx], pred, average="macro")
                fold_scores.append(float(f1))
                print(f"  val macro F1: {f1:.5f}")
                cm = confusion_matrix(y[va_idx], pred, labels=list(range(num_labels)))
                cm_df = pd.DataFrame(cm, index=[f"true_{id2label[i]}" for i in range(num_labels)], columns=[f"pred_{id2label[i]}" for i in range(num_labels)])
                print("  Confusion matrix (fold):\n", cm_df)
                
                # Save evaluation metrics for this fold
                fold_metrics_path = os.path.join(ckpt_root, f"fold_{fold}_metrics.txt")
                with open(fold_metrics_path, "w") as f:
                    f.write(f"Fold {fold} Validation Metrics\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Macro F1 Score: {f1:.5f}\n")
                    f.write(f"Checkpoint Used: {final_ckpt_to_use}\n")
                    f.write(f"\nConfusion Matrix:\n")
                    f.write(cm_df.to_string())
                    f.write(f"\n\nDetailed Metrics:\n")
                    report = classification_report(
                        y[va_idx], pred,
                        target_names=[str(id2label[i]) for i in range(num_labels)],
                        digits=5
                    )
                    f.write(report)
                rank0_print(rank, f"  [INFO] Saved fold {fold} metrics to: {fold_metrics_path}")

                test_embs = eval_model.encode(test["text"].tolist(), batch_size=args.embed_batch_size, convert_to_numpy=True, normalize_embeddings=True)
                retriever_test = Retriever(normalize=True)
                retriever_test.fit(train_embs)
                rr_test = retriever_test.search(test_embs, top_k=args.top_k)
                rag_proba_test = weighted_vote(
                    retrieved_indices=rr_test.indices,
                    retrieved_scores=rr_test.scores,
                    base_labels=y_tr,
                    num_labels=num_labels,
                    temperature=args.temperature,
                    min_sim_clip=args.min_sim_clip,
                )
                test_fold_preds.append(rag_proba_test)

            if is_distributed:
                dist.barrier()

        if rank == 0:
            cv_mean = float(np.mean(fold_scores)) if fold_scores else 0.0
            cv_std = float(np.std(fold_scores)) if fold_scores else 0.0
            oof_f1 = f1_score(y, oof_pred, average="macro")
            print("\nCV mean:", f"{cv_mean:.5f}", "| CV std:", f"{cv_std:.5f}")
            print("OOF macro F1:", f"{oof_f1:.5f}")
            cm = confusion_matrix(y, oof_pred, labels=list(range(num_labels)))
            cm_df = pd.DataFrame(cm, index=[f"true_{id2label[i]}" for i in range(num_labels)], columns=[f"pred_{id2label[i]}" for i in range(num_labels)])
            print("OOF Confusion matrix:\n", cm_df)
            
            # Save OOF metrics summary
            oof_metrics_path = os.path.join(ckpt_root, "oof_metrics_summary.txt")
            with open(oof_metrics_path, "w") as f:
                f.write("Out-of-Fold (OOF) Validation Metrics Summary\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"CV Mean F1 Score: {cv_mean:.5f}\n")
                f.write(f"CV Std F1 Score: {cv_std:.5f}\n")
                f.write(f"OOF Macro F1 Score: {oof_f1:.5f}\n")
                f.write(f"\nFold Scores: {fold_scores}\n")
                f.write(f"\nConfusion Matrix:\n")
                f.write(cm_df.to_string())
                f.write(f"\n\nDetailed Metrics:\n")
                report = classification_report(
                    y, oof_pred,
                    target_names=[str(id2label[i]) for i in range(num_labels)],
                    digits=5
                )
                f.write(report)
            rank0_print(rank, f"[INFO] Saved OOF metrics summary to: {oof_metrics_path}")

            print("\n[INFO] Ensemble test predictions from all folds...")
            test_fold_preds = np.stack(test_fold_preds, axis=0)
            test_proba_mean = np.mean(test_fold_preds, axis=0)
            pred_ids = np.argmax(test_proba_mean, axis=1)
            pred_labels = [id2label[int(i)] for i in pred_ids]
            submission = pd.DataFrame({"ID": test["ID"], "target": pred_labels})
            print(submission.head())
    finally:
        cleanup_distributed(is_distributed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-only retriever classifier for SPR2026")
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--sub_path", type=str, default=SUB_PATH)
    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-m3", help="Embedding model name or path")
    parser.add_argument("--embed_batch_size", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--skip_folds", type=str, default="", help="Comma-separated fold numbers to skip (e.g., '1,3')")
    parser.add_argument("--ckpt_mode", type=str, default="use", choices=["use", "resume"],
                        help="How to handle existing checkpoints: 'use' (default) = skip training and use checkpoint directly, "
                             "'resume' = resume training from best checkpoint")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--min_sim_clip", type=float, default=0.0)
    args, _ = parser.parse_known_args()
    main(args)

# torchrun --nnodes=1 --nproc_per_node=4 /proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/rag_only_classifier.py --skip_folds "1,2,3,4"
# 
# Examples:
# 1. Use existing checkpoints directly (skip training):
#    torchrun --nnodes=1 --nproc_per_node=4 rag_only_classifier.py --ckpt_mode use
#
# 2. Resume training from best checkpoint:
#    torchrun --nnodes=1 --nproc_per_node=4 rag_only_classifier.py --ckpt_mode resume
#
# 3. Skip specific folds and use checkpoints for others:
#    torchrun --nnodes=1 --nproc_per_node=4 rag_only_classifier.py --skip_folds "2,4" --ckpt_mode use