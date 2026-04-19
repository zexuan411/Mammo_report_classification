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
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, BertTokenizer

from modules import set_seed, clean_text, clean_and_isolate_text, stable_hash, save_confusion_matrix_as_table, save_classification_report_as_table

RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_PATH = "./train_new.csv"
TEST_PATH = "./test.csv"
SUB_PATH = "./submission.csv"

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextReportDataset(Dataset):
    """Custom dataset for text reports."""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class AttentionPooling(nn.Module):
    """Attention pooling layer to aggregate LSTM outputs."""
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len) - attention mask
        
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        # Compute attention scores
        scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # Apply attention weights to LSTM output
        pooled_output = torch.sum(
            lstm_output * attention_weights.unsqueeze(-1),
            dim=1
        )  # (batch_size, hidden_size)
        
        return pooled_output


class LSTMAttentionClassifier(nn.Module):
    """LSTM + Attention Pooling classifier."""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMAttentionClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention pooling for bidirectional LSTM (2 * hidden_size)
        self.attention_pooling = AttentionPooling(2 * hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Embedding
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_output, (hidden, cell) = self.lstm(embeds)  # (batch_size, seq_len, 2*hidden_size)
        
        # Attention pooling
        pooled_output = self.attention_pooling(lstm_output, attention_mask)  # (batch_size, 2*hidden_size)
        
        # Classification
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)
        
        return logits


def compute_metrics(y_true, y_pred, id2label):
    """Compute and return classification metrics."""
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=[str(id2label[i]) for i in range(len(id2label))]
    )
    return f1, cm, report


def save_misclassified_samples(y_true, y_pred, X_text, train_ids, id2label, model_name, ckpt_dir):
    """Save misclassified samples to CSV file."""
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        print(f"[INFO] No misclassified samples for {model_name}")
        return
    
    misclassified_data = []
    for idx in misclassified_indices:
        misclassified_data.append({
            "ID": train_ids[idx] if idx < len(train_ids) else f"sample_{idx}",
            "text": X_text[idx],
            "true_label": id2label.get(y_true[idx], y_true[idx]),
            "predicted_label": id2label.get(y_pred[idx], y_pred[idx]),
            "true_label_id": y_true[idx],
            "predicted_label_id": y_pred[idx],
        })
    
    misclassified_df = pd.DataFrame(misclassified_data)
    output_path = os.path.join(ckpt_dir, f"misclassified_samples_{model_name}.csv")
    misclassified_df.to_csv(output_path, index=False)
    
    print(f"[INFO] {len(misclassified_indices)} misclassified samples saved to {output_path}")
    print(f"       Misclassification rate: {len(misclassified_indices)/len(y_true)*100:.2f}%")
    
    id_txt_path = os.path.join(ckpt_dir, f"misclassified_ids_{model_name}.txt")
    with open(id_txt_path, "w") as f:
        for idx in misclassified_indices:
            f.write(str(train_ids[idx] if idx < len(train_ids) else f"sample_{idx}") + "\n")
    print(f"[INFO] Misclassified IDs saved to {id_txt_path}")
    return misclassified_df


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = total_loss / len(val_loader)
    return val_loss, np.array(all_preds), np.array(all_labels)


def train_lstm_attention(
    X_train, y_train, X_val, y_val,
    fold, ckpt_dir, id2label,
    tokenizer, num_classes,
    embedding_dim=100, hidden_size=128, num_layers=2,
    batch_size=32, epochs=15, learning_rate=1e-3,
    max_length=256
):
    """Train LSTM + Attention model."""
    print(f"\n{'='*80}")
    print(f"[LSTM + Attention Pooling] Fold {fold}")
    print(f"{'='*80}")
    
    # Setup vocabulary size
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    
    # Create datasets
    train_dataset = TextReportDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = TextReportDataset(X_val, y_val, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMAttentionClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.3
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    best_f1 = -1.0
    best_model_state = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, DEVICE)
        val_f1, _, _ = compute_metrics(val_labels, val_preds, id2label)
        
        scheduler.step(val_f1)
        
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_f1={val_f1:.5f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    val_loss, y_pred, val_labels_check = evaluate(model, val_loader, criterion, DEVICE)
    val_f1, cm, report = compute_metrics(val_labels_check, y_pred, id2label)
    
    print(f"\nValidation F1 (macro): {val_f1:.5f}\n")
    print(f"Confusion Matrix:\n{cm}\n")
    print(f"Classification Report:\n{report}\n")
    
    # Save model
    model_path = os.path.join(ckpt_dir, f"fold_{fold}_lstm_attention_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
        }
    }, model_path)
    
    # Save confusion matrix and report as CSV
    fold_cm_csv_path = os.path.join(ckpt_dir, f"fold_{fold}_lstm_attention_confusion_matrix.csv")
    fold_report_csv_path = os.path.join(ckpt_dir, f"fold_{fold}_lstm_attention_classification_report.csv")
    save_confusion_matrix_as_table(cm, id2label, fold_cm_csv_path)
    save_classification_report_as_table(val_labels_check, y_pred, id2label, fold_report_csv_path)
    
    print(f"[INFO] Model saved: {model_path}")
    print(f"[INFO] Confusion matrix saved: {fold_cm_csv_path}")
    print(f"[INFO] Classification report saved: {fold_report_csv_path}")
    sys.stdout.flush()
    
    return model, val_f1, y_pred, cm, report


def main(args):
    set_seed(RANDOM_STATE)
    
    print("[INFO] Loading data...")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    
    assert {"ID", "report", "target"}.issubset(train.columns), train.columns
    assert {"ID", "report"}.issubset(test.columns), test.columns
    
    print("[INFO] Building text features...")
    train["text"] = train["report"].apply(clean_and_isolate_text)
    test["text"] = test["report"].apply(clean_and_isolate_text)
    train["group"] = train["text"].apply(stable_hash)
    
    labels_sorted = sorted(train["target"].astype(int).unique().tolist())
    label2id = {label: i for i, label in enumerate(labels_sorted)}
    id2label = {i: label for label, i in label2id.items()}
    
    y = train["target"].astype(int).map(label2id).values
    X_text = train["text"].values
    groups = train["group"].values
    train_ids = train["ID"].values
    
    num_labels = len(labels_sorted)
    
    print(f"[INFO] Dataset: {len(train)} samples, {num_labels} classes")
    print(f"[INFO] Class distribution: {np.bincount(y)}")
    
    # Load tokenizer
    print(f"[INFO] Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Initialize summary file
    summary_path = os.path.join(args.ckpt_dir, "lstm_attention_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LSTM + ATTENTION POOLING CLASSIFIER VALIDATION RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  Tokenizer: {args.tokenizer_name}\n")
        f.write(f"  Embedding Dim: {args.embedding_dim}\n")
        f.write(f"  Hidden Size: {args.hidden_size}\n")
        f.write(f"  LSTM Layers: {args.num_layers}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  Max Length: {args.max_length}\n")
        f.write("=" * 80 + "\n\n")
    
    # K-Fold Cross-Validation
    gkf = GroupKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    lstm_scores = []
    lstm_oof_pred = np.empty(len(train), dtype=int)
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_text, y, groups=groups), 1):
        print(f"\n{'#'*80}")
        print(f"# FOLD {fold}/{args.n_splits}")
        print(f"{'#'*80}")
        
        X_tr = X_text[tr_idx]
        X_va = X_text[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        
        print(f"Train: {len(X_tr)}, Val: {len(X_va)}")
        
        # Train LSTM + Attention
        lstm_model, lstm_f1, lstm_pred, lstm_cm, lstm_report = train_lstm_attention(
            X_tr, y_tr, X_va, y_va,
            fold, args.ckpt_dir, id2label,
            tokenizer, num_labels,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length
        )
        
        lstm_scores.append(lstm_f1)
        lstm_oof_pred[va_idx] = lstm_pred
        
        # Append fold results to summary
        with open(summary_path, "a") as f:
            f.write(f"Fold {fold} - LSTM + Attention Results:\n")
            f.write(f"  F1 Score (macro): {lstm_f1:.5f}\n")
            f.write(f"  Confusion Matrix:\n{str(lstm_cm).replace(chr(10), chr(10)+'    ')}\n")
            f.write(f"  Classification Report:\n{lstm_report}\n")
            f.write("-" * 80 + "\n\n")
    
    # ===== Cross-Validation Summary =====
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    lstm_cv_mean = float(np.mean(lstm_scores))
    lstm_cv_std = float(np.std(lstm_scores))
    lstm_oof_f1 = f1_score(y, lstm_oof_pred, average="macro")
    
    print(f"LSTM + Attention Pooling:")
    print(f"  CV mean: {lstm_cv_mean:.5f}, CV std: {lstm_cv_std:.5f}")
    print(f"  OOF F1: {lstm_oof_f1:.5f}")
    
    # OOF analysis
    try:
        lstm_cm_oof = confusion_matrix(y, lstm_oof_pred)
        lstm_report_oof = classification_report(
            y, lstm_oof_pred,
            target_names=[str(id2label[i]) for i in range(num_labels)]
        )
        
        oof_lstm_path = os.path.join(args.ckpt_dir, "oof_analysis_lstm_attention.txt")
        with open(oof_lstm_path, "w") as f:
            f.write(f"OOF F1 (macro): {lstm_oof_f1:.5f}\n")
            f.write(f"CV mean: {lstm_cv_mean:.5f}, CV std: {lstm_cv_std:.5f}\n\n")
            f.write(f"Confusion Matrix:\n{lstm_cm_oof}\n\n")
            f.write(f"Classification Report:\n{lstm_report_oof}\n")
        
        oof_lstm_cm_csv = os.path.join(args.ckpt_dir, "oof_confusion_matrix_lstm_attention.csv")
        oof_lstm_report_csv = os.path.join(args.ckpt_dir, "oof_classification_report_lstm_attention.csv")
        save_confusion_matrix_as_table(lstm_cm_oof, id2label, oof_lstm_cm_csv)
        save_classification_report_as_table(y, lstm_oof_pred, id2label, oof_lstm_report_csv)
        
        print(f"  OOF analysis saved to {oof_lstm_path}")
    except Exception as e:
        print(f"[ERROR] Failed to compute OOF metrics: {e}")
    
    # Append CV summary to main summary file
    with open(summary_path, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERALL CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"LSTM + Attention Pooling:\n")
        f.write(f"  CV mean F1 (macro): {lstm_cv_mean:.5f}\n")
        f.write(f"  CV std F1 (macro): {lstm_cv_std:.5f}\n")
        f.write(f"  OOF F1 (macro): {lstm_oof_f1:.5f}\n")
    
    print(f"\n[INFO] Summary saved to {summary_path}")
    print(f"\n[INFO] All results saved to {args.ckpt_dir}")
    
    # ===== Save Misclassified Samples =====
    print(f"\n{'='*80}")
    print("SAVING MISCLASSIFIED SAMPLES")
    print(f"{'='*80}\n")
    
    save_misclassified_samples(
        y, lstm_oof_pred, X_text, train_ids, id2label, "lstm_attention", args.ckpt_dir
    )
    
    print(f"\n[INFO] Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSTM + Attention Pooling classifier for report classification"
    )
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--sub_path", type=str, default=SUB_PATH)
    
    # Tokenizer
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    
    # Model hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    # Cross-validation
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    
    # Output
    parser.add_argument("--ckpt_dir", type=str, default="spr2026/lstm_attention_models")
    
    args, _ = parser.parse_known_args()
    
    main(args)
