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
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import lightgbm as lgb

from modules import set_seed, clean_text, clean_and_isolate_text, stable_hash, save_confusion_matrix_as_table, save_classification_report_as_table

RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_PATH = "./train_new.csv"
TEST_PATH = "./test.csv"
SUB_PATH = "./submission.csv"


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
    # Find misclassified indices
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        print(f"[INFO] No misclassified samples for {model_name}")
        return
    
    # Create dataframe with misclassified samples
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
    
    # Save to CSV
    output_path = os.path.join(ckpt_dir, f"misclassified_samples_{model_name}.csv")
    misclassified_df.to_csv(output_path, index=False)
    
    print(f"[INFO] {len(misclassified_indices)} misclassified samples saved to {output_path}")
    print(f"       Misclassification rate: {len(misclassified_indices)/len(y_true)*100:.2f}%")
    
    # 额外保存错分ID到txt
    id_txt_path = os.path.join(ckpt_dir, f"misclassified_ids_{model_name}.txt")
    with open(id_txt_path, "w") as f:
        for idx in misclassified_indices:
            f.write(str(train_ids[idx] if idx < len(train_ids) else f"sample_{idx}") + "\n")
    print(f"[INFO] Misclassified IDs saved to {id_txt_path}")
    return misclassified_df


def train_tfidf_svc(
    X_train, y_train, X_val, y_val,
    fold, ckpt_dir, id2label,
    max_features=5000, ngram_range=(1, 2), C=1.0
):
    """Train TF-IDF + LinearSVC model."""
    print(f"\n{'='*80}")
    print(f"[TF-IDF + LinearSVC] Fold {fold}")
    print(f"{'='*80}")
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    
    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    
    # Train LinearSVC
    svc = LinearSVC(
        C=C,
        max_iter=2000,
        random_state=RANDOM_STATE + fold,
        verbose=0
    )
    
    svc.fit(X_train_tfidf, y_train)
    
    # Evaluate on validation set
    y_pred = svc.predict(X_val_tfidf)
    val_f1, cm, report = compute_metrics(y_val, y_pred, id2label)
    
    print(f"Validation F1 (macro): {val_f1:.5f}\n")
    print(f"Confusion Matrix:\n{cm}\n")
    print(f"Classification Report:\n{report}\n")
    
    # Save models and results
    model_path = os.path.join(ckpt_dir, f"fold_{fold}_tfidf_svc_model.pkl")
    tfidf_path = os.path.join(ckpt_dir, f"fold_{fold}_tfidf_vectorizer.pkl")
    
    joblib.dump(svc, model_path)
    joblib.dump(tfidf, tfidf_path)
    
    # Save evaluation results
    results = {
        "fold": fold,
        "method": "TF-IDF + LinearSVC",
        "val_f1": val_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    
    # Save to text file
    # fold_analysis_path = os.path.join(ckpt_dir, f"fold_{fold}_tfidf_svc_analysis.txt")
    # with open(fold_analysis_path, "w") as f:
    #     f.write(f"Fold {fold} - TF-IDF + LinearSVC Validation Analysis\n")
    #     f.write(f"{'='*80}\n")
    #     f.write(f"F1 Score (macro): {val_f1:.5f}\n\n")
    #     f.write(f"Confusion Matrix:\n{cm}\n\n")
    #     f.write(f"Classification Report:\n{report}\n")
    
    # Save confusion matrix and report as CSV
    fold_cm_csv_path = os.path.join(ckpt_dir, f"fold_{fold}_tfidf_svc_confusion_matrix.csv")
    fold_report_csv_path = os.path.join(ckpt_dir, f"fold_{fold}_tfidf_svc_classification_report.csv")
    save_confusion_matrix_as_table(cm, id2label, fold_cm_csv_path)
    save_classification_report_as_table(y_val, y_pred, id2label, fold_report_csv_path)
    
    print(f"[INFO] Models saved: {model_path}, {tfidf_path}")
    # print(f"[INFO] Results saved: {fold_analysis_path}")
    print(f"[INFO] Confusion matrix saved: {fold_cm_csv_path}")
    print(f"[INFO] Classification report saved: {fold_report_csv_path}")
    sys.stdout.flush()
    
    return (svc, tfidf), val_f1, y_pred, cm, report


def train_tfidf_lgb(
    X_train, y_train, X_val, y_val,
    fold, ckpt_dir, id2label,
    max_features=5000, ngram_range=(1, 2), num_leaves=31, learning_rate=0.1
):
    """Train TF-IDF + LightGBM model."""
    print(f"\n{'='*80}")
    print(f"[TF-IDF + LightGBM] Fold {fold}")
    print(f"{'='*80}")
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    
    # Convert sparse matrix to dense if needed for LightGBM
    if hasattr(X_train_tfidf, 'toarray'):
        X_train_tfidf = X_train_tfidf.toarray()
    if hasattr(X_val_tfidf, 'toarray'):
        X_val_tfidf = X_val_tfidf.toarray()
    
    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    
    # Train LightGBM
    train_data = lgb.Dataset(
        X_train_tfidf,
        label=y_train,
        free_raw_data=True
    )
    
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'metric': 'multi_logloss',
        'verbose': -1,
        'random_state': RANDOM_STATE + fold,
        'num_threads': -1,
    }
    
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10, verbose=0)]
    )
    
    # Evaluate on validation set
    y_pred_proba = lgb_model.predict(X_val_tfidf)
    y_pred = np.argmax(y_pred_proba, axis=1)
    val_f1, cm, report = compute_metrics(y_val, y_pred, id2label)
    
    print(f"Validation F1 (macro): {val_f1:.5f}\n")
    print(f"Confusion Matrix:\n{cm}\n")
    print(f"Classification Report:\n{report}\n")
    
    # Save models and results
    model_path = os.path.join(ckpt_dir, f"fold_{fold}_lgb_model.pkl")
    tfidf_path = os.path.join(ckpt_dir, f"fold_{fold}_lgb_tfidf_vectorizer.pkl")
    
    joblib.dump(lgb_model, model_path)
    joblib.dump(tfidf, tfidf_path)
    
    # Save evaluation results
    results = {
        "fold": fold,
        "method": "TF-IDF + LightGBM",
        "val_f1": val_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    
    # Save to text file
    # fold_analysis_path = os.path.join(ckpt_dir, f"fold_{fold}_lgb_analysis.txt")
    # with open(fold_analysis_path, "w") as f:
    #     f.write(f"Fold {fold} - TF-IDF + LightGBM Validation Analysis\n")
    #     f.write(f"{'='*80}\n")
    #     f.write(f"F1 Score (macro): {val_f1:.5f}\n\n")
    #     f.write(f"Confusion Matrix:\n{cm}\n\n")
    #     f.write(f"Classification Report:\n{report}\n")
    
    # Save confusion matrix and report as CSV
    fold_cm_csv_path = os.path.join(ckpt_dir, f"fold_{fold}_lgb_confusion_matrix.csv")
    fold_report_csv_path = os.path.join(ckpt_dir, f"fold_{fold}_lgb_classification_report.csv")
    save_confusion_matrix_as_table(cm, id2label, fold_cm_csv_path)
    save_classification_report_as_table(y_val, y_pred, id2label, fold_report_csv_path)
    
    print(f"[INFO] Models saved: {model_path}, {tfidf_path}")
    # print(f"[INFO] Results saved: {fold_analysis_path}")
    print(f"[INFO] Confusion matrix saved: {fold_cm_csv_path}")
    print(f"[INFO] Classification report saved: {fold_report_csv_path}")
    sys.stdout.flush()
    
    return (lgb_model, tfidf), val_f1, y_pred, cm, report


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
    
    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Initialize summary file
    summary_path = os.path.join(args.ckpt_dir, "baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BASELINE MODELS VALIDATION RESULTS SUMMARY\n")
        f.write("(TF-IDF + LinearSVC, TF-IDF + LightGBM)\n")
        f.write("=" * 80 + "\n\n")
    
    # K-Fold Cross-Validation
    gkf = GroupKFold(n_splits=args.n_splits)
    
    svc_scores = []
    lgb_scores = []
    
    svc_oof_pred = np.empty(len(train), dtype=int)
    lgb_oof_pred = np.empty(len(train), dtype=int)
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_text, y, groups=groups), 1):
        print(f"\n{'#'*80}")
        print(f"# FOLD {fold}/{args.n_splits}")
        print(f"{'#'*80}")
        
        X_tr = X_text[tr_idx]
        X_va = X_text[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        
        print(f"Train: {len(X_tr)}, Val: {len(X_va)}")
        
        # ===== TF-IDF + LinearSVC =====
        if not args.no_train_svc:
            svc_models, svc_f1, svc_pred, svc_cm, svc_report = train_tfidf_svc(
                X_tr, y_tr, X_va, y_va,
                fold, args.ckpt_dir, id2label,
                max_features=args.tfidf_max_features,
                C=args.svc_c
            )
            svc_scores.append(svc_f1)
            svc_oof_pred[va_idx] = svc_pred
            
            # Append fold results to summary
            with open(summary_path, "a") as f:
                f.write(f"Fold {fold} - TF-IDF + LinearSVC Results:\n")
                f.write(f"  F1 Score (macro): {svc_f1:.5f}\n")
                f.write(f"  Confusion Matrix:\n{str(svc_cm).replace(chr(10), chr(10)+'    ')}\n")
                f.write(f"  Classification Report:\n{svc_report}\n")
                f.write("-" * 80 + "\n\n")
        
        # ===== TF-IDF + LightGBM =====
        if not args.no_train_lgb:
            lgb_models, lgb_f1, lgb_pred, lgb_cm, lgb_report = train_tfidf_lgb(
                X_tr, y_tr, X_va, y_va,
                fold, args.ckpt_dir, id2label,
                max_features=args.tfidf_max_features,
                num_leaves=args.lgb_num_leaves,
                learning_rate=args.lgb_lr
            )
            lgb_scores.append(lgb_f1)
            lgb_oof_pred[va_idx] = lgb_pred
            
            # Append fold results to summary
            with open(summary_path, "a") as f:
                f.write(f"Fold {fold} - TF-IDF + LightGBM Results:\n")
                f.write(f"  F1 Score (macro): {lgb_f1:.5f}\n")
                f.write(f"  Confusion Matrix:\n{str(lgb_cm).replace(chr(10), chr(10)+'    ')}\n")
                f.write(f"  Classification Report:\n{lgb_report}\n")
                f.write("-" * 80 + "\n\n")
    
    # ===== Cross-Validation Summary =====
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    if svc_scores:
        svc_cv_mean = float(np.mean(svc_scores))
        svc_cv_std = float(np.std(svc_scores))
        svc_oof_f1 = f1_score(y, svc_oof_pred, average="macro")
        
        print(f"TF-IDF + LinearSVC:")
        print(f"  CV mean: {svc_cv_mean:.5f}, CV std: {svc_cv_std:.5f}")
        print(f"  OOF F1: {svc_oof_f1:.5f}")
        
        # OOF analysis for SVC
        try:
            svc_cm_oof = confusion_matrix(y, svc_oof_pred)
            svc_report_oof = classification_report(
                y, svc_oof_pred,
                target_names=[str(id2label[i]) for i in range(num_labels)]
            )
            
            oof_svc_path = os.path.join(args.ckpt_dir, "oof_analysis_svc.txt")
            with open(oof_svc_path, "w") as f:
                f.write(f"OOF F1 (macro): {svc_oof_f1:.5f}\n")
                f.write(f"CV mean: {svc_cv_mean:.5f}, CV std: {svc_cv_std:.5f}\n\n")
                f.write(f"Confusion Matrix:\n{svc_cm_oof}\n\n")
                f.write(f"Classification Report:\n{svc_report_oof}\n")
            
            oof_svc_cm_csv = os.path.join(args.ckpt_dir, "oof_confusion_matrix_svc.csv")
            oof_svc_report_csv = os.path.join(args.ckpt_dir, "oof_classification_report_svc.csv")
            save_confusion_matrix_as_table(svc_cm_oof, id2label, oof_svc_cm_csv)
            save_classification_report_as_table(y, svc_oof_pred, id2label, oof_svc_report_csv)
            
            print(f"  OOF analysis saved to {oof_svc_path}")
        except Exception as e:
            print(f"[ERROR] Failed to compute OOF metrics for SVC: {e}")
    
    if lgb_scores:
        lgb_cv_mean = float(np.mean(lgb_scores))
        lgb_cv_std = float(np.std(lgb_scores))
        lgb_oof_f1 = f1_score(y, lgb_oof_pred, average="macro")
        
        print(f"\nTF-IDF + LightGBM:")
        print(f"  CV mean: {lgb_cv_mean:.5f}, CV std: {lgb_cv_std:.5f}")
        print(f"  OOF F1: {lgb_oof_f1:.5f}")
        
        # OOF analysis for LGB
        try:
            lgb_cm_oof = confusion_matrix(y, lgb_oof_pred)
            lgb_report_oof = classification_report(
                y, lgb_oof_pred,
                target_names=[str(id2label[i]) for i in range(num_labels)]
            )
            
            oof_lgb_path = os.path.join(args.ckpt_dir, "oof_analysis_lgb.txt")
            with open(oof_lgb_path, "w") as f:
                f.write(f"OOF F1 (macro): {lgb_oof_f1:.5f}\n")
                f.write(f"CV mean: {lgb_cv_mean:.5f}, CV std: {lgb_cv_std:.5f}\n\n")
                f.write(f"Confusion Matrix:\n{lgb_cm_oof}\n\n")
                f.write(f"Classification Report:\n{lgb_report_oof}\n")
            
            oof_lgb_cm_csv = os.path.join(args.ckpt_dir, "oof_confusion_matrix_lgb.csv")
            oof_lgb_report_csv = os.path.join(args.ckpt_dir, "oof_classification_report_lgb.csv")
            save_confusion_matrix_as_table(lgb_cm_oof, id2label, oof_lgb_cm_csv)
            save_classification_report_as_table(y, lgb_oof_pred, id2label, oof_lgb_report_csv)
            
            print(f"  OOF analysis saved to {oof_lgb_path}")
        except Exception as e:
            print(f"[ERROR] Failed to compute OOF metrics for LGB: {e}")
    
    # Append CV summary to main summary file
    with open(summary_path, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERALL CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        if svc_scores:
            f.write(f"TF-IDF + LinearSVC:\n")
            f.write(f"  CV mean F1 (macro): {svc_cv_mean:.5f}\n")
            f.write(f"  CV std F1 (macro): {svc_cv_std:.5f}\n")
            f.write(f"  OOF F1 (macro): {svc_oof_f1:.5f}\n\n")
        
        if lgb_scores:
            f.write(f"TF-IDF + LightGBM:\n")
            f.write(f"  CV mean F1 (macro): {lgb_cv_mean:.5f}\n")
            f.write(f"  CV std F1 (macro): {lgb_cv_std:.5f}\n")
            f.write(f"  OOF F1 (macro): {lgb_oof_f1:.5f}\n")
    
    print(f"\n[INFO] Summary saved to {summary_path}")
    print(f"\n[INFO] All results saved to {args.ckpt_dir}")
    
    # ===== Save Misclassified Samples =====
    print(f"\n{'='*80}")
    print("SAVING MISCLASSIFIED SAMPLES")
    print(f"{'='*80}\n")
    
    if svc_scores:
        save_misclassified_samples(
            y, svc_oof_pred, X_text, train_ids, id2label, "svc", args.ckpt_dir
        )
    
    if lgb_scores:
        save_misclassified_samples(
            y, lgb_oof_pred, X_text, train_ids, id2label, "lgb", args.ckpt_dir
        )
    
    print(f"\n[INFO] Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline models: TF-IDF + LinearSVC, TF-IDF + LightGBM"
    )
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--sub_path", type=str, default=SUB_PATH)
    
    # TF-IDF parameters
    parser.add_argument("--tfidf_max_features", type=int, default=5000)
    # ngram_range is fixed at (1, 2) for unigrams and bigrams
    
    # LinearSVC parameters
    parser.add_argument("--svc_c", type=float, default=1.0)
    parser.add_argument("--no_train_svc", action="store_true", help="Skip training LinearSVC model")
    
    # LightGBM parameters
    parser.add_argument("--lgb_num_leaves", type=int, default=31)
    parser.add_argument("--lgb_lr", type=float, default=0.1)
    parser.add_argument("--no_train_lgb", action="store_true", help="Skip training LightGBM model")
    
    # Cross-validation
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    
    # Output
    parser.add_argument("--ckpt_dir", type=str, default="results/baseline_models_429")
    
    args, _ = parser.parse_known_args()
    
    main(args)
