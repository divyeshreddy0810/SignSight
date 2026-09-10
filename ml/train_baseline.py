"""
Quick classical-baseline check: single-seed stratified CV for RF and SVM.

ml/evaluate.py is the source of truth for reported results (repeated CV,
mean +/- std, significance tests, subject-independent splits); this script is
a fast exploratory check on unsmoothed features. It writes NO model
artifacts — the deployment model is produced solely by ml/export_final_model.py,
so running this can never clobber what vision-service serves.

Usage: python ml/train_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_dataset

N_FOLDS = 5

CANDIDATES = {
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42,
    ),
    "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42),
}


def evaluate_model(name, model, X, y, cv):
    preds = cross_val_predict(model, X, y, cv=cv)
    report = classification_report(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds, labels=sorted(set(y)))
    print(f"\n=== {name} ({N_FOLDS}-fold stratified CV) ===")
    print(report)
    print("Confusion matrix (rows=true, cols=pred), labels:", sorted(set(y)))
    print(cm)
    return preds


def main():
    X, y, groups = load_dataset()
    print(f"Loaded {X.shape[0]} clips, {X.shape[1]} features, {len(set(y))} classes.")

    min_class_count = min(np.unique(y, return_counts=True)[1])
    n_folds = min(N_FOLDS, min_class_count)
    if n_folds < N_FOLDS:
        print(f"Warning: smallest class has only {min_class_count} samples; using {n_folds}-fold CV instead of {N_FOLDS}.")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for name, model in CANDIDATES.items():
        evaluate_model(name, model, X, y, cv)


if __name__ == "__main__":
    main()
