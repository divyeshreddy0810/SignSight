"""
LSTM/GRU sequence model: trained directly on raw (window, 75, 4) landmark
sequences, no hand-crafted features. This is the "advanced method" compared
against the classical Random Forest/SVM baseline in train_baseline.py.

ml/evaluate.py is the source of truth for reported GRU/LSTM results (repeated
CV, mean +/- std); this script is single-seed and exploratory. Its .pt
artifacts are not served by any service.

Usage: python ml/train_lstm.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
MODELS_DIR = ROOT / "ml" / "models"

N_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 8
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.3
LEARNING_RATE = 1e-3


class GestureSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, cell="gru"):
        super().__init__()
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if isinstance(self.rnn, nn.LSTM):
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) — final layer's hidden state
        return self.fc(self.dropout(last_hidden))


def load_sequences():
    labels = pd.read_csv(LABELS_CSV)
    X, y = [], []
    for _, row in labels.iterrows():
        npy_path = PROCESSED_DIR / f"{row['clip_id']}.npy"
        if not npy_path.exists():
            continue
        seq = np.load(npy_path)  # (T, 75, 4)
        X.append(seq.reshape(seq.shape[0], -1))  # (T, 300) — flatten keypoint+coord dims
        y.append(row["gloss"])
    return np.array(X, dtype=np.float32), np.array(y)


def train_one_fold(X_train, y_train, X_val, y_val, num_classes, cell="gru"):
    model = GestureSequenceModel(
        input_size=X_train.shape[-1], hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT, cell=cell,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_preds = val_logits.argmax(dim=1).numpy()
    return model, val_preds


def main(cell="gru"):
    X, y_str = load_sequences()
    print(f"Loaded {X.shape[0]} clips, sequence length {X.shape[1]}, feature dim {X.shape[2]}.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    min_class_count = min(np.bincount(y))
    n_folds = min(N_FOLDS, min_class_count)
    if n_folds < N_FOLDS:
        print(f"Warning: smallest class has only {min_class_count} samples; using {n_folds}-fold CV instead of {N_FOLDS}.")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_preds = np.zeros_like(y)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        model, val_preds = train_one_fold(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            num_classes=len(encoder.classes_), cell=cell,
        )
        all_preds[val_idx] = val_preds
        print(f"Fold {fold + 1}/{n_folds} done.")

    print(f"\n=== {cell.upper()} ({n_folds}-fold stratified CV) ===")
    print(classification_report(y, all_preds, target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), labels:", list(encoder.classes_))
    print(confusion_matrix(y, all_preds))

    # Final fit on all data for deployment.
    final_model = GestureSequenceModel(
        input_size=X.shape[-1], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        num_classes=len(encoder.classes_), dropout=DROPOUT, cell=cell,
    )
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(final_model(xb), yb)
            loss.backward()
            optimizer.step()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": final_model.state_dict(),
        "classes": list(encoder.classes_),
        "input_size": X.shape[-1],
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "cell": cell,
    }, MODELS_DIR / f"{cell}_model.pt")
    print(f"Saved {cell} model -> {MODELS_DIR / f'{cell}_model.pt'}")


if __name__ == "__main__":
    cell = sys.argv[1] if len(sys.argv) > 1 else "gru"
    main(cell=cell)
