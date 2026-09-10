"""
Evaluation sweep: systematically varies parametrisations and sampling methods
to characterise performance beyond a single train/test number.

Dimensions swept
  1. Landmark subset (RF): hands-only vs arm-pose-only vs hands+arms.
     Tests whether hand shape, gross arm motion, or both carry the signal.
  2. Feature group (RF): mean / std / range / velocity alone vs all four.
     Identifies which summary statistics the classical model actually uses.
  3. Temporal resolution (RF): 30 -> 15 -> 10-frame downsampled windows.
     Tests how much temporal detail the classifier needs.
  4. Augmentation (RF + GRU): none / mirror / noise / mirror+noise, applied to
     TRAINING folds only. Mirroring swaps handedness (left<->right keypoint
     blocks + x-reflection), the standard trick for sign data where signers
     differ in dominant hand.
  5. Sampling strategy (RF): stratified random CV vs subject-independent
     (grouped by signer_id) CV. The gap between them estimates how much the
     random split leaks signer identity.
  6. Temporal smoothing (RF + GRU, deployment config): raw vs the 5-frame
     moving average the preprocessing service applies at inference time.
     Makes the deployed configuration's headline number reproducible here.
  7. Feature representation (RF, deployment config): absolute hand
     coordinates (v1) vs wrist-relative shape + body-relative location (v2),
     each with and without handedness canonicalization.

Every stratified configuration is run as REPEATED CV (N_FOLDS folds x
N_REPEATS seeds); the CSV reports mean +/- std across seeds. Exact McNemar
tests on paired out-of-fold predictions (first seed) back the report's
headline comparisons. Subject-independent GroupKFold is deterministic, so it
is run once.

Outputs: results/evaluation_results.csv, results/significance_tests.csv,
plus report-ready figures (with std error bars) under results/figures/.

Usage: python ml/evaluate.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lstm import BATCH_SIZE, DROPOUT, EPOCHS, HIDDEN_SIZE, LEARNING_RATE, NUM_LAYERS, GestureSequenceModel
from features import canonicalize, features_v2, mirror  # single-sourced with serving
from stgcn import STGCN, to_graph_input

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42
N_FOLDS = 5
N_REPEATS = 10
SEEDS = list(range(SEED, SEED + N_REPEATS))

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
ARM_POSE_INDICES = [11, 12, 13, 14, 15, 16]
LEFT_HAND = list(range(33, 54))
RIGHT_HAND = list(range(54, 75))

LANDMARK_SUBSETS = {
    "hands_only": LEFT_HAND + RIGHT_HAND,
    "arms_only": ARM_POSE_INDICES,
    "hands+arms": ARM_POSE_INDICES + LEFT_HAND + RIGHT_HAND,
}

FEATURE_GROUPS = ["mean", "std", "range", "velocity"]


# ---------- data ----------

def load_raw():
    labels = pd.read_csv(LABELS_CSV)
    X, y, groups = [], [], []
    for _, row in labels.iterrows():
        npy_path = PROCESSED_DIR / f"{row['clip_id']}.npy"
        if not npy_path.exists():
            continue
        X.append(np.load(npy_path))  # (30, 75, 4)
        y.append(row["gloss"])
        groups.append(row["signer_id"])
    return np.stack(X), np.array(y), np.array(groups)


# ---------- features ----------

def features(sequences, keypoint_indices, groups_used=FEATURE_GROUPS):
    """sequences: (N, T, 75, 4) -> (N, D) summary-statistic features."""
    kp = sequences[:, :, keypoint_indices, :3]  # (N, T, K, 3)
    parts = []
    if "mean" in groups_used:
        parts.append(kp.mean(axis=1).reshape(len(kp), -1))
    if "std" in groups_used:
        parts.append(kp.std(axis=1).reshape(len(kp), -1))
    if "range" in groups_used:
        parts.append((kp.max(axis=1) - kp.min(axis=1)).reshape(len(kp), -1))
    if "velocity" in groups_used:
        parts.append(np.abs(np.diff(kp, axis=1)).mean(axis=1).reshape(len(kp), -1))
    return np.concatenate(parts, axis=1)


# ---------- augmentation (training folds only) ----------
# mirror() lives in ml/features.py (shared with canonicalization at serve time).

def jitter(sequences, sigma=0.01, rng=None):
    """Gaussian noise on x,y,z of detected keypoints only."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    out = sequences.copy()
    noise = rng.normal(0, sigma, out[..., :3].shape).astype(out.dtype)
    detected = (out[..., 3] > 0)[..., None]
    out[..., :3] = out[..., :3] + noise * detected
    return out


SMOOTHING_KERNEL = np.ones(5) / 5


def smooth(sequences):
    """Same 5-frame moving average the preprocessing service applies at
    inference time (train/serve consistency); denoises MediaPipe jitter."""
    return np.apply_along_axis(
        lambda x: np.convolve(x, SMOOTHING_KERNEL, mode="same"), 1, sequences
    ).astype(np.float32)


AUGMENTATIONS = {
    "none": lambda X, rng=None: X,
    "mirror": lambda X, rng=None: np.concatenate([X, mirror(X)]),
    "noise": lambda X, rng=None: np.concatenate([X, jitter(X, rng=rng)]),
    "mirror+noise": lambda X, rng=None: np.concatenate([X, mirror(X), jitter(X, rng=rng)]),
}


def augment_labels(y, factor):
    return np.concatenate([y] * factor)


# ---------- models ----------

def rf(seed=SEED):
    return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                  class_weight="balanced", random_state=seed)


def svm(seed=SEED):
    # same config as train_baseline.py's svm_rbf candidate
    return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed)


def train_stgcn_fold(raw_train, y_train, raw_val, num_classes, seed=SEED):
    """One CV fold of the ST-GCN (the model family the proposal specified).

    Takes RAW landmark windows, not flattened features: the whole point of the
    architecture is that it consumes the skeleton as a graph. Same optimizer
    budget as the GRU/LSTM arms so the comparison is like-for-like.
    """
    torch.manual_seed(seed)
    model = STGCN(num_classes=num_classes, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(to_graph_input(raw_train)),
                                        torch.from_numpy(y_train).long())
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(to_graph_input(raw_val))).argmax(dim=1).numpy()


def train_gru_fold(X_train, y_train, X_val, num_classes, seed=SEED, cell="gru"):
    torch.manual_seed(seed)
    model = GestureSequenceModel(input_size=X_train.shape[-1], hidden_size=HIDDEN_SIZE,
                                 num_layers=NUM_LAYERS, num_classes=num_classes,
                                 dropout=DROPOUT, cell=cell)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_val)).argmax(dim=1).numpy()


# ---------- sweep harness ----------

def cv_run(model_kind, raw, y_enc, num_classes, cv_splits, *, seed=SEED,
           keypoints=None, feature_groups=FEATURE_GROUPS, aug="none", frame_step=1,
           canonical=False, feature_fn=None):
    """Runs one config through CV; returns (accuracy, macro_f1, oof_preds).
    canonical: apply handedness canonicalization (train and val alike).
    feature_fn: batch feature extractor overriding the default summary stats."""
    preds = np.zeros_like(y_enc)
    aug_fn = AUGMENTATIONS[aug]
    factor = len(aug_fn(np.zeros((1, 2, 75, 4), dtype=np.float32)))
    rng = np.random.default_rng(seed)
    for train_idx, val_idx in cv_splits:
        raw_tr = aug_fn(raw[train_idx], rng=rng)[:, ::frame_step]
        raw_va = raw[val_idx][:, ::frame_step]
        if canonical:
            raw_tr, raw_va = canonicalize(raw_tr), canonicalize(raw_va)
        y_tr = augment_labels(y_enc[train_idx], factor)
        if model_kind in ("rf", "svm"):
            X_tr = feature_fn(raw_tr) if feature_fn else features(raw_tr, keypoints, feature_groups)
            X_va = feature_fn(raw_va) if feature_fn else features(raw_va, keypoints, feature_groups)
            m = (rf(seed) if model_kind == "rf" else svm(seed)).fit(X_tr, y_tr)
            preds[val_idx] = m.predict(X_va)
        elif model_kind == "stgcn":
            preds[val_idx] = train_stgcn_fold(raw_tr, y_tr, raw_va, num_classes, seed=seed)
        else:  # gru / lstm
            X_tr = raw_tr.reshape(len(raw_tr), raw_tr.shape[1], -1)
            X_va = raw_va.reshape(len(raw_va), raw_va.shape[1], -1)
            preds[val_idx] = train_gru_fold(X_tr, y_tr, X_va, num_classes, seed=seed, cell=model_kind)
    return accuracy_score(y_enc, preds), f1_score(y_enc, preds, average="macro"), preds


def repeated_cv(model_kind, raw, y_enc, num_classes, **cfg):
    """Repeated stratified CV over SEEDS. Returns (accs, f1s, first_seed_preds).
    Each seed reshuffles the folds AND reseeds model training / noise, so the
    std captures split variance and training variance together."""
    accs, f1s, preds0 = [], [], None
    for s in SEEDS:
        splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=s).split(raw, y_enc))
        acc, f1, preds = cv_run(model_kind, raw, y_enc, num_classes, splits, seed=s, **cfg)
        accs.append(acc)
        f1s.append(f1)
        if preds0 is None:
            preds0 = preds
    return accs, f1s, preds0


def mcnemar_exact(y_true, preds_a, preds_b):
    """Exact McNemar test on paired predictions (binomial on discordant pairs).
    Returns (n_a_only_correct, n_b_only_correct, two_sided_p)."""
    from scipy.stats import binomtest
    a_ok, b_ok = preds_a == y_true, preds_b == y_true
    n_a = int(np.sum(a_ok & ~b_ok))
    n_b = int(np.sum(~a_ok & b_ok))
    n = n_a + n_b
    p = binomtest(min(n_a, n_b), n, 0.5).pvalue if n > 0 else 1.0
    return n_a, n_b, p


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw, y_str, groups = load_raw()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    classes = list(encoder.classes_)
    print(f"{len(raw)} clips, {len(classes)} classes.")

    grouped = list(GroupKFold(n_splits=N_FOLDS).split(raw, y, groups))

    rows = []
    oof = {}  # "sweep/config" -> first-seed OOF preds, for paired McNemar tests
    best = {"acc": -1}

    def record(sweep, config, accs, f1s, preds=None, splits="stratified"):
        acc_m, acc_s = float(np.mean(accs)), float(np.std(accs))
        f1_m, f1_s = float(np.mean(f1s)), float(np.std(f1s))
        rows.append({"sweep": sweep, "config": config, "splits": splits,
                     "accuracy_mean": round(acc_m, 4), "accuracy_std": round(acc_s, 4),
                     "macro_f1_mean": round(f1_m, 4), "macro_f1_std": round(f1_s, 4),
                     "n_seeds": len(accs)})
        print(f"[{sweep}] {config} ({splits}): acc={acc_m:.3f}+/-{acc_s:.3f} "
              f"f1={f1_m:.3f}+/-{f1_s:.3f} ({len(accs)} seeds)", flush=True)
        nonlocal best
        if preds is not None:
            oof[f"{sweep}/{config}"] = preds
            if acc_m > best["acc"]:
                best = {"acc": acc_m, "preds": preds, "config": f"{sweep}/{config}"}

    hands_arms = LANDMARK_SUBSETS["hands+arms"]
    hands_only = LANDMARK_SUBSETS["hands_only"]

    # 0. Model family comparison (Table II config: hands+arms, 30 frames, no aug)
    for kind in ["rf", "svm", "gru", "lstm", "stgcn"]:
        accs, f1s, _ = repeated_cv(kind, raw, y, len(classes), keypoints=hands_arms)
        record("model_family", kind, accs, f1s)

    # 1. Landmark subset (RF)
    for name, idxs in LANDMARK_SUBSETS.items():
        accs, f1s, preds = repeated_cv("rf", raw, y, len(classes), keypoints=idxs)
        record("landmark_subset", name, accs, f1s, preds)

    # 2. Feature-group ablation (RF, hands+arms)
    for grp in FEATURE_GROUPS:
        accs, f1s, _ = repeated_cv("rf", raw, y, len(classes), keypoints=hands_arms, feature_groups=[grp])
        record("feature_group", grp, accs, f1s)
    accs, f1s, _ = repeated_cv("rf", raw, y, len(classes), keypoints=hands_arms)
    record("feature_group", "all", accs, f1s)

    # 3. Temporal resolution (RF)
    for step, label in [(1, "30_frames"), (2, "15_frames"), (3, "10_frames")]:
        accs, f1s, _ = repeated_cv("rf", raw, y, len(classes), keypoints=hands_arms, frame_step=step)
        record("temporal_resolution", label, accs, f1s)

    # 4. Augmentation (RF and GRU)
    for aug in AUGMENTATIONS:
        accs, f1s, preds = repeated_cv("rf", raw, y, len(classes), keypoints=hands_arms, aug=aug)
        record("augmentation_rf", aug, accs, f1s, preds)
    for aug in AUGMENTATIONS:
        accs, f1s, preds = repeated_cv("gru", raw, y, len(classes), aug=aug)
        record("augmentation_gru", aug, accs, f1s, preds)

    # 5. Sampling strategy (RF): stratified vs subject-independent.
    # GroupKFold is deterministic (no shuffle), so the grouped protocol is a
    # single run; only model-training seeds would vary, not the splits.
    accs, f1s, _ = repeated_cv("rf", raw, y, len(classes), keypoints=hands_arms)
    record("sampling", "stratified_random", accs, f1s)
    acc, f1, _ = cv_run("rf", raw, y, len(classes), grouped, keypoints=hands_arms)
    record("sampling", "subject_independent", [acc], [f1], splits="grouped_by_signer")

    # 6. Temporal smoothing (deployment config: hands-only + mirror), plus the
    # augmentation choices on smoothed data so the deployed recipe is the
    # measured argmax rather than an assumption carried over from raw data.
    raw_smoothed = smooth(raw)
    accs, f1s, preds = repeated_cv("rf", raw, y, len(classes), keypoints=hands_only, aug="mirror")
    record("smoothing", "raw", accs, f1s, preds)
    accs, f1s, preds = repeated_cv("rf", raw_smoothed, y, len(classes), keypoints=hands_only, aug="mirror")
    record("smoothing", "smoothed", accs, f1s, preds)
    accs, f1s, preds = repeated_cv("rf", raw_smoothed, y, len(classes), keypoints=hands_only, aug="none")
    record("smoothing", "smoothed_noaug", accs, f1s, preds)
    accs, f1s, preds = repeated_cv("rf", raw_smoothed, y, len(classes), keypoints=hands_only, aug="noise")
    record("smoothing", "smoothed_noise", accs, f1s, preds)
    accs, f1s, preds = repeated_cv("gru", raw_smoothed, y, len(classes), aug="mirror")
    record("smoothing", "smoothed_gru", accs, f1s, preds)

    # 7. Feature representation (deployment config: smoothed, no augmentation).
    # v1 = absolute hand coordinates; v2 = wrist-relative hand shape plus
    # nose-relative, shoulder-scaled hand location; canonical = mirror
    # left-dominant clips so every clip is right-hand-dominant.
    accs, f1s, preds = repeated_cv("rf", raw_smoothed, y, len(classes), keypoints=hands_only)
    record("features", "v1_absolute", accs, f1s, preds)
    accs, f1s, preds = repeated_cv("rf", raw_smoothed, y, len(classes), keypoints=hands_only, canonical=True)
    record("features", "v1_canonical", accs, f1s, preds)
    accs, f1s, preds = repeated_cv("rf", raw_smoothed, y, len(classes), feature_fn=features_v2)
    record("features", "v2_normalized", accs, f1s, preds)
    accs, f1s, preds = repeated_cv("rf", raw_smoothed, y, len(classes), feature_fn=features_v2, canonical=True)
    record("features", "v2_canonical", accs, f1s, preds)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "evaluation_results.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'evaluation_results.csv'}")

    # ---------- significance tests (exact McNemar on paired first-seed OOF preds) ----------
    comparisons = [
        ("hands vs arms (RF)", "landmark_subset/hands_only", "landmark_subset/arms_only"),
        ("mirror vs none (RF)", "augmentation_rf/mirror", "augmentation_rf/none"),
        ("RF vs GRU (no aug)", "augmentation_rf/none", "augmentation_gru/none"),
        ("RF vs GRU (best of each)", "smoothing/smoothed", "smoothing/smoothed_gru"),
        ("smoothed vs raw (deploy config)", "smoothing/smoothed", "smoothing/raw"),
        ("v2+canonical vs v1 (deploy config)", "features/v2_canonical", "features/v1_absolute"),
    ]
    sig_rows = []
    for label, key_a, key_b in comparisons:
        n_a, n_b, p = mcnemar_exact(y, oof[key_a], oof[key_b])
        sig_rows.append({
            "comparison": label, "config_a": key_a, "config_b": key_b,
            "acc_a": round(accuracy_score(y, oof[key_a]), 4),
            "acc_b": round(accuracy_score(y, oof[key_b]), 4),
            "a_only_correct": n_a, "b_only_correct": n_b,
            "p_value": round(p, 4),
        })
        print(f"[mcnemar] {label}: a_only={n_a} b_only={n_b} p={p:.4f}", flush=True)
    pd.DataFrame(sig_rows).to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'significance_tests.csv'}")

    # ---------- figures ----------
    for sweep_name in results["sweep"].unique():
        sub = results[results["sweep"] == sweep_name]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["accuracy_mean"], width=0.36, label="Accuracy",
               yerr=sub["accuracy_std"], capsize=3)
        ax.bar(x + 0.18, sub["macro_f1_mean"], width=0.36, label="Macro F1",
               yerr=sub["macro_f1_std"], capsize=3)
        ax.axhline(1 / len(classes), color="gray", linestyle="--", linewidth=1, label="Chance")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["config"], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(f"Sweep: {sweep_name.replace('_', ' ')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"sweep_{sweep_name}.png", dpi=150)
        plt.close(fig)

    cm = confusion_matrix(y, best["preds"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Best config: {best['config']} (mean acc={best['acc']:.2f})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_best.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved under {FIGURES_DIR}")
    print(f"Best config: {best['config']} acc={best['acc']:.3f}")


if __name__ == "__main__":
    main()
