"""
Trains and evaluates the fingerspelling (A-Z) classifier on hand landmarks
from ml/build_alphabet_dataset.py, then exports the deployment artifact
(ml/models/letter_rf.joblib) fitted on all data.

Two protocols are reported, and the gap between them is the point:

  random   — repeated stratified 5-fold CV, the protocol the word model uses.
             Here it LEAKS: the source images are sequential frames of one
             hand, so a random split puts near-duplicate frames in train and
             test at once. Reported only as the number not to believe.
  blocked  — GroupKFold over contiguous frame blocks, so every frame of a
             block is either trained on or tested on, never both. This is the
             honest within-setup estimate.

Neither measures the webcam domain gap: all images come from one capture
setup, and live accuracy on a different camera and hand will be lower again
(the same lesson the word model's signer-leakage gap teaches). J and Z are
motion letters scored here from static frames.

Usage: python ml/train_letters.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import N_FOLDS, SEEDS, mcnemar_exact
from features import augment_hand, letter_features, letter_features_v2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed_alphabet"
MODEL_PATH = ROOT / "ml" / "models" / "letter_rf.joblib"
RESULTS_CSV = ROOT / "results" / "letter_results.csv"
BLOCK_SIZE = 250  # frames per group: coarse enough that neighbours share a block
AUG_COPIES = 2    # augmented duplicates per training sample
LETTER_DEPTH = 16


def letter_rf(seed=42):
    """Deliberately NOT evaluate.rf(): that forest is capped at depth 8, tuned
    for the word model's 104 clips. This task has ~6k samples over 26 classes,
    where depth 8 underfits (blocked CV 0.923 -> 0.941 at depth 16) and, worse,
    leaves probabilities too flat to threshold on — median top-probability of a
    correct call rises 0.72 -> 0.93, which is what the serve-time abstention
    gate keys off."""
    return RandomForestClassifier(n_estimators=200, max_depth=LETTER_DEPTH,
                                  min_samples_leaf=2, class_weight="balanced",
                                  random_state=seed)


def featurize(lms, is_left, fn):
    return np.stack([fn(lm, il) for lm, il in zip(lms, is_left)])


def build_training_set(lms, is_left, y, fn, rng, aug=True, mirror=True):
    """Training rows for one fold.

    mirror=True adds every sample a second time with its handedness flag
    flipped, i.e. the mirror-image handshape. 97% of this dataset is labelled
    'Left' by MediaPipe, so without it the model only ever learns one
    handedness form — and MediaPipe Holistic (what the browser runs) routinely
    assigns the opposite label when a single hand is visible. Measured: a
    flipped label costs 0.94 -> 0.56 without this, and nothing with it.
    """
    lms_out = [lms]
    left_out = [is_left]
    y_out = [y]
    if mirror:
        lms_out.append(lms)
        left_out.append(~is_left)
        y_out.append(y)
    if aug:
        for _ in range(AUG_COPIES):
            jittered = np.stack([augment_hand(lm, rng) for lm in lms])
            lms_out.append(jittered)
            left_out.append(is_left)
            y_out.append(y)
            if mirror:
                lms_out.append(jittered)
                left_out.append(~is_left)
                y_out.append(y)
    return (np.concatenate(lms_out), np.concatenate(left_out), np.concatenate(y_out))


def run_cv(lms, is_left, y, splits, fn, seed, aug=False, mirror=True, flip_test=False):
    """Features are derived inside the fold; augmentation touches training rows
    only, so no augmented copy of a validation sample can reach the model.
    flip_test inverts handedness on the VALIDATION rows only — it simulates
    MediaPipe mislabelling the hand at serve time."""
    preds = np.empty_like(y)
    rng = np.random.default_rng(seed)
    for tr, va in splits:
        lms_tr, left_tr, y_tr = build_training_set(
            lms[tr], is_left[tr], y[tr], fn, rng, aug=aug, mirror=mirror)
        X_tr = featurize(lms_tr, left_tr, fn)
        left_va = ~is_left[va] if flip_test else is_left[va]
        X_va = featurize(lms[va], left_va, fn)
        preds[va] = letter_rf(seed).fit(X_tr, y_tr).predict(X_va)
    return preds


CALIBRATION_JSONL = ROOT / "data" / "calibration_samples.jsonl"
CALIBRATION_WEIGHT = 4  # repeats per calibration sample; see load_calibration()


def load_calibration():
    """Hands captured from the live camera via the Calibrate panel.

    These are worth far more per sample than the Kaggle images: they carry the
    deployment camera, hand and lighting, and they travel the serving path, so
    they also absorb any train/serve transform difference. There are only a few
    hundred of them against ~6k images, so they are repeated CALIBRATION_WEIGHT
    times to actually influence the forest rather than be averaged away.
    """
    if not CALIBRATION_JSONL.exists():
        return None
    lms, left, y, burst = [], [], [], []
    prev_letter, b = None, -1
    with open(CALIBRATION_JSONL) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            # One Capture press writes a run of consecutive frames for one
            # letter. Those frames are near-duplicates, so they must share a CV
            # group — otherwise blocked CV trains and tests on the same press
            # and reports a fantasy. A new burst starts whenever the letter
            # changes or a 10-frame window boundary passes.
            if rec["letter"] != prev_letter or len(y) % 10 == 0:
                b += 1
                prev_letter = rec["letter"]
            lms.append(rec["landmarks"])
            left.append(rec["is_left"])
            y.append(rec["letter"])
            burst.append(b)
    if not y:
        return None
    return (np.array(lms, dtype=np.float32), np.array(left), np.array(y), np.array(burst))


def main():
    use_cal = "--calibration" in sys.argv
    lms = np.load(DATA / "landmarks.npy")
    is_left = np.load(DATA / "is_left.npy")
    y = np.load(DATA / "labels.npy")
    frame_idx = np.load(DATA / "frame_idx.npy")

    cal = load_calibration() if use_cal else None
    if use_cal and cal is None:
        print("--calibration given but no samples in data/calibration_samples.jsonl "
              "— capture some in the Calibrate panel first.")
        raise SystemExit(1)
    # Image-source groups: contiguous frame blocks of the sequential captures.
    groups = np.array([f"img_{lab}_{idx // BLOCK_SIZE}" for lab, idx in zip(y, frame_idx)])

    cal_mask = np.zeros(len(y), dtype=bool)
    if cal is not None:
        c_lms, c_left, c_y, c_burst = cal
        per = Counter(c_y)
        print(f"calibration: {len(c_y)} live frames, {len(set(c_burst))} capture bursts, "
              f"{len(per)} letters (min {min(per.values())}, max {max(per.values())}), "
              f"weight x{CALIBRATION_WEIGHT}")
        missing = sorted(set(y) - set(c_y))
        if missing:
            print(f"  no calibration for {''.join(missing)} — those fall back to image data only")
        lms = np.concatenate([lms, np.repeat(c_lms, CALIBRATION_WEIGHT, axis=0)])
        is_left = np.concatenate([is_left, np.repeat(c_left, CALIBRATION_WEIGHT)])
        y = np.concatenate([y, np.repeat(c_y, CALIBRATION_WEIGHT)])
        # Group by capture burst, so a held-out fold contains presses the model
        # never trained on — the closest offline proxy for "a new live session".
        groups = np.concatenate([groups, np.repeat([f"cal_{b}" for b in c_burst], CALIBRATION_WEIGHT)])
        cal_mask = np.concatenate([cal_mask, np.ones(len(c_y) * CALIBRATION_WEIGHT, dtype=bool)])

    classes = sorted(set(y))
    print(f"{len(lms)} samples, {len(classes)} classes "
          f"(support {min(Counter(y).values())}-{max(Counter(y).values())})")
    blocked = list(GroupKFold(n_splits=N_FOLDS).split(lms, y, groups))
    print(f"blocked CV over {len(set(groups))} frame blocks of {BLOCK_SIZE} frames\n")

    # --- the number not to believe: random splits over sequential frames ---
    X_v1 = featurize(lms, is_left, letter_features)
    leaky = [accuracy_score(y, run_cv(lms, is_left, y,
                StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(X_v1, y),
                letter_features, s))
             for s in SEEDS[:3]]
    print(f"[random CV, LEAKY — do not quote] acc {np.mean(leaky):.4f} +/- {np.std(leaky):.4f}\n")

    # --- honest sweep: features x augmentation x handedness-mirroring ---
    # Each config is scored twice: with MediaPipe's handedness label as given,
    # and with it inverted. The second column is the one that matters live.
    configs = [
        ("v1_screen", letter_features, False, False),
        ("v1_screen+aug", letter_features, True, False),
        ("v1_screen+aug+mirror", letter_features, True, True),
        ("v2_palmframe+aug", letter_features_v2, True, False),
        ("v2_palmframe+aug+mirror", letter_features_v2, True, True),
    ]
    rows, oof = [], {}
    for name, fn, aug, mirror in configs:
        preds = run_cv(lms, is_left, y, blocked, fn, SEEDS[0], aug=aug, mirror=mirror)
        flipped = run_cv(lms, is_left, y, blocked, fn, SEEDS[0], aug=aug, mirror=mirror, flip_test=True)
        acc, f1 = accuracy_score(y, preds), f1_score(y, preds, average="macro")
        acc_flip = accuracy_score(y, flipped)
        oof[name] = preds
        row = {"config": name, "protocol": "blocked_cv", "accuracy": round(acc, 4),
               "macro_f1": round(f1, 4), "augmented": aug, "handedness_mirrored": mirror,
               "accuracy_handedness_flipped": round(acc_flip, 4),
               "worst_case": round(min(acc, acc_flip), 4)}
        line = (f"[blocked] {name:24s} acc {acc:.4f}  F1 {f1:.4f}  "
                f"| flipped {acc_flip:.4f}  worst {min(acc, acc_flip):.4f}")
        if cal_mask.any():
            # Held-out live frames only: the closest offline estimate of what
            # the browser will actually do, since these came from that camera.
            acc_live = accuracy_score(y[cal_mask], preds[cal_mask])
            row["accuracy_live_calibration"] = round(acc_live, 4)
            line += f"  | LIVE {acc_live:.4f}"
        rows.append(row)
        print(line, flush=True)

    print(f"\nleakage gap (random - blocked, v1): {np.mean(leaky) - rows[0]['accuracy']:+.4f}")
    # Select on WORST CASE across handedness, not best case: live accuracy is
    # whatever MediaPipe's coin-flip hands us, so a config that scores 0.94 one
    # way and 0.56 the other is worse in practice than a flat 0.93.
    # Prefer live-calibration accuracy when we have it: that is the deployment
    # domain. Fall back to the handedness worst case otherwise.
    key = "accuracy_live_calibration" if cal_mask.any() else "worst_case"
    best = max(rows, key=lambda r: r.get(key, r["worst_case"]))
    n_a, n_b, p = mcnemar_exact(y, oof[best["config"]], oof["v1_screen"])
    print(f"McNemar {best['config']} vs v1_screen: only_new={n_a} only_v1={n_b} p={p:.2e}")

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"saved {RESULTS_CSV}")

    preds_b = oof[best["config"]]
    cm = confusion_matrix(y, preds_b, labels=classes)
    worst = sorted(range(len(classes)), key=lambda i: cm[i, i] / max(cm[i].sum(), 1))[:6]
    print("\nweakest letters:", [(classes[i], f"{cm[i, i]}/{cm[i].sum()}") for i in worst])
    confused = sorted(((cm[i, j], classes[i], classes[j])
                       for i in range(len(classes)) for j in range(len(classes)) if i != j),
                      reverse=True)[:5]
    print("top confusions:", [(f"{t}->{p_}", int(n)) for n, t, p_ in confused])
    print("caveat: one capture setup — webcam domain gap unmeasured; J/Z static.")

    # --- export the winner, fitted on everything ---
    fn = {n: f for n, f, _, _ in configs}[best["config"]]
    rng = np.random.default_rng(SEEDS[0])
    lms_all, left_all, y_all = build_training_set(
        lms, is_left, y, fn, rng, aug=best["augmented"], mirror=best["handedness_mirrored"])
    model = letter_rf().fit(featurize(lms_all, left_all, fn), y_all)
    feature_version = best["config"].split("+")[0]
    # compress=3: a depth-16 forest on ~95k rows is ~180 MB raw, over GitHub's
    # 100 MB file limit. Compression takes it to ~22 MB with identical behaviour.
    joblib.dump({"model": model, "features": feature_version}, MODEL_PATH, compress=3)
    print(f"\nsaved {MODEL_PATH} ({best['config']}, {len(y_all)} training rows)")


if __name__ == "__main__":
    main()
