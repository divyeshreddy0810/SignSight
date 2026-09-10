"""
SVM frame-window validator (proposal Model 1): decides whether a window holds
sign content or is idle/noise, so the recogniser is never asked to classify an
empty window.

The deployed system originally used two hand-picked thresholds (mean pose
visibility >= 0.5, hands present in >= 30% of frames). That works, but the cut
points were chosen by eye and never measured. This trains an RBF SVM on the
same cheap signals and reports F1, as the proposal specified.

Positive class: real sign windows (the 104 WLASL clips).
Negative class: idle windows synthesised from the same clips — hands dropped
out of frame, tracking lost, or the signer still. Synthesising negatives is
necessary because the corpus contains no labelled idle footage; each mode
mirrors a way the live camera actually produces a contentless window.

SCOPE: this validator gates the WORD path only. Its "signer still" negative is
indistinguishable from a held fingerspelled letter — wired onto the letter
path it rejected 23/23 genuine letters — so gateway /fingerspell opts out and
keeps the cheap threshold rule. Extending it to fingerspelling would require
held-still letters as POSITIVES (the calibration corpus would supply them).

Usage: python ml/train_validator.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_raw, NUM_POSE_LANDMARKS, SEED

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "models" / "svm_validator.joblib"
HAND_START = NUM_POSE_LANDMARKS


def window_features(window: np.ndarray) -> np.ndarray:
    """(T, 75, 4) -> 5 cheap signals available at serve time with no model.

    Deliberately not the recogniser's 504-dim features: a gatekeeper that costs
    as much as the model it guards saves nothing.
    """
    pose_vis = window[:, :HAND_START, 3]
    hands = window[:, HAND_START:, :]
    hand_present = (hands[:, :, 3] > 0).any(axis=1)
    detected = hands[hand_present][:, :, :3] if hand_present.any() else np.zeros((1, 42, 3))
    motion = np.abs(np.diff(window[:, HAND_START:, :3], axis=0)).mean() if len(window) > 1 else 0.0
    return np.array([
        pose_vis.mean(),                    # is a person in frame at all
        pose_vis.std(),
        hand_present.mean(),                # fraction of frames with a hand
        float(detected.std()),              # hand keypoint spread: a real handshape has structure
        float(motion),                      # movement: idle hands are static
    ], dtype=np.float32)


def synthesise_negatives(raw: np.ndarray, rng) -> np.ndarray:
    """Idle/noise windows in the three ways the live camera produces them."""
    negatives = []
    for i, window in enumerate(raw):
        mode = i % 3
        w = window.copy()
        if mode == 0:                     # hands out of frame entirely
            w[:, HAND_START:, :] = 0.0
        elif mode == 1:                   # tracking lost: hands flicker in <15% of frames
            keep = rng.random(len(w)) < 0.12
            w[~keep, HAND_START:, :] = 0.0
        else:                             # signer still: one frame held for the window
            w = np.repeat(w[:1], len(w), axis=0)
            w[:, HAND_START:, :3] += rng.normal(0, 0.001, w[:, HAND_START:, :3].shape)
        negatives.append(w)
    return np.stack(negatives)


def main():
    raw, _, _ = load_raw()
    rng = np.random.default_rng(SEED)
    negatives = synthesise_negatives(raw, rng)

    X = np.stack([window_features(w) for w in np.concatenate([raw, negatives])])
    y = np.concatenate([np.ones(len(raw)), np.zeros(len(negatives))]).astype(int)
    print(f"{len(X)} windows: {int(y.sum())} sign, {int((1 - y).sum())} idle/noise")

    # No probability=True: the gate only needs a decision, and sklearn 1.9
    # deprecates it (Platt scaling would also be meaningless on 208 windows).
    model = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", C=1.0, gamma="scale",
                              class_weight="balanced", random_state=SEED))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = cross_val_predict(model, X, y, cv=cv)
    print(f"\n5-fold CV F1 (sign class): {f1_score(y, preds):.4f}")
    print(classification_report(y, preds, target_names=["idle/noise", "sign"], zero_division=0))

    # Compare against the hand-tuned rule the service currently uses, on the
    # same data — the point is whether learning the boundary beats guessing it.
    rule = np.array([1 if (w[:, :HAND_START, 3].mean() >= 0.5 and
                           (w[:, HAND_START:, 3].max(axis=1) > 0).mean() >= 0.3) else 0
                     for w in np.concatenate([raw, negatives])])
    print(f"hand-tuned threshold rule F1: {f1_score(y, rule):.4f}")

    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
