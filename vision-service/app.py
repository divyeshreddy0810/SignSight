"""
Vision Inference Service: classifies a windowed landmark sequence into a sign
gloss using the trained Random Forest model (see ml/export_final_model.py).

Input:  {"keypoints": [[[x, y, z, visibility] * 75] * WINDOW_SIZE]}
        (pose 33 + left hand 21 + right hand 21, as produced by the frontend
        and smoothed by the preprocessing service)
Output: {"gloss": str, "confidence": float}
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

# Feature extraction is single-sourced in ml/features.py so training and
# serving can never drift apart. Serving mirrors training exactly:
# canonicalize handedness, then v2 normalized features (the incoming window
# is already smoothed by the preprocessing service).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from features import (canonicalize, clip_features_v2, letter_features,  # noqa: E402
                      letter_features_v2, LEFT_HAND_INDICES, RIGHT_HAND_INDICES)

MODEL_PATH = ROOT / "ml" / "models" / "random_forest.joblib"
LETTER_MODEL_PATH = ROOT / "ml" / "models" / "letter_rf.joblib"
NUM_KEYPOINTS = 75
COORDS_PER_KEYPOINT = 4
# ponytail: assumes the standard 640x480 webcam; make configurable if the
# capture setup changes (training crops are square, so serve must isotropize).
CAMERA_ASPECT = 4 / 3
# Abstain rather than guess: a wrong letter costs the user a backspace, a
# withheld one costs nothing. Tuned against the blocked-CV confusion pairs.
LETTER_MIN_PROB = 0.30    # top class must clear this
LETTER_MIN_MARGIN = 0.08  # ...and beat the runner-up by this

LETTER_FEATURE_FNS = {"v1_screen": letter_features, "v2_palmframe": letter_features_v2}

app = FastAPI(title="Vision Inference Service")

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

# The letter artifact carries the feature version it was trained with, so
# serving can never silently pair a model with the wrong descriptor.
letter_model = letter_feature_fn = None
if LETTER_MODEL_PATH.exists():
    _bundle = joblib.load(LETTER_MODEL_PATH)
    letter_model = _bundle["model"]
    letter_feature_fn = LETTER_FEATURE_FNS[_bundle["features"]]


@app.post("/predict")
def predict_gloss(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}. Run ml/export_final_model.py first.")

    keypoints = np.array(data["keypoints"], dtype=np.float32)
    if keypoints.size == 0:
        return {"gloss": "", "confidence": 0.0}
    if keypoints.ndim != 3 or keypoints.shape[1] != NUM_KEYPOINTS or keypoints.shape[2] != COORDS_PER_KEYPOINT:
        raise HTTPException(
            status_code=422,
            detail=f"Expected keypoints shape (frames, {NUM_KEYPOINTS}, {COORDS_PER_KEYPOINT}), got {list(keypoints.shape)}",
        )

    features = clip_features_v2(canonicalize(keypoints[None])[0]).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    best = int(np.argmax(probabilities))
    return {"gloss": str(model.classes_[best]), "confidence": float(probabilities[best])}


@app.post("/predict_letter")
def predict_letter(data: dict):
    """Fingerspelling: mean class probability across the window's frames for the
    dominant hand (the block detected in more frames).

    Averaging probabilities rather than taking a majority vote keeps each
    frame's uncertainty — five hesitant frames no longer outvote three
    confident ones — and yields a usable number to abstain on.
    """
    if letter_model is None:
        raise HTTPException(status_code=503, detail=f"Letter model not found at {LETTER_MODEL_PATH}. Run ml/train_letters.py first.")

    keypoints = np.array(data["keypoints"], dtype=np.float32)
    if keypoints.size == 0:
        return {"letter": "", "confidence": 0.0}
    if keypoints.ndim != 3 or keypoints.shape[1] != NUM_KEYPOINTS or keypoints.shape[2] != COORDS_PER_KEYPOINT:
        raise HTTPException(
            status_code=422,
            detail=f"Expected keypoints shape (frames, {NUM_KEYPOINTS}, {COORDS_PER_KEYPOINT}), got {list(keypoints.shape)}",
        )

    left = keypoints[:, LEFT_HAND_INDICES]
    right = keypoints[:, RIGHT_HAND_INDICES]
    use_left = (left[:, :, 3] > 0).any(axis=1).sum() > (right[:, :, 3] > 0).any(axis=1).sum()
    block = left if use_left else right

    per_frame = [
        letter_feature_fn(frame[:, :3], is_left=bool(use_left), aspect=CAMERA_ASPECT)
        for frame in block if (frame[:, 3] > 0).any()
    ]
    if not per_frame:
        return {"letter": "", "confidence": 0.0}

    mean_probs = letter_model.predict_proba(np.stack(per_frame)).mean(axis=0)
    order = np.argsort(mean_probs)[::-1]
    top, runner_up = float(mean_probs[order[0]]), float(mean_probs[order[1]])
    # Always report the top candidates, even when abstaining: a silent window
    # is indistinguishable from a missing hand or a wrong guess otherwise, and
    # that ambiguity is untriageable from the UI.
    candidates = [{"letter": str(letter_model.classes_[i]), "p": round(float(mean_probs[i]), 3)}
                  for i in order[:3]]
    if top < LETTER_MIN_PROB or (top - runner_up) < LETTER_MIN_MARGIN:
        return {"letter": "", "confidence": top, "candidates": candidates,
                "reason": "low_confidence" if top < LETTER_MIN_PROB else "ambiguous"}
    return {"letter": str(letter_model.classes_[order[0]]), "confidence": top,
            "candidates": candidates, "frames": len(per_frame)}


CALIBRATION_PATH = ROOT / "data" / "calibration_samples.jsonl"


@app.post("/calibrate/save")
def calibrate_save(data: dict):
    """Append one labelled hand from the live camera to the calibration set.

    Samples captured here travel the exact serving path, so they are immune to
    any train/serve transform mismatch as well as to the domain gap — the
    thing the Kaggle-trained model cannot overcome (live confidence ~0.37 vs
    ~0.93 in-sample).
    """
    letter = str(data.get("letter", "")).strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        raise HTTPException(status_code=422, detail=f"Expected a single A-Z letter, got {letter!r}")

    keypoints = np.array(data["keypoints"], dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1] != NUM_KEYPOINTS:
        raise HTTPException(status_code=422, detail=f"Expected (frames, {NUM_KEYPOINTS}, 4), got {list(keypoints.shape)}")

    left = keypoints[:, LEFT_HAND_INDICES]
    right = keypoints[:, RIGHT_HAND_INDICES]
    use_left = (left[:, :, 3] > 0).any(axis=1).sum() > (right[:, :, 3] > 0).any(axis=1).sum()
    block = left if use_left else right

    saved = 0
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "a") as f:
        for frame in block:
            if not (frame[:, 3] > 0).any():
                continue
            f.write(json.dumps({"letter": letter, "is_left": bool(use_left),
                                "landmarks": frame[:, :3].tolist()}) + "\n")
            saved += 1
    total = sum(1 for _ in open(CALIBRATION_PATH)) if CALIBRATION_PATH.exists() else 0
    return {"saved": saved, "total": total, "letter": letter}


@app.get("/calibrate/stats")
def calibrate_stats():
    if not CALIBRATION_PATH.exists():
        return {"total": 0, "per_letter": {}}
    per = {}
    with open(CALIBRATION_PATH) as f:
        for line in f:
            if line.strip():
                per[json.loads(line)["letter"]] = per.get(json.loads(line)["letter"], 0) + 1
    return {"total": sum(per.values()), "per_letter": per}


@app.get("/health")
def health():
    return {"status": "Vision Service Healthy", "model_loaded": model is not None,
            "letter_model_loaded": letter_model is not None}
