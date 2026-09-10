import sys
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

app = FastAPI(title="Preprocessing Service")

# SVM validator (proposal Model 1). Falls back to the threshold rule when the
# artifact is absent, so the service still runs on a fresh clone.
VALIDATOR_PATH = ROOT / "ml" / "models" / "svm_validator.joblib"
try:
    from train_validator import window_features
    validator = joblib.load(VALIDATOR_PATH) if VALIDATOR_PATH.exists() else None
except Exception:
    validator, window_features = None, None

NUM_POSE_LANDMARKS = 33
MIN_POSE_VISIBILITY = 0.5   # mean pose visibility: is a person actually in frame?
MIN_HAND_FRAME_FRACTION = 0.3  # fraction of frames with >=1 hand: is there sign content?
SMOOTHING_KERNEL = np.ones(5) / 5

# DBSCAN outlier rejection (proposal Model 2). MediaPipe occasionally throws a
# single hand keypoint far from the rest of the hand — a tracking glitch, not a
# real pose. Those frames survive the moving average (which blurs the error into
# its neighbours instead of removing it), so they are caught here first.
#
# eps is a FRACTION OF THE HAND'S OWN SPAN, not a fixed distance. A fixed eps
# cannot work: a hand's median inter-keypoint distance is ~0.16 in normalized
# coordinates, so any eps small enough to isolate an outlier also fragments a
# normal open hand — at eps=0.12 this destroyed 11.8% of real keypoints, and
# letters with spread fingers (V, W) lost a third of theirs. Measured over the
# calibration corpus with synthetic injected outliers:
#     eps=0.12 fixed     11.8% real keypoints destroyed, 100% outliers caught
#     eps=0.30 fixed      0.0% destroyed,                 77% caught
#     eps=0.8 x span      0.0% destroyed,                 98% caught  <- chosen
# Scaling by span also makes it distance-invariant: the same hand near or far
# from the camera is filtered identically.
DBSCAN_EPS_SPAN_FRACTION = 0.8
DBSCAN_MIN_SAMPLES = 4  # a real handshape always has >3 mutually close keypoints


def reject_outlier_keypoints(keypoints: np.ndarray) -> tuple:
    """Zero out hand keypoints DBSCAN labels as noise, per frame and per hand.

    Returns (cleaned, n_rejected). Only hand blocks are examined: pose keypoints
    are legitimately spread across the body, so density clustering would flag
    the extremities of a normal skeleton.
    """
    cleaned = keypoints.copy()
    n_rejected = 0
    for f in range(cleaned.shape[0]):
        for start in (NUM_POSE_LANDMARKS, NUM_POSE_LANDMARKS + 21):
            block = cleaned[f, start:start + 21]
            present = block[:, 3] > 0
            if present.sum() < DBSCAN_MIN_SAMPLES:
                continue  # absent or barely-tracked hand: nothing to clean
            pts = block[present, :2]
            # 90th percentile rather than max: the max is itself the outlier we
            # are hunting, and would inflate eps enough to hide it.
            span = np.percentile(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1), 90)
            if span < 1e-6:
                continue
            labels = DBSCAN(eps=DBSCAN_EPS_SPAN_FRACTION * span,
                            min_samples=DBSCAN_MIN_SAMPLES).fit(pts).labels_
            noise = np.zeros(21, dtype=bool)
            noise[np.where(present)[0][labels == -1]] = True
            if noise.any():
                cleaned[f, start:start + 21][noise] = 0.0  # mark as not-detected
                n_rejected += int(noise.sum())
    return cleaned, n_rejected

@app.post("/filter")
def filter_keypoints(data: dict):
    # keypoints shape: (frames, 75, 4) -> [x, y, z, visibility];
    # keypoints [0:33] are pose (continuous visibility), [33:75] are hands
    # (binary presence: 1.0 when detected, all-zero rows when absent —
    # including the legitimately absent non-dominant hand in one-handed signs).
    #
    # smooth=False skips the moving average. Fingerspelling needs this: its
    # window is 10 frames, and np.convolve(..., mode='same') zero-pads the
    # edges, so the first and last two frames get averaged with nothing and
    # collapse toward the origin (a held-still 0.5 reads 0.3). That corrupts
    # 4 of 10 frames, and the letter model is trained on unsmoothed static
    # landmarks anyway — smoothing it is pure train/serve skew.
    keypoints = np.array(data["keypoints"])
    if keypoints.size == 0:
        return {"clean_keypoints": []}

    # Quality gate. The SVM validator learns this boundary (CV F1 0.938) rather
    # than asserting it; the threshold rule it replaces scored 0.917 on the same
    # windows and remains the fallback.
    if validator is not None and data.get("use_validator", True):
        if validator.predict(window_features(keypoints)[None])[0] == 0:
            return {"clean_keypoints": [], "rejected_by": "svm_validator"}
    else:
        pose_visibility = keypoints[:, :NUM_POSE_LANDMARKS, 3].mean()
        hand_frame_fraction = (keypoints[:, NUM_POSE_LANDMARKS:, 3].max(axis=1) > 0).mean()
        if pose_visibility < MIN_POSE_VISIBILITY or hand_frame_fraction < MIN_HAND_FRAME_FRACTION:
            return {"clean_keypoints": [], "rejected_by": "threshold_rule"}

    n_rejected = 0
    if data.get("dbscan", True):
        keypoints, n_rejected = reject_outlier_keypoints(keypoints)

    if not data.get("smooth", True):
        return {"clean_keypoints": keypoints.tolist(), "outliers_rejected": n_rejected}

    # Smooth each keypoint's trajectory across the time axis (frames).
    cleaned = np.apply_along_axis(lambda x: np.convolve(x, SMOOTHING_KERNEL, mode='same'), 0, keypoints)
    return {"clean_keypoints": cleaned.tolist(), "outliers_rejected": n_rejected}
