"""
Trains and exports the deployment model using the best configuration found by
the evaluation sweep (ml/evaluate.py): Random Forest on smoothed, handedness-
canonicalized, v2-normalized features (wrist-relative hand shape +
body-relative location) — see the "features" sweep in
results/evaluation_results.csv for its cross-validated score.

Unlike train_baseline.py (which reports CV scores), this fits on ALL data,
since the deployed model should use every available sample. Overwrites
ml/models/random_forest.joblib, the artifact vision-service/app.py loads at
startup.

Usage: python ml/export_final_model.py
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_raw, rf, smooth
from features import canonicalize, features_v2

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "models" / "random_forest.joblib"


def main():
    raw, y, _ = load_raw()
    X = features_v2(canonicalize(smooth(raw)))
    print(f"Training final model on {len(X)} clips, {X.shape[1]} features.")

    model = rf().fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved deployment model -> {MODEL_PATH}")
    print(f"Classes: {list(model.classes_)}")


if __name__ == "__main__":
    main()
