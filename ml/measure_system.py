"""
System-level performance measurement against the proposal's stated targets:
end-to-end latency < 300 ms, and >= 25 FPS on a standard laptop CPU.

Latency is measured server-side, over the real HTTP chain
(gateway -> preprocessing -> vision -> grammar), because that is the part the
system controls. Client-side MediaPipe extraction is measured separately in
the browser and reported alongside — the two sum to the figure the proposal
called end-to-end.

Run the services first (./start_all.sh), then: python ml/measure_system.py
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from evaluate import load_raw  # noqa: E402

GATEWAY = "http://127.0.0.1:8000"
RESULTS = ROOT / "results" / "system_performance.csv"
N_TRIALS = 30
LATENCY_TARGET_MS = 300
FPS_TARGET = 25


def time_endpoint(path, payload, n=N_TRIALS):
    """Median and p95 wall-clock over n calls. Median, not mean: one GC pause
    or scheduler hiccup should not define the reported latency."""
    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = requests.post(f"{GATEWAY}{path}", json=payload, timeout=30)
        timings.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
    return {
        "median_ms": round(statistics.median(timings), 1),
        "p95_ms": round(sorted(timings)[int(0.95 * len(timings))], 1),
        "min_ms": round(min(timings), 1),
    }


def main():
    raw, _, _ = load_raw()
    word_window = raw[0].tolist()                      # 30 frames, as the word path sends
    letter_window = raw[0][:10].tolist()               # 10 frames, the fingerspell path

    try:
        requests.get(f"{GATEWAY}/health", timeout=5).raise_for_status()
    except Exception:
        print("Gateway not reachable — run ./start_all.sh first.")
        raise SystemExit(1)

    rows = []
    for name, path, payload in [
        ("word_sign (30-frame window)", "/translate", {"keypoints": word_window}),
        ("fingerspell (10-frame window)", "/fingerspell", {"keypoints": letter_window}),
    ]:
        stats = time_endpoint(path, payload)
        # A window is only sent once it is full, so the recogniser must keep up
        # with window arrivals, not with individual frames.
        budget_ms = (30 if "word" in name else 10) / FPS_TARGET * 1000
        rows.append({
            "path": name,
            **stats,
            "target_ms": LATENCY_TARGET_MS,
            "meets_latency_target": stats["median_ms"] < LATENCY_TARGET_MS,
            "window_budget_ms_at_25fps": round(budget_ms, 1),
            "keeps_up_with_25fps": stats["median_ms"] < budget_ms,
        })
        print(f"{name:32s} median {stats['median_ms']:6.1f} ms  p95 {stats['p95_ms']:6.1f} ms  "
              f"{'PASS' if stats['median_ms'] < LATENCY_TARGET_MS else 'FAIL'} (<{LATENCY_TARGET_MS} ms)")

    import pandas as pd
    RESULTS.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"\nsaved {RESULTS}")
    print("Note: server-side only. Browser-side MediaPipe extraction runs per frame "
          "and is reported separately from the live session.")


if __name__ == "__main__":
    main()
