"""Generate a small SYNTHETIC file with the same schema as the Criteo Uplift data.

It lets anyone (and CI) run the whole pipeline in seconds without downloading
the 311 MB real file. Numbers from this file are NOT real results.

Usage:
    python scripts/make_sample_data.py --rows 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data/raw/sample.csv.gz"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.rows
    X = rng.normal(size=(n, 12))
    X[:, [1, 5, 9]] = np.round(X[:, [1, 5, 9]] * 2) / 2  # a few discretised features
    treatment = (rng.random(n) < 0.85).astype(int)

    # baseline visit probability ~4%, ad lift concentrated in users with high f0
    logit = -3.2 + 0.4 * X[:, 2] - 0.3 * X[:, 7]
    uplift = treatment * (0.15 + 0.35 * (X[:, 0] > 0.5))
    p_visit = 1 / (1 + np.exp(-(logit + uplift)))
    visit = (rng.random(n) < p_visit).astype(int)
    conversion = (visit * (rng.random(n) < 0.06 + 0.02 * treatment)).astype(int)
    exposure = (treatment * (rng.random(n) < 0.036)).astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]).round(6)
    df["treatment"], df["conversion"], df["visit"], df["exposure"] = treatment, conversion, visit, exposure
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="gzip")
    print(f"wrote {n:,} synthetic rows -> {args.out}")


if __name__ == "__main__":
    main()
