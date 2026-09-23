"""Step 1 - Ingest the raw Criteo Uplift CSV with PySpark, profile it, write clean Parquet.

Usage:
    python scripts/01_ingest_and_profile.py                      # real data
    python scripts/01_ingest_and_profile.py --input data/raw/sample.csv.gz   # sample

Outputs:
    data/processed/criteo_uplift.parquet   cleaned data, used by every later step
    reports/01_data_profile.json           machine-readable profile
    reports/01_data_profile.md             human-readable profile
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyspark.sql import functions as F  # noqa: E402

from criteo_ab.spark import BINARY_COLS, FEATURES, SCHEMA, get_spark  # noqa: E402

DEFAULT_INPUT = ROOT / "data/raw/criteo-research-uplift-v2.1.csv.gz"
OUTPUT = ROOT / "data/processed/criteo_uplift.parquet"
REPORTS = ROOT / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    args = ap.parse_args()

    t0 = time.time()
    spark = get_spark("01-ingest")

    # ---- 1. Read with an explicit schema --------------------------------------
    raw = spark.read.csv(args.input, header=True, schema=SCHEMA, mode="PERMISSIVE")
    # A .gz file is not splittable -> Spark reads it with a single task.
    # Repartition once so every downstream step runs in parallel.
    raw = raw.repartition(16).cache()
    n_rows = raw.count()
    print(f"rows read: {n_rows:,}")

    # ---- 2. Missing values (one pass, all columns) ----------------------------
    nulls_row = raw.select(
        [F.sum(F.col(c).isNull().cast("long")).alias(c) for c in raw.columns]
    ).first().asDict()

    # ---- 3. Binary columns must be exactly {0, 1} -----------------------------
    binary_values = {
        c: sorted(r[c] for r in raw.select(c).distinct().collect() if r[c] is not None)
        for c in BINARY_COLS
    }

    # ---- 4. Logical consistency checks ----------------------------------------
    consistency = raw.agg(
        # control users should never see the ad
        F.sum(((F.col("treatment") == 0) & (F.col("exposure") == 1)).cast("long")).alias("control_exposed"),
        # a conversion without a visit is suspicious
        F.sum(((F.col("conversion") == 1) & (F.col("visit") == 0)).cast("long")).alias("conversion_without_visit"),
    ).first().asDict()

    # ---- 5. Duplicates ---------------------------------------------------------
    n_distinct_rows = raw.distinct().count()
    n_distinct_features = raw.select(FEATURES).distinct().count()

    # ---- 6. Distributions ------------------------------------------------------
    group_counts = {
        int(r["treatment"]): int(r["n"])
        for r in raw.groupBy("treatment").agg(F.count("*").alias("n")).collect()
    }
    binary_means = raw.agg(*[F.avg(c).alias(c) for c in BINARY_COLS]).first().asDict()

    feat_summary = (
        raw.select(FEATURES)
        .summary("mean", "stddev", "min", "25%", "50%", "75%", "max")
        .toPandas()
        .set_index("summary")
        .astype(float)
    )
    n_unique_feat = raw.agg(
        *[F.approx_count_distinct(c).alias(c) for c in FEATURES]
    ).first().asDict()

    # ---- 7. Clean + write Parquet ---------------------------------------------
    # Only rows with a missing value in any column are dropped.
    # Duplicate rows are KEPT: each row is a different user, and anonymised
    # features can legitimately collide. Dropping them would bias the rates.
    clean = raw.dropna(how="any")
    n_clean = clean.count()
    clean.write.mode("overwrite").parquet(str(OUTPUT))

    # ---- 8. Report -------------------------------------------------------------
    profile = {
        "input": str(args.input),
        "rows_read": n_rows,
        "rows_written": n_clean,
        "rows_dropped_null": n_rows - n_clean,
        "null_counts": {k: int(v) for k, v in nulls_row.items()},
        "binary_values": binary_values,
        "consistency": {k: int(v) for k, v in consistency.items()},
        "duplicate_rows_full": n_rows - n_distinct_rows,
        "duplicate_feature_vectors": n_rows - n_distinct_features,
        "group_counts": group_counts,
        "treatment_share": group_counts.get(1, 0) / n_rows,
        "overall_means": {k: float(v) for k, v in binary_means.items()},
        "feature_approx_unique": {k: int(v) for k, v in n_unique_feat.items()},
        "feature_summary": feat_summary.round(4).to_dict(),
        "runtime_sec": round(time.time() - t0, 1),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "01_data_profile.json").write_text(json.dumps(profile, indent=2))
    (REPORTS / "01_data_profile.md").write_text(render_md(profile, feat_summary, n_unique_feat))
    print(f"done in {profile['runtime_sec']}s -> {OUTPUT}")
    spark.stop()


def render_md(p: dict, feat_summary, n_unique) -> str:
    n = p["rows_read"]
    lines = [
        "# Step 1 - Data profile",
        "",
        f"- Rows read: **{n:,}**; rows written to Parquet: **{p['rows_written']:,}** "
        f"(dropped for missing values: {p['rows_dropped_null']:,})",
        f"- Treatment group: {p['group_counts'].get(1, 0):,} ({p['treatment_share']:.2%}); "
        f"control group: {p['group_counts'].get(0, 0):,}",
        f"- Overall visit rate {p['overall_means']['visit']:.4%}, conversion rate "
        f"{p['overall_means']['conversion']:.4%}, exposure rate {p['overall_means']['exposure']:.4%}",
        "",
        "## Data quality checks",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Missing values (all columns) | {sum(p['null_counts'].values()):,} |",
    ]
    for c, vals in p["binary_values"].items():
        ok = "OK" if vals == [0, 1] else "UNEXPECTED"
        lines.append(f"| `{c}` values | {vals} {ok} |")
    lines += [
        f"| Control users with exposure = 1 | {p['consistency']['control_exposed']:,} |",
        f"| Conversions without a visit | {p['consistency']['conversion_without_visit']:,} |",
        f"| Fully duplicated rows | {p['duplicate_rows_full']:,} ({p['duplicate_rows_full'] / n:.2%}) |",
        f"| Duplicated feature vectors | {p['duplicate_feature_vectors']:,} ({p['duplicate_feature_vectors'] / n:.2%}) |",
        "",
        "Duplicates are kept on purpose: each row is a distinct user and the features are",
        "anonymised/discretised, so identical rows are expected and are not data errors.",
        "",
        "## Feature summary",
        "",
        "| feature | mean | std | min | p25 | median | p75 | max | ~unique |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in FEATURES:
        s = feat_summary[f]
        lines.append(
            f"| {f} | {s['mean']:.3f} | {s['stddev']:.3f} | {s['min']:.3f} | {s['25%']:.3f} | "
            f"{s['50%']:.3f} | {s['75%']:.3f} | {s['max']:.3f} | {n_unique[f]:,} |"
        )
    lines += ["", f"_Runtime: {p['runtime_sec']}s_", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
