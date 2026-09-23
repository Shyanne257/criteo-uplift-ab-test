"""Step 2 - Experiment validity: Sample Ratio Mismatch (SRM) + covariate balance.

Before reading ANY treatment effect we check that randomisation worked:
  1. SRM: does the observed treatment/control split match the designed 85:15?
  2. Covariate balance: are pre-treatment features f0-f11 distributed the same
     in both groups? Measured with the standardised mean difference (SMD).

Usage:
    python scripts/02_validity_checks.py
    python scripts/02_validity_checks.py --design-share 0.85

Outputs:
    reports/02_validity.json
    reports/02_validity.md
    reports/figures/02_covariate_balance.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from scipy import stats  # noqa: E402

from criteo_ab.spark import FEATURES, get_spark  # noqa: E402

DEFAULT_INPUT = ROOT / "data/processed/criteo_uplift.parquet"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures/02_covariate_balance.png"

SRM_ALPHA = 0.001  # industry convention: SRM alarms use a strict threshold
SMD_THRESHOLD = 0.10  # common rule of thumb for "meaningful" imbalance


def srm_test(n_t: int, n_c: int, design_share: float) -> dict:
    """Chi-square goodness-of-fit test of observed counts vs the designed split."""
    n = n_t + n_c
    expected = [design_share * n, (1 - design_share) * n]
    chi2, p = stats.chisquare([n_t, n_c], f_exp=expected)
    share = n_t / n
    se = math.sqrt(share * (1 - share) / n)
    return {
        "n_treatment": n_t,
        "n_control": n_c,
        "design_share": design_share,
        "observed_share": share,
        "observed_share_ci95": [share - 1.96 * se, share + 1.96 * se],
        "chi2": float(chi2),
        "p_value": float(p),
        "alpha": SRM_ALPHA,
        "srm_detected": bool(p < SRM_ALPHA),
    }


def balance_table(spark_df) -> list[dict]:
    """SMD, variance ratio and Welch t-test for every feature, from ONE Spark aggregation."""
    aggs = []
    for f in FEATURES:
        aggs += [F.avg(f).alias(f"{f}__mean"), F.var_samp(f).alias(f"{f}__var")]
    rows = {
        int(r["treatment"]): r.asDict()
        for r in spark_df.groupBy("treatment").agg(F.count("*").alias("n"), *aggs).collect()
    }
    t, c = rows[1], rows[0]

    out = []
    for f in FEATURES:
        mt, mc = t[f"{f}__mean"], c[f"{f}__mean"]
        vt, vc = t[f"{f}__var"], c[f"{f}__var"]
        pooled_sd = math.sqrt((vt + vc) / 2)
        smd = (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0
        # Welch t-test from summary statistics (no need to pull 14M rows to Python)
        _, p = stats.ttest_ind_from_stats(
            mt, math.sqrt(vt), t["n"], mc, math.sqrt(vc), c["n"], equal_var=False
        )
        out.append(
            {
                "feature": f,
                "mean_treatment": mt,
                "mean_control": mc,
                "smd": smd,
                "variance_ratio": vt / vc if vc > 0 else float("nan"),
                "welch_p": float(p),
                "balanced": abs(smd) < SMD_THRESHOLD,
            }
        )
    return out


def plot_balance(table: list[dict], path: Path) -> None:
    """Love plot: |SMD| per feature with the 0.1 threshold."""
    rows = sorted(table, key=lambda r: abs(r["smd"]))
    names = [r["feature"] for r in rows]
    vals = [abs(r["smd"]) for r in rows]
    xmax = max(0.12, max(vals) * 1.15)

    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.hlines(names, 0, vals, color=grid, lw=2, zorder=1)
    ax.scatter(vals, names, s=48, color=blue, edgecolor="#fcfcfb", linewidth=2, zorder=3)
    ax.axvline(SMD_THRESHOLD, color=muted, lw=1, ls="--", zorder=2)
    ax.text(SMD_THRESHOLD, len(names) - 0.4, f" imbalance threshold ({SMD_THRESHOLD})",
            color=muted, fontsize=8, va="bottom")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.set_xlabel("|Standardised mean difference|, treatment vs control", color=muted, fontsize=9)
    ax.set_title("Covariate balance: every feature is far below the 0.1 threshold"
                 if all(v < SMD_THRESHOLD for v in vals)
                 else "Covariate balance: some features exceed the 0.1 threshold",
                 loc="left", color=ink, fontsize=11)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.grid(axis="x", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_md(srm: dict, table: list[dict], runtime: float) -> str:
    verdict = "**SRM detected - stop and debug before analysing results.**" if srm["srm_detected"] \
        else "No SRM: the observed split is consistent with the design."
    lo, hi = srm["observed_share_ci95"]
    lines = [
        "# Step 2 - Experiment validity",
        "",
        "## 1. Sample Ratio Mismatch (SRM)",
        "",
        "| | Treatment | Control |",
        "|---|---|---|",
        f"| Observed users | {srm['n_treatment']:,} | {srm['n_control']:,} |",
        f"| Designed share | {srm['design_share']:.2%} | {1 - srm['design_share']:.2%} |",
        f"| Observed share | {srm['observed_share']:.5%} | {1 - srm['observed_share']:.5%} |",
        "",
        f"- 95% CI of observed treatment share: [{lo:.5%}, {hi:.5%}]",
        f"- Chi-square = {srm['chi2']:.4g}, p = {srm['p_value']:.4g} (alarm threshold p < {SRM_ALPHA})",
        f"- {verdict}",
        "",
        "## 2. Covariate balance (f0-f11)",
        "",
        f"SMD = (mean_T - mean_C) / sqrt((var_T + var_C) / 2); |SMD| < {SMD_THRESHOLD} is treated as balanced.",
        "",
        "| feature | mean T | mean C | SMD | var ratio | Welch p | balanced |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['feature']} | {r['mean_treatment']:.4f} | {r['mean_control']:.4f} | "
            f"{r['smd']:+.4f} | {r['variance_ratio']:.3f} | {r['welch_p']:.3g} | "
            f"{'yes' if r['balanced'] else '**no**'} |"
        )
    max_smd = max(abs(r["smd"]) for r in table)
    n_sig = sum(r["welch_p"] < 0.05 for r in table)
    lines += [
        "",
        f"- Largest |SMD|: **{max_smd:.4f}**; features balanced: "
        f"{sum(r['balanced'] for r in table)}/{len(table)}",
        f"- Features with Welch p < 0.05: {n_sig}/{len(table)}. With millions of users even a "
        "trivial difference becomes 'significant', which is why balance is judged on the "
        "effect size (SMD), not the p-value.",
        "",
        "![Covariate balance](figures/02_covariate_balance.png)",
        "",
        f"_Runtime: {runtime:.1f}s_",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--design-share", type=float, default=0.85,
                    help="designed share of users in treatment")
    args = ap.parse_args()

    t0 = time.time()
    spark = get_spark("02-validity")
    df = spark.read.parquet(args.input).cache()

    counts = {int(r["treatment"]): int(r["n"])
              for r in df.groupBy("treatment").agg(F.count("*").alias("n")).collect()}
    srm = srm_test(counts[1], counts[0], args.design_share)
    table = balance_table(df)
    plot_balance(table, FIG)
    runtime = time.time() - t0

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "02_validity.json").write_text(
        json.dumps({"srm": srm, "balance": table, "runtime_sec": round(runtime, 1)}, indent=2))
    (REPORTS / "02_validity.md").write_text(render_md(srm, table, runtime))

    print(f"SRM: share={srm['observed_share']:.5%}  p={srm['p_value']:.4g}  "
          f"detected={srm['srm_detected']}")
    print(f"max |SMD| = {max(abs(r['smd']) for r in table):.4f}")
    print(f"done in {runtime:.1f}s -> reports/02_validity.md")
    spark.stop()


if __name__ == "__main__":
    main()
