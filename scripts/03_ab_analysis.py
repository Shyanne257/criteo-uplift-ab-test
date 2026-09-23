"""Step 3 - Core A/B analysis: visit and conversion lift with z-tests and bootstrap CIs.

For each metric (visit, conversion):
  * rates per group, absolute lift (pp) and relative lift (%)
  * two-proportion z-test: p-value (pooled SE) + 95% CI of the difference (unpooled SE)
  * 95% CI of the relative lift via the delta method on log(p_T / p_C)
  * bootstrap 95% percentile CIs as an independent cross-check

Bootstrap note: outcomes are binary and users are i.i.d., so resampling n users with
replacement gives a success count that is exactly Binomial(n, p_hat). We therefore draw
the bootstrap replicates from that distribution - identical to a row-level bootstrap,
but 10,000 replicates take milliseconds instead of re-scanning 14M rows 10,000 times.

Usage:
    python scripts/03_ab_analysis.py
    python scripts/03_ab_analysis.py --n-boot 10000 --seed 7

Outputs:
    reports/03_ab_results.json
    reports/03_ab_results.md
    reports/figures/03_lift_ci.png
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
import numpy as np  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from scipy import stats  # noqa: E402

from criteo_ab.spark import get_spark  # noqa: E402

DEFAULT_INPUT = ROOT / "data/processed/criteo_uplift.parquet"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures/03_lift_ci.png"
METRICS = ["visit", "conversion"]
Z = stats.norm.ppf(0.975)


def z_test(x_t: int, n_t: int, x_c: int, n_c: int) -> dict:
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    # hypothesis test uses the pooled SE (under H0 both groups share one rate)
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    # the confidence interval uses the unpooled SE
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    # relative lift CI: delta method on the log risk ratio
    se_log = math.sqrt((1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c))
    rr = p_t / p_c
    return {
        "rate_treatment": p_t,
        "rate_control": p_c,
        "abs_lift": diff,
        "abs_lift_ci95": [diff - Z * se, diff + Z * se],
        "rel_lift": rr - 1,
        "rel_lift_ci95": [rr * math.exp(-Z * se_log) - 1, rr * math.exp(Z * se_log) - 1],
        "z": z,
        "p_value": p_value,
    }


def bootstrap(x_t: int, n_t: int, x_c: int, n_c: int, n_boot: int, rng) -> dict:
    p_t = rng.binomial(n_t, x_t / n_t, n_boot) / n_t
    p_c = rng.binomial(n_c, x_c / n_c, n_boot) / n_c
    diff, rel = p_t - p_c, p_t / p_c - 1
    return {
        "n_boot": n_boot,
        "abs_lift_ci95": np.percentile(diff, [2.5, 97.5]).tolist(),
        "rel_lift_ci95": np.percentile(rel, [2.5, 97.5]).tolist(),
        "share_boot_lift_le_0": float((diff <= 0).mean()),
    }


def plot(results: dict, path: Path) -> None:
    """Relative lift with 95% CIs: z-test/delta method vs bootstrap, per metric."""
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = {"z-test (delta method)": "#2a78d6", "bootstrap": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for i, m in enumerate(METRICS):
        r = results[m]
        for j, (label, ci) in enumerate(
            [("z-test (delta method)", r["z_test"]["rel_lift_ci95"]),
             ("bootstrap", r["bootstrap"]["rel_lift_ci95"])]
        ):
            y = i + (0.14 if j == 0 else -0.14)
            est = r["z_test"]["rel_lift"] * 100
            ax.hlines(y, ci[0] * 100, ci[1] * 100, color=colors[label], lw=2,
                      label=label if i == 0 else None)
            ax.scatter(est, y, s=48, color=colors[label], edgecolor=surface, linewidth=2, zorder=3)
        ax.text(r["z_test"]["rel_lift_ci95"][1] * 100, i + 0.14,
                f"  +{r['z_test']['rel_lift']:.1%}", va="center", color=ink, fontsize=9)

    ax.axvline(0, color=muted, lw=1)
    ax.set_yticks(range(len(METRICS)), [m.capitalize() + " rate" for m in METRICS])
    ax.set_ylim(-0.6, len(METRICS) - 0.4)
    ax.set_xlabel("Relative lift, treatment vs control (%) — 95% CI", color=muted, fontsize=9)
    sig = [m for m in METRICS if results[m]["z_test"]["rel_lift_ci95"][0] > 0]
    title = ("Ads significantly raise both visit and conversion rates" if len(sig) == len(METRICS)
             else f"Significant lift only for: {', '.join(sig) or 'none'}")
    ax.set_title(title, loc="left", color=ink, fontsize=11)
    xmax = max(results[m]["bootstrap"]["rel_lift_ci95"][1] for m in METRICS) * 100
    ax.set_xlim(min(0, ax.get_xlim()[0]), xmax * 1.25)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.grid(axis="x", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=muted)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def fmt_ci(ci, pct=True, pp=False):
    if pp:
        return f"[{ci[0] * 100:+.3f}, {ci[1] * 100:+.3f}] pp"
    return f"[{ci[0]:+.2%}, {ci[1]:+.2%}]"


def cross_check_line(results: dict) -> str:
    """Compare analytic vs bootstrap relative-lift CI endpoints (in percentage points of lift)."""
    gaps = {
        m: max(abs(a - b) for a, b in zip(results[m]["z_test"]["rel_lift_ci95"],
                                          results[m]["bootstrap"]["rel_lift_ci95"]))
        for m in METRICS
    }
    detail = ", ".join(f"{m} {g * 100:.2f} pts" for m, g in gaps.items())
    if all(g < 0.05 * abs(results[m]["z_test"]["rel_lift"]) + 0.005 for m, g in gaps.items()):
        return (f"- **Cross-check:** bootstrap and analytic CIs agree (largest endpoint gap: "
                f"{detail}), so the normal approximation holds at this sample size.")
    return (f"- **Cross-check:** bootstrap and analytic CIs differ noticeably ({detail}); "
            "the rarer the event, the more skewed the ratio - prefer the bootstrap CI.")


def render_md(counts: dict, results: dict, n_boot: int, runtime: float) -> str:
    t, c = counts[1], counts[0]
    lines = [
        "# Step 3 - Core A/B results",
        "",
        f"Treatment n = {t['n']:,} · Control n = {c['n']:,} · bootstrap replicates = {n_boot:,}",
        "",
        "| Metric | Control | Treatment | Abs. lift | 95% CI (z) | Rel. lift | 95% CI (delta) "
        "| 95% CI (bootstrap) | z | p-value |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in METRICS:
        z, b = results[m]["z_test"], results[m]["bootstrap"]
        lines.append(
            f"| {m} | {z['rate_control']:.4%} | {z['rate_treatment']:.4%} | "
            f"{z['abs_lift'] * 100:+.3f} pp | {fmt_ci(z['abs_lift_ci95'], pp=True)} | "
            f"**{z['rel_lift']:+.2%}** | {fmt_ci(z['rel_lift_ci95'])} | "
            f"{fmt_ci(b['rel_lift_ci95'])} | {z['z']:.1f} | {z['p_value']:.3g} |"
        )
    v, cv = results["visit"]["z_test"], results["conversion"]["z_test"]
    inc_visits = v["abs_lift"] * t["n"]
    inc_conv = cv["abs_lift"] * t["n"]
    width = lambda r: r["rel_lift_ci95"][1] - r["rel_lift_ci95"][0]  # noqa: E731
    lines += [
        "",
        "## Reading the results",
        "",
        f"- **Visits:** ads lift the visit rate by {v['abs_lift'] * 100:.3f} pp "
        f"({v['rel_lift']:+.1%}). Across the {t['n']:,} treated users that is roughly "
        f"**{inc_visits:,.0f} incremental visits**.",
        f"- **Conversions:** {cv['abs_lift'] * 100:.3f} pp ({cv['rel_lift']:+.1%}), roughly "
        f"**{inc_conv:,.0f} incremental conversions**.",
        f"- **Signal vs noise:** conversion is ~{v['rate_control'] / cv['rate_control']:.0f}x rarer "
        f"than a visit, so its relative-lift CI is {width(cv) / width(v):.1f}x wider "
        f"({width(cv):.1%} vs {width(v):.1%}). Visit is the stronger, more stable signal; "
        "conversion is the business outcome but noisier.",
        cross_check_line(results),
        "- These are **intent-to-treat (ITT)** effects: they compare users *assigned* to see ads, "
        "most of whom never saw one. Step 5 scales them to users actually exposed.",
        "",
        "![Relative lift with 95% CIs](figures/03_lift_ci.png)",
        "",
        f"_Runtime: {runtime:.1f}s_",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()
    spark = get_spark("03-ab")
    df = spark.read.parquet(args.input)

    # one Spark pass: users and successes per group
    agg = df.groupBy("treatment").agg(
        F.count("*").alias("n"), *[F.sum(m).cast("long").alias(m) for m in METRICS]
    ).collect()
    counts = {int(r["treatment"]): r.asDict() for r in agg}
    t, c = counts[1], counts[0]

    rng = np.random.default_rng(args.seed)
    results = {
        m: {
            "z_test": z_test(t[m], t["n"], c[m], c["n"]),
            "bootstrap": bootstrap(t[m], t["n"], c[m], c["n"], args.n_boot, rng),
        }
        for m in METRICS
    }
    plot(results, FIG)
    runtime = time.time() - t0

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "03_ab_results.json").write_text(json.dumps(
        {"counts": {str(k): v for k, v in counts.items()}, "results": results,
         "runtime_sec": round(runtime, 1)}, indent=2))
    (REPORTS / "03_ab_results.md").write_text(render_md(counts, results, args.n_boot, runtime))

    for m in METRICS:
        z = results[m]["z_test"]
        print(f"{m:<10} C={z['rate_control']:.4%} T={z['rate_treatment']:.4%} "
              f"lift={z['rel_lift']:+.2%} p={z['p_value']:.3g}")
    print(f"done in {runtime:.1f}s -> reports/03_ab_results.md")
    spark.stop()


if __name__ == "__main__":
    main()
