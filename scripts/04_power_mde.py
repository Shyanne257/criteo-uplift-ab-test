"""Step 4 - Statistical power and minimum detectable effect (MDE).

Questions answered:
  1. With the full sample, what is the smallest lift we could reliably detect (MDE)?
  2. If the test had run on only 1% (or 0.1%, ...) of the traffic, could we still
     detect the effect we observed? -> analytic power curve
  3. Does the analytic answer hold on real data? -> empirical power: split users at
     random into k disjoint buckets (each = 1/k of traffic), re-run the z-test in
     every bucket, count how often it is significant. One Spark groupBy does all k tests.
  4. What did the 85:15 allocation cost compared with a 50:50 split?

Usage:
    python scripts/04_power_mde.py
    python scripts/04_power_mde.py --alpha 0.05 --power 0.8

Outputs:
    reports/04_power_mde.json
    reports/04_power_mde.md
    reports/figures/04_mde_curve.png
    reports/figures/04_power_curve.png
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
FIG_MDE = REPORTS / "figures/04_mde_curve.png"
FIG_POWER = REPORTS / "figures/04_power_curve.png"
METRICS = ["visit", "conversion"]
FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
EMPIRICAL_BUCKETS = [10, 100, 1000]  # 10%, 1%, 0.1% of traffic

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"visit": "#2a78d6", "conversion": "#eb6834"}


# ---------------------------------------------------------------- formulas --
def mde_abs(p: float, n_t: float, n_c: float, alpha: float, power: float) -> float:
    """Smallest absolute difference detectable with the given power (two-sided test)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * math.sqrt(p * (1 - p) * (1 / n_t + 1 / n_c))


def power_two_prop(p_c: float, p_t: float, n_t: float, n_c: float, alpha: float) -> float:
    """Power of the two-proportion z-test to detect a true difference p_t - p_c."""
    p_bar = (n_t * p_t + n_c * p_c) / (n_t + n_c)
    se0 = math.sqrt(p_bar * (1 - p_bar) * (1 / n_t + 1 / n_c))  # under H0
    se1 = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)  # under H1
    z_a = stats.norm.ppf(1 - alpha / 2)
    d = abs(p_t - p_c)
    return float(stats.norm.cdf((d - z_a * se0) / se1) + stats.norm.cdf((-d - z_a * se0) / se1))


def n_required(p_c: float, p_t: float, share_t: float, alpha: float, power: float) -> float:
    """Total users needed to reach `power` for the true effect, at a given treatment share."""
    lo, hi = 10.0, 1e12
    for _ in range(200):  # bisection on total N (power is monotone in N)
        mid = math.sqrt(lo * hi)
        if power_two_prop(p_c, p_t, mid * share_t, mid * (1 - share_t), alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def z_pvalues(x_t, n_t, x_c, n_c) -> np.ndarray:
    p_t, p_c = x_t / n_t, x_c / n_c
    p = (x_t + x_c) / (n_t + n_c)
    se = np.sqrt(p * (1 - p) * (1 / n_t + 1 / n_c))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (p_t - p_c) / se, 0.0)
    return 2 * stats.norm.sf(np.abs(z))


# ---------------------------------------------------------------- plotting --
def style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def pct_axis(ax):
    ticks = [f for f in FRACTIONS if f in (1.0, 0.1, 0.01, 0.001, 0.5, 0.05, 0.005)]
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{t:.1%}".replace(".0%", "%") for t in ticks])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel("Share of traffic in the test (log scale)", color=MUTED, fontsize=9)


def plot_mde(table, observed, path):
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    style(ax, fig)
    for m in METRICS:
        xs = [r["fraction"] for r in table]
        ys = [r[m]["mde_rel"] * 100 for r in table]
        ax.plot(xs, ys, color=COLORS[m], lw=2, marker="o", ms=4, label=f"{m.capitalize()} MDE")
        ax.axhline(observed[m] * 100, color=COLORS[m], lw=1, ls="--")
        ax.text(FRACTIONS[0], observed[m] * 100, f"observed {m} lift {observed[m]:.0%}  ",
                color=MUTED, fontsize=8, va="bottom", ha="left")
    ax.set_yscale("log")
    ax.set_ylabel("Relative MDE, 80% power (%, log)", color=MUTED, fontsize=9)
    pct_axis(ax)
    ax.set_title("Smaller tests can only detect larger lifts", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_power(table, empirical, target_power, path):
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    style(ax, fig)
    for m in METRICS:
        xs = [r["fraction"] for r in table]
        ys = [r[m]["power_observed_effect"] * 100 for r in table]
        ax.plot(xs, ys, color=COLORS[m], lw=2, label=f"{m.capitalize()} (analytic)")
        ex = [e["fraction"] for e in empirical]
        ey = [e[m]["empirical_power"] * 100 for e in empirical]
        ax.scatter(ex, ey, s=56, facecolor=SURFACE, edgecolor=COLORS[m], linewidth=2, zorder=3,
                   label=f"{m.capitalize()} (empirical, disjoint buckets)")
    ax.axhline(target_power * 100, color=MUTED, lw=1, ls="--")
    ax.text(FRACTIONS[-1], target_power * 100, f" {target_power:.0%} target",
            color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Power to detect the observed lift (%)", color=MUTED, fontsize=9)
    pct_axis(ax)
    ax.set_title("Power to detect the observed effect as traffic shrinks",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- report ----
def render_md(base, table, empirical, alloc, alpha, power, runtime) -> str:
    L = [
        "# Step 4 - Power & minimum detectable effect",
        "",
        f"Two-sided α = {alpha}, target power = {power:.0%}, allocation 85:15 (as run). "
        "Baselines are the control-group rates.",
        "",
        "## 1. MDE with the full sample",
        "",
        "| Metric | Control rate | MDE (abs) | MDE (relative) | Observed lift | Observed / MDE |",
        "|---|---|---|---|---|---|",
    ]
    full = table[0]
    for m in METRICS:
        r = full[m]
        L.append(f"| {m} | {base[m]['p_c']:.4%} | {r['mde_abs'] * 100:.4f} pp | "
                 f"{r['mde_rel']:.2%} | {base[m]['rel_lift']:+.2%} | "
                 f"{base[m]['rel_lift'] / r['mde_rel']:.0f}x |")
    L += [
        "",
        "## 2. What if the test had used less traffic?",
        "",
        "| Traffic | Users | Visit MDE (rel) | Power: visit | Conversion MDE (rel) | Power: conversion |",
        "|---|---|---|---|---|---|",
    ]
    for r in table:
        L.append(f"| {r['fraction']:.1%} | {r['n_total']:,.0f} | {r['visit']['mde_rel']:.1%} | "
                 f"{r['visit']['power_observed_effect']:.1%} | {r['conversion']['mde_rel']:.1%} | "
                 f"{r['conversion']['power_observed_effect']:.1%} |")
    L += [
        "",
        "## 3. Empirical check on the real data",
        "",
        "Users are randomly split into *k* disjoint buckets (each bucket = 1/k of traffic, "
        "same 85:15 split); the z-test is re-run in every bucket in one Spark `groupBy`.",
        "",
        "| Traffic per bucket | Buckets | Visit: share significant | Visit: analytic power "
        "| Conversion: share significant | Conversion: analytic power |",
        "|---|---|---|---|---|---|",
    ]
    for e in empirical:
        L.append(f"| {e['fraction']:.1%} | {e['k']} | {e['visit']['empirical_power']:.1%} | "
                 f"{e['visit']['analytic_power']:.1%} | {e['conversion']['empirical_power']:.1%} | "
                 f"{e['conversion']['analytic_power']:.1%} |")
    L += [
        "",
        "## 4. Cost of the 85:15 allocation",
        "",
        "| Metric | Users needed at 85:15 | Users needed at 50:50 | Ratio |",
        "|---|---|---|---|",
    ]
    for m in METRICS:
        a = alloc[m]
        L.append(f"| {m} | {a['n_85_15']:,.0f} | {a['n_50_50']:,.0f} | "
                 f"{a['n_85_15'] / a['n_50_50']:.2f}x |")
    one = next(r for r in table if abs(r["fraction"] - 0.01) < 1e-9)
    weak = [m for m in METRICS if one[m]["power_observed_effect"] < power]
    if not weak:
        one_verdict = "both effects would still be detected reliably."
    elif len(weak) == len(METRICS):
        one_verdict = "neither effect would be detected reliably at this size."
    else:
        one_verdict = (f"{', '.join(weak)} would be missed too often - the rarer metric needs "
                       "more traffic to reach the same power.")
    # smallest traffic share at which each metric still reaches target power
    min_frac = {m: min((r["fraction"] for r in table if r[m]["power_observed_effect"] >= power),
                       default=None) for m in METRICS}
    L += [
        "",
        "## Takeaways",
        "",
        f"- With {full['n_total']:,.0f} users the test could detect a "
        f"{full['visit']['mde_rel']:.1%} relative lift in visits and "
        f"{full['conversion']['mde_rel']:.1%} in conversions at {power:.0%} power; the observed "
        f"lifts are {base['visit']['rel_lift'] / full['visit']['mde_rel']:.0f}x and "
        f"{base['conversion']['rel_lift'] / full['conversion']['mde_rel']:.0f}x those thresholds.",
        f"- **At 1% of traffic** ({one['n_total']:,.0f} users) power is "
        f"{one['visit']['power_observed_effect']:.0%} for visits and "
        f"{one['conversion']['power_observed_effect']:.0%} for conversions: " + one_verdict,
        "- Smallest traffic share (of those tested) that still reaches "
        f"{power:.0%} power: " + ", ".join(
            f"{m} {min_frac[m]:.1%}" if min_frac[m] else f"{m} none" for m in METRICS) + ".",
        "- Rare metrics drive sample size: to plan a test, size it for the *hardest* metric you "
        "must read (here conversion), not the easiest.",
        f"- An 85:15 split needs ~{alloc['visit']['n_85_15'] / alloc['visit']['n_50_50']:.1f}x the "
        "users of a 50:50 split for the same power "
        "(variance ∝ 1/(0.85·0.15) = 7.84 vs 1/(0.5·0.5) = 4). Unequal splits are chosen to limit "
        "the business cost of the control holdout, and this is the statistical price.",
        "",
        "![MDE curve](figures/04_mde_curve.png)",
        "",
        "![Power curve](figures/04_power_curve.png)",
        "",
        f"_Runtime: {runtime:.1f}s_",
        "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------- main ------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    t0 = time.time()

    spark = get_spark("04-power")
    df = spark.read.parquet(args.input).cache()

    g = {int(r["treatment"]): r.asDict() for r in df.groupBy("treatment").agg(
        F.count("*").alias("n"), *[F.sum(m).cast("long").alias(m) for m in METRICS]).collect()}
    n_t, n_c = g[1]["n"], g[0]["n"]
    share_t = n_t / (n_t + n_c)
    base = {}
    for m in METRICS:
        p_t, p_c = g[1][m] / n_t, g[0][m] / n_c
        base[m] = {"p_t": p_t, "p_c": p_c, "rel_lift": p_t / p_c - 1}

    # analytic MDE & power across traffic fractions (same 85:15 split)
    table = []
    for f in FRACTIONS:
        row = {"fraction": f, "n_total": (n_t + n_c) * f}
        for m in METRICS:
            b = base[m]
            mde = mde_abs(b["p_c"], n_t * f, n_c * f, args.alpha, args.power)
            row[m] = {
                "mde_abs": mde,
                "mde_rel": mde / b["p_c"],
                "power_observed_effect": power_two_prop(b["p_c"], b["p_t"], n_t * f, n_c * f,
                                                        args.alpha),
            }
        table.append(row)

    # empirical power: k disjoint random buckets, all z-tests from ONE groupBy per k
    empirical = []
    for k in EMPIRICAL_BUCKETS:
        agg = (
            df.withColumn("bucket", (F.rand(args.seed) * k).cast("int"))
            .groupBy("bucket")
            .agg(
                F.sum("treatment").alias("n_t"),
                F.sum(1 - F.col("treatment")).alias("n_c"),
                *[F.sum(F.col(m) * F.col("treatment")).alias(f"{m}_t") for m in METRICS],
                *[F.sum(F.col(m) * (1 - F.col("treatment"))).alias(f"{m}_c") for m in METRICS],
            )
            .toPandas()
        )
        entry = {"k": k, "fraction": 1 / k}
        for m in METRICS:
            p = z_pvalues(agg[f"{m}_t"].values, agg["n_t"].values,
                          agg[f"{m}_c"].values, agg["n_c"].values)
            sig = (p < args.alpha) & (agg[f"{m}_t"] / agg["n_t"] > agg[f"{m}_c"] / agg["n_c"])
            entry[m] = {
                "empirical_power": float(sig.mean()),
                "analytic_power": power_two_prop(base[m]["p_c"], base[m]["p_t"],
                                                 n_t / k, n_c / k, args.alpha),
            }
        empirical.append(entry)

    # cost of 85:15 vs 50:50 for detecting the observed effect
    alloc = {
        m: {
            "n_85_15": n_required(base[m]["p_c"], base[m]["p_t"], share_t, args.alpha, args.power),
            "n_50_50": n_required(base[m]["p_c"], base[m]["p_t"], 0.5, args.alpha, args.power),
        }
        for m in METRICS
    }

    observed = {m: base[m]["rel_lift"] for m in METRICS}
    FIG_MDE.parent.mkdir(parents=True, exist_ok=True)
    plot_mde(table, observed, FIG_MDE)
    plot_power(table, empirical, args.power, FIG_POWER)
    runtime = time.time() - t0

    (REPORTS / "04_power_mde.json").write_text(json.dumps(
        {"baseline": base, "by_fraction": table, "empirical": empirical, "allocation": alloc,
         "alpha": args.alpha, "power": args.power, "runtime_sec": round(runtime, 1)}, indent=2))
    (REPORTS / "04_power_mde.md").write_text(
        render_md(base, table, empirical, alloc, args.alpha, args.power, runtime))

    one = next(r for r in table if abs(r["fraction"] - 0.01) < 1e-9)
    for m in METRICS:
        print(f"{m:<10} full-sample MDE={table[0][m]['mde_rel']:.2%}  "
              f"power@1%={one[m]['power_observed_effect']:.1%}")
    print(f"done in {runtime:.1f}s -> reports/04_power_mde.md")
    spark.stop()


if __name__ == "__main__":
    main()
