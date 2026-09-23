"""Step 5 - Assignment vs. exposure: ITT vs. CACE.

Being assigned to the treatment group does not mean a user saw an ad: only a small
share of treated users were actually exposed, and control users can never be exposed
(one-sided non-compliance). So we report two effects:

  * ITT  (intention-to-treat)  = E[Y | assigned T] - E[Y | assigned C]
        "what does turning the campaign on do, per user targeted?"
  * CACE (complier average causal effect) = ITT / (exposure rate T - exposure rate C)
        "what does actually seeing the ad do, for users who would see it?"
    This is the Wald / instrumental-variable estimator, with random assignment as the
    instrument. Under one-sided non-compliance compliers = exposed treated users, so
    CACE is also the effect of treatment on the treated (ATT).

We also show why the tempting shortcut - comparing exposed users with everyone else -
is biased: exposure is not random, exposed users differ systematically (selection).

Usage:
    python scripts/05_itt_cace.py
Outputs:
    reports/05_itt_cace.json
    reports/05_itt_cace.md
    reports/figures/05_itt_vs_cace.png
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

from criteo_ab.spark import FEATURES, get_spark  # noqa: E402

DEFAULT_INPUT = ROOT / "data/processed/criteo_uplift.parquet"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures/05_itt_vs_cace.png"
METRICS = ["visit", "conversion"]
Z = stats.norm.ppf(0.975)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EST_COLORS = {"ITT": "#2a78d6", "CACE": "#1baf7a", "Naive (exposed vs control)": "#eb6834"}


# ------------------------------------------------------------- estimators --
def estimates(c: dict, m: str) -> dict:
    """All point estimates for metric m from cell counts.

    c holds, for the treatment group, counts by (exposure e, outcome y) and, for control,
    n and successes.
    """
    n_t = c["t_e0_y0"] + c["t_e0_y1"] + c["t_e1_y0"] + c["t_e1_y1"]
    n_e = c["t_e1_y0"] + c["t_e1_y1"]
    y_t = (c["t_e0_y1"] + c["t_e1_y1"]) / n_t
    y_c = c["c_y1"] / c["c_n"]
    y_exp = c["t_e1_y1"] / n_e            # treated & exposed
    y_unexp = c["t_e0_y1"] / (n_t - n_e)  # treated & not exposed ("never-takers")
    pi = n_e / n_t                        # compliance rate (control exposure is 0)
    itt = y_t - y_c
    cace = itt / pi
    y0_compliers = y_exp - cace           # what compliers would have done without the ad
    return {
        "rate_control": y_c,
        "rate_treatment": y_t,
        "rate_treated_exposed": y_exp,
        "rate_treated_unexposed": y_unexp,
        "compliance": pi,
        "itt": itt,
        "itt_rel": itt / y_c,
        "cace": cace,
        "complier_rate_without_ad": y0_compliers,
        "cace_rel": cace / y0_compliers if y0_compliers > 0 else float("nan"),
        "naive_exposed_vs_control": y_exp - y_c,
        "naive_exposed_vs_unexposed": y_exp - y_unexp,
    }


def delta_ci(c: dict, e: dict) -> dict:
    """Analytic CIs. CACE uses the delta method incl. the covariance of ITT and compliance."""
    n_t = c["t_e0_y0"] + c["t_e0_y1"] + c["t_e1_y0"] + c["t_e1_y1"]
    n_c = c["c_n"]
    y_t, y_c, pi = e["rate_treatment"], e["rate_control"], e["compliance"]
    var_itt = y_t * (1 - y_t) / n_t + y_c * (1 - y_c) / n_c
    var_pi = pi * (1 - pi) / n_t
    p_ye = c["t_e1_y1"] / n_t
    cov = (p_ye - y_t * pi) / n_t          # Cov(mean Y_T, mean E_T), same users
    cace = e["cace"]
    var_cace = (var_itt - 2 * cace * cov + cace ** 2 * var_pi) / pi ** 2
    se_itt, se_cace = math.sqrt(var_itt), math.sqrt(var_cace)
    return {
        "itt_ci95": [e["itt"] - Z * se_itt, e["itt"] + Z * se_itt],
        "cace_ci95": [cace - Z * se_cace, cace + Z * se_cace],
    }


def bootstrap(c: dict, m: str, n_boot: int, rng) -> dict:
    """Multinomial bootstrap: equivalent to resampling users, done on the 4 treatment cells."""
    keys = ["t_e0_y0", "t_e0_y1", "t_e1_y0", "t_e1_y1"]
    n_t = sum(c[k] for k in keys)
    probs = np.array([c[k] for k in keys]) / n_t
    draws_t = rng.multinomial(n_t, probs, size=n_boot)
    draws_c = rng.binomial(c["c_n"], c["c_y1"] / c["c_n"], size=n_boot)
    out = {"itt": [], "cace": [], "cace_rel": [], "naive_exposed_vs_control": []}
    for i in range(n_boot):
        cc = dict(zip(keys, draws_t[i].tolist()))
        cc["c_n"], cc["c_y1"] = c["c_n"], int(draws_c[i])
        e = estimates(cc, m)
        for k in out:
            out[k].append(e[k])
    return {f"{k}_ci95": np.nanpercentile(v, [2.5, 97.5]).tolist() for k, v in out.items()}


# ------------------------------------------------------------- plotting ----
def plot(results: dict, path: Path) -> None:
    """Small multiples (one panel per metric, own x-scale): ITT vs CACE vs naive, with CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    labels = list(EST_COLORS)
    for ax, m in zip(axes, METRICS):
        r = results[m]
        e, b = r["estimates"], r["bootstrap"]
        vals = [e["itt"], e["cace"], e["naive_exposed_vs_control"]]
        cis = [b["itt_ci95"], b["cace_ci95"], b["naive_exposed_vs_control_ci95"]]
        ax.set_facecolor(SURFACE)
        for i, (lab, v, ci) in enumerate(zip(labels, vals, cis)):
            y = len(labels) - 1 - i
            ax.hlines(y, ci[0] * 100, ci[1] * 100, color=EST_COLORS[lab], lw=2)
            ax.scatter(v * 100, y, s=48, color=EST_COLORS[lab], edgecolor=SURFACE,
                       linewidth=2, zorder=3)
            ax.text(ci[1] * 100, y, f"  {v * 100:+.2f} pp", va="center", fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(labels)), labels[::-1] if ax is axes[0] else [""] * len(labels))
        ax.set_ylim(-0.6, len(labels) - 0.4)
        right = max(max(ci[1] for ci in cis) * 100, 0)
        ax.set_xlim(min(0, min(ci[0] for ci in cis) * 100) - right * 0.02, right * 1.45)
        ax.set_title(f"{m.capitalize()} rate", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Effect (percentage points), 95% CI", color=MUTED, fontsize=8)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Effect per user assigned (ITT) vs. per user actually exposed (CACE)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------- report ------
def pp(x):
    return f"{x * 100:+.3f} pp"


def ci_pp(ci):
    return f"[{ci[0] * 100:+.3f}, {ci[1] * 100:+.3f}]"


def naive_line(v: dict, cv: dict) -> str:
    rv = v["naive_exposed_vs_control"] / v["cace"]
    rc = cv["naive_exposed_vs_control"] / cv["cace"]
    if max(abs(rv - 1), abs(rc - 1)) < 0.1:
        return (f"- The naive exposed-vs-control gap ({rv:.2f}x visit, {rc:.2f}x conversion of "
                "CACE) is close to the causal effect here, but that is luck, not a guarantee.")
    direction = "overstates" if (rv > 1 and rc > 1) else "understates" if (rv < 1 and rc < 1) \
        else "distorts"
    return (f"- The naive exposed-vs-control comparison {direction} the effect: {rv:.2f}x (visit) "
            f"and {rc:.2f}x (conversion) the causal estimate. The gap is selection bias - who "
            "gets exposed - not ad impact.")


def render_md(counts, results, smd, n_boot, runtime) -> str:
    pi = results["visit"]["estimates"]["compliance"]
    L = [
        "# Step 5 - Assignment vs. exposure: ITT vs. CACE",
        "",
        "## Compliance",
        "",
        f"- Treated users actually exposed to an ad: **{pi:.2%}** "
        f"({counts['n_exposed']:,} of {counts['n_t']:,}).",
        f"- Control users exposed: {counts['c_exposed']:,} → one-sided non-compliance, so there "
        "are no 'always-takers' and CACE = effect on the treated (ATT).",
        "",
        "## Effects",
        "",
        "| Metric | ITT | ITT 95% CI | CACE | CACE 95% CI (delta) | CACE 95% CI (bootstrap) "
        "| Exposed users' rate | Same users without ad (est.) | CACE relative lift |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in METRICS:
        e, d, b = results[m]["estimates"], results[m]["delta"], results[m]["bootstrap"]
        L.append(
            f"| {m} | {pp(e['itt'])} | {ci_pp(d['itt_ci95'])} | **{pp(e['cace'])}** | "
            f"{ci_pp(d['cace_ci95'])} | {ci_pp(b['cace_ci95'])} | {e['rate_treated_exposed']:.3%} | "
            f"{e['complier_rate_without_ad']:.3%} | **{e['cace_rel']:+.0%}** "
            f"[{b['cace_rel_ci95'][0]:+.0%}, {b['cace_rel_ci95'][1]:+.0%}] |"
        )
    L += [
        "",
        "## Why not just compare exposed users with everyone else?",
        "",
        "| Metric | Naive: exposed − control | Naive: exposed − unexposed treated | CACE (causal) "
        "| Naive / CACE |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        e = results[m]["estimates"]
        L.append(f"| {m} | {pp(e['naive_exposed_vs_control'])} | "
                 f"{pp(e['naive_exposed_vs_unexposed'])} | {pp(e['cace'])} | "
                 f"{e['naive_exposed_vs_control'] / e['cace']:.2f}x |")
    top = sorted(smd, key=lambda r: -abs(r["smd"]))[:5]
    n_imb = sum(abs(r["smd"]) >= 0.1 for r in smd)
    L += [
        "",
        "Exposure is **not** randomised - it is likely driven by user behaviour (e.g. how active "
        "a user is on sites where ads are served). Exposed vs. unexposed treated users on f0-f11:",
        "",
        "| feature | SMD exposed vs unexposed |",
        "|---|---|",
        *[f"| {r['feature']} | {r['smd']:+.3f} |" for r in top],
        "",
        f"{n_imb}/{len(smd)} features have |SMD| ≥ 0.1 (vs. 0/12 between randomised groups in "
        "Step 2). Exposed users are a different population, so the naive comparison mixes the "
        "ad effect with who gets exposed. CACE avoids this by only comparing *randomised* groups "
        "and rescaling by the compliance rate.",
        "",
        "## Takeaways",
        "",
    ]
    v, cv = results["visit"]["estimates"], results["conversion"]["estimates"]
    L += [
        f"- **ITT** answers the campaign question: turning ads on raises visits by "
        f"{v['itt'] * 100:.2f} pp and conversions by {cv['itt'] * 100:.3f} pp *per targeted user*.",
        f"- **CACE** answers the creative/media question: for users who actually saw an ad, visits "
        f"rise by {v['cace'] * 100:.1f} pp ({v['cace_rel']:+.0%} vs. their own counterfactual) "
        f"and conversions by {cv['cace'] * 100:.2f} pp ({cv['cace_rel']:+.0%}).",
        f"- CACE = ITT ÷ {pi:.3f}, so it is ~{1 / pi:.0f}x the ITT - and its CI is ~{1 / pi:.0f}x "
        "wider too: dividing by a small compliance rate amplifies noise.",
        naive_line(v, cv),
        "- Assumptions: (1) random assignment (checked in Step 2); (2) **exclusion restriction** - "
        "assignment affects outcomes only through exposure (plausible, but violated if unlogged "
        "impressions exist); (3) monotonicity - trivially true since control cannot be exposed.",
        "",
        "![ITT vs CACE](figures/05_itt_vs_cace.png)",
        "",
        f"_Bootstrap replicates: {n_boot:,} · Runtime: {runtime:.1f}s_",
        "",
    ]
    return "\n".join(L)


# ------------------------------------------------------------- main --------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    spark = get_spark("05-cace")
    df = spark.read.parquet(args.input).cache()

    # one pass: cell counts by (treatment, exposure, visit, conversion)
    cells = {
        (int(r.treatment), int(r.exposure), int(r.visit), int(r.conversion)): int(r.n)
        for r in df.groupBy("treatment", "exposure", "visit", "conversion")
        .agg(F.count("*").alias("n")).collect()
    }

    def total(**kw):
        names = ["treatment", "exposure", "visit", "conversion"]
        return sum(n for k, n in cells.items()
                   if all(k[names.index(a)] == v for a, v in kw.items()))

    per_metric_counts = {}
    for m in METRICS:
        per_metric_counts[m] = {
            "t_e0_y0": total(treatment=1, exposure=0, **{m: 0}),
            "t_e0_y1": total(treatment=1, exposure=0, **{m: 1}),
            "t_e1_y0": total(treatment=1, exposure=1, **{m: 0}),
            "t_e1_y1": total(treatment=1, exposure=1, **{m: 1}),
            "c_n": total(treatment=0),
            "c_y1": total(treatment=0, **{m: 1}),
        }
    counts = {"n_t": total(treatment=1), "n_c": total(treatment=0),
              "n_exposed": total(treatment=1, exposure=1),
              "c_exposed": total(treatment=0, exposure=1)}

    rng = np.random.default_rng(args.seed)
    results = {}
    for m in METRICS:
        c = per_metric_counts[m]
        e = estimates(c, m)
        results[m] = {"counts": c, "estimates": e, "delta": delta_ci(c, e),
                      "bootstrap": bootstrap(c, m, args.n_boot, rng)}

    # selection check: exposed vs unexposed within the treatment group (one Spark agg)
    aggs = []
    for f in FEATURES:
        aggs += [F.avg(f).alias(f"{f}_m"), F.var_samp(f).alias(f"{f}_v")]
    g = {int(r["exposure"]): r.asDict()
         for r in df.filter("treatment = 1").groupBy("exposure").agg(*aggs).collect()}
    smd = []
    for f in FEATURES:
        sd = math.sqrt((g[1][f"{f}_v"] + g[0][f"{f}_v"]) / 2)
        smd.append({"feature": f, "smd": (g[1][f"{f}_m"] - g[0][f"{f}_m"]) / sd if sd else 0.0})

    plot(results, FIG)
    runtime = time.time() - t0
    (REPORTS / "05_itt_cace.json").write_text(json.dumps(
        {"counts": counts, "results": results, "exposure_smd": smd,
         "runtime_sec": round(runtime, 1)}, indent=2))
    (REPORTS / "05_itt_cace.md").write_text(render_md(counts, results, smd, args.n_boot, runtime))

    for m in METRICS:
        e = results[m]["estimates"]
        print(f"{m:<10} ITT={e['itt'] * 100:+.3f}pp  CACE={e['cace'] * 100:+.2f}pp "
              f"({e['cace_rel']:+.0%})  compliance={e['compliance']:.2%}")
    print(f"done in {runtime:.1f}s -> reports/05_itt_cace.md")
    spark.stop()


if __name__ == "__main__":
    main()
