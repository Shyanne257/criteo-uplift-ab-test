"""Step 6 - Heterogeneous treatment effects and a T-learner uplift model.

Part A (PySpark, full data) - segment analysis
    Every feature f0-f11 is cut into quantile buckets. In each bucket we estimate the ITT
    lift for visit and conversion. Cochran's Q tests whether the lift really differs across
    buckets. Because we run many tests, p-values are corrected (Bonferroni for the per-feature
    Q tests, Benjamini-Hochberg for the per-bucket tests).

Part B (sample + LightGBM) - T-learner uplift model
    On a random sample: train one classifier on treated users, one on control users,
    uplift(x) = P(y | x, T) - P(y | x, C). Users in a held-out test set are ranked by
    predicted uplift and evaluated with Qini curves: "if we only target the top k% of
    users, what share of all incremental visits/conversions do we keep?"

Usage:
    python scripts/06_heterogeneity_uplift.py
    python scripts/06_heterogeneity_uplift.py --sample-rows 2000000 --n-buckets 5

Outputs:
    reports/06_heterogeneity_uplift.json
    reports/06_heterogeneity_uplift.md
    reports/figures/06_segment_lift.png
    reports/figures/06_qini_curves.png
    reports/figures/06_uplift_by_decile.png
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

import lightgbm as lgb  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from criteo_ab.spark import FEATURES, get_spark  # noqa: E402

DEFAULT_INPUT = ROOT / "data/processed/criteo_uplift.parquet"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
METRICS = ["visit", "conversion"]
TARGET_SHARES = [0.1, 0.2, 0.3, 0.5]
Z = stats.norm.ppf(0.975)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"visit": "#2a78d6", "conversion": "#eb6834", "random": "#8a8983"}


# =========================================================== Part A: segments
def bucket_edges(df, n_buckets: int) -> dict:
    probs = [i / n_buckets for i in range(1, n_buckets)]
    qs = df.approxQuantile(FEATURES, probs, 0.001)
    # many features are discretised -> identical quantiles; keep unique edges only
    return {f: sorted(set(q)) for f, q in zip(FEATURES, qs)}


def bucket_col(f: str, edges: list[float]):
    """Bucket index: 0 for x <= e0, 1 for e0 < x <= e1, ..."""
    expr = F.lit(len(edges))
    for i in reversed(range(len(edges))):
        expr = F.when(F.col(f) <= edges[i], F.lit(i)).otherwise(expr)
    return expr


def segment_table(df, edges: dict) -> pd.DataFrame:
    """One Spark job: explode (feature, bucket) pairs, aggregate per bucket x group."""
    pairs = F.array(*[
        F.struct(F.lit(f).alias("feature"), bucket_col(f, edges[f]).alias("bucket"))
        for f in FEATURES
    ])
    long = df.select("treatment", *METRICS, F.explode(pairs).alias("seg")).select(
        "treatment", *METRICS, "seg.feature", "seg.bucket")
    agg = long.groupBy("feature", "bucket").agg(
        F.sum("treatment").alias("n_t"),
        F.sum(1 - F.col("treatment")).alias("n_c"),
        *[F.sum(F.col(m) * F.col("treatment")).alias(f"{m}_t") for m in METRICS],
        *[F.sum(F.col(m) * (1 - F.col("treatment"))).alias(f"{m}_c") for m in METRICS],
    ).toPandas()

    rows = []
    for _, r in agg.iterrows():
        e = edges[r.feature]
        b = int(r.bucket)
        lo = "-inf" if b == 0 else f"{e[b - 1]:.3g}"
        hi = "+inf" if b == len(e) else f"{e[b]:.3g}"
        row = {"feature": r.feature, "bucket": b, "range": f"({lo}, {hi}]",
               "n": int(r.n_t + r.n_c), "n_t": int(r.n_t), "n_c": int(r.n_c)}
        for m in METRICS:
            p_t, p_c = r[f"{m}_t"] / r.n_t, r[f"{m}_c"] / r.n_c
            se = math.sqrt(p_t * (1 - p_t) / r.n_t + p_c * (1 - p_c) / r.n_c)
            pool = (r[f"{m}_t"] + r[f"{m}_c"]) / (r.n_t + r.n_c)
            se0 = math.sqrt(pool * (1 - pool) * (1 / r.n_t + 1 / r.n_c))
            row.update({
                f"{m}_rate_c": p_c, f"{m}_rate_t": p_t,
                f"{m}_lift": p_t - p_c, f"{m}_se": se,
                f"{m}_rel": (p_t / p_c - 1) if p_c > 0 else np.nan,
                f"{m}_p": 2 * stats.norm.sf(abs(p_t - p_c) / se0) if se0 > 0 else 1.0,
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feature", "bucket"]).reset_index(drop=True)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1)
    return out


def heterogeneity_tests(seg: pd.DataFrame) -> pd.DataFrame:
    """Cochran's Q per feature and metric: do bucket-level lifts differ more than chance?"""
    out = []
    for f, g in seg.groupby("feature"):
        row = {"feature": f, "n_buckets": len(g)}
        for m in METRICS:
            w = 1 / g[f"{m}_se"] ** 2
            d = g[f"{m}_lift"]
            d_bar = (w * d).sum() / w.sum()
            q = float((w * (d - d_bar) ** 2).sum())
            p = float(stats.chi2.sf(q, len(g) - 1)) if len(g) > 1 else 1.0
            row.update({f"{m}_Q": q, f"{m}_p": p,
                        f"{m}_lift_range": float(d.max() - d.min())})
        out.append(row)
    het = pd.DataFrame(out)
    n_tests = len(het) * len(METRICS)
    for m in METRICS:
        het[f"{m}_p_bonf"] = np.minimum(het[f"{m}_p"] * n_tests, 1)
    return het.sort_values("visit_Q", ascending=False).reset_index(drop=True)


# =========================================================== Part B: T-learner
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_t_learner(train: pd.DataFrame, y: str, seed: int):
    models = {}
    for t in (0, 1):
        part = train[train.treatment == t]
        clf = lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)
        clf.fit(part[FEATURES], part[y])
        models[t] = clf
    return models


def predict_uplift(models, X: pd.DataFrame) -> np.ndarray:
    return models[1].predict_proba(X)[:, 1] - models[0].predict_proba(X)[:, 1]


def qini_curve(uplift: np.ndarray, y: np.ndarray, t: np.ndarray, n_points: int = 101):
    """Cumulative incremental outcomes when targeting the top-phi share by predicted uplift.

    Q(phi) = Y_T(phi) - Y_C(phi) * N_T(phi) / N_C(phi)
    """
    order = np.argsort(-uplift, kind="stable")
    y, t = y[order], t[order]
    cy_t, cy_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    cn_t, cn_c = np.cumsum(t), np.cumsum(1 - t)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(cn_c[idx] > 0, cy_t[idx] - cy_c[idx] * cn_t[idx] / cn_c[idx], 0.0)
    phi = (idx + 1) / len(y)
    return np.concatenate([[0], phi]), np.concatenate([[0], q])


def qini_summary(phi, q) -> dict:
    total = q[-1]
    area_model = float(np.sum((q[1:] + q[:-1]) / 2 * np.diff(phi)))  # trapezoid rule
    area_random = total / 2
    capture = {f"{s:.0%}": float(np.interp(s, phi, q) / total) if total else float("nan")
               for s in TARGET_SHARES}
    return {"total_incremental": float(total),
            "qini_coefficient": float((area_model - area_random) / abs(total)) if total else 0.0,
            "capture": capture}


def bootstrap_capture(uplift, y, t, n_boot, rng) -> dict:
    """CIs for 'share of incremental outcomes captured at top k%' by resampling test users."""
    n = len(y)
    res = {f"{s:.0%}": [] for s in TARGET_SHARES}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        phi, q = qini_curve(uplift[i], y[i], t[i])
        if q[-1] <= 0:
            continue
        for s in TARGET_SHARES:
            res[f"{s:.0%}"].append(np.interp(s, phi, q) / q[-1])
    return {k: np.percentile(v, [2.5, 97.5]).tolist() if v else [np.nan, np.nan]
            for k, v in res.items()}


def uplift_by_decile(uplift, y, t) -> pd.DataFrame:
    dec = pd.qcut(pd.Series(uplift).rank(method="first", ascending=False), 10, labels=False)
    rows = []
    for d in range(10):
        m = (dec == d).values
        yt, yc = y[m & (t == 1)], y[m & (t == 0)]
        lift = yt.mean() - yc.mean()
        se = math.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
        rows.append({"decile": d + 1, "predicted_uplift": float(uplift[m].mean()),
                     "observed_uplift": float(lift), "ci_lo": lift - Z * se,
                     "ci_hi": lift + Z * se, "n": int(m.sum())})
    return pd.DataFrame(rows)


# =========================================================== plotting
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_segments(seg, het, path):
    top = het.head(3).feature.tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(9, 3.2), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, f in zip(np.atleast_1d(axes), top):
        style(ax)
        g = seg[seg.feature == f]
        x = np.arange(len(g))
        ax.bar(x, g.visit_lift * 100, width=0.6, color=COLORS["visit"], edgecolor=SURFACE,
               linewidth=2)
        ax.errorbar(x, g.visit_lift * 100, yerr=Z * g.visit_se * 100, fmt="none",
                    ecolor=INK, elinewidth=1, capsize=3)
        ax.set_xticks(x, [f"Q{b + 1}" for b in g.bucket])
        ax.set_title(f"{f}", loc="left", color=INK, fontsize=10)
        ax.axhline(0, color=MUTED, lw=1)
    np.atleast_1d(axes)[0].set_ylabel("Visit lift (pp), 95% CI", color=MUTED, fontsize=9)
    fig.suptitle("Visit lift by feature bucket - the 3 most heterogeneous features",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_qini(curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes, METRICS):
        style(ax)
        phi, q, summ = curves[m]
        total = q[-1]
        ax.plot(phi * 100, q / total * 100, color=COLORS[m], lw=2, label="T-learner")
        ax.plot([0, 100], [0, 100], color=COLORS["random"], lw=1.5, ls="--", label="Random")
        c30 = summ["capture"]["30%"] * 100
        ax.scatter([30], [c30], s=48, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"top 30% → {c30:.0f}%", (30, c30), xytext=(8, -14),
                    textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlabel("Share of users targeted, ranked by predicted uplift (%)",
                      color=MUTED, fontsize=8)
        ax.set_title(f"{m.capitalize()} — Qini coef. {summ['qini_coefficient']:.3f}",
                     loc="left", color=INK, fontsize=10)
        ax.set_xlim(0, 100)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower right")
    axes[0].set_ylabel("% of all incremental outcomes captured", color=MUTED, fontsize=9)
    fig.suptitle("Qini curves on held-out users", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_deciles(dec_tables, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes, METRICS):
        style(ax)
        d = dec_tables[m]
        x = d.decile.values
        ax.bar(x, d.observed_uplift * 100, width=0.65, color=COLORS[m], edgecolor=SURFACE,
               linewidth=2, label="Observed (test set)")
        ax.errorbar(x, d.observed_uplift * 100,
                    yerr=[(d.observed_uplift - d.ci_lo) * 100, (d.ci_hi - d.observed_uplift) * 100],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        ax.scatter(x, d.predicted_uplift * 100, s=40, marker="D", color=SURFACE,
                   edgecolor=INK, linewidth=1.5, zorder=3, label="Predicted")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x)
        ax.set_xlabel("Decile of predicted uplift (1 = highest)", color=MUTED, fontsize=8)
        ax.set_title(f"{m.capitalize()}", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")
    axes[0].set_ylabel("Uplift (pp), 95% CI", color=MUTED, fontsize=9)
    fig.suptitle("Observed vs predicted uplift by decile - the model ranks users correctly "
                 "if bars fall from left to right", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# =========================================================== report
def render_md(seg, het, model_res, meta) -> str:
    L = ["# Step 6 - Heterogeneous effects & uplift model", "",
         "## Part A - Segment analysis (full data, PySpark)", "",
         f"Each feature cut into up to {meta['n_buckets']} quantile buckets "
         "(fewer when a feature has few distinct values). ITT lift per bucket; "
         "Cochran's Q tests whether lifts differ across buckets.", "",
         "| feature | buckets | Visit Q | Visit p (Bonferroni) | Visit lift range (pp) "
         "| Conversion Q | Conversion p (Bonferroni) | Conversion lift range (pp) |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in het.iterrows():
        L.append(f"| {r.feature} | {r.n_buckets} | {r.visit_Q:,.0f} | {r.visit_p_bonf:.2g} | "
                 f"{r.visit_lift_range * 100:.2f} | {r.conversion_Q:,.0f} | "
                 f"{r.conversion_p_bonf:.2g} | {r.conversion_lift_range * 100:.3f} |")
    n_sig_v = int((het.visit_p_bonf < 0.05).sum())
    n_sig_c = int((het.conversion_p_bonf < 0.05).sum())
    L += ["", f"- Features with significant heterogeneity after Bonferroni ({len(het) * 2} tests): "
          f"**{n_sig_v}/{len(het)}** for visit, **{n_sig_c}/{len(het)}** for conversion.", ""]

    for f in het.head(3).feature:
        g = seg[seg.feature == f]
        L += [f"**{f}** - lift by bucket", "",
              "| bucket | range | users | visit C → T | visit lift | conv. C → T | conv. lift "
              "| BH-adj. p (visit) |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| Q{r.bucket + 1} | {r.range} | {r.n:,} | {r.visit_rate_c:.2%} → "
                     f"{r.visit_rate_t:.2%} | {r.visit_lift * 100:+.2f} pp | "
                     f"{r.conversion_rate_c:.3%} → {r.conversion_rate_t:.3%} | "
                     f"{r.conversion_lift * 100:+.3f} pp | {r.visit_p_bh:.2g} |")
        L.append("")
    n_tests = len(seg) * len(METRICS)
    n_raw = int((seg.visit_p < 0.05).sum() + (seg.conversion_p < 0.05).sum())
    n_bh = int((seg.visit_p_bh < 0.05).sum() + (seg.conversion_p_bh < 0.05).sum())
    L += ["**Multiple comparisons.** "
          f"{n_tests} bucket-level tests were run. {n_raw} have raw p < 0.05, {n_bh} survive "
          "Benjamini-Hochberg (FDR 5%). With this many cuts some 'significant' segments are "
          "expected by chance alone (~5% of true nulls), so segment findings are treated as "
          "hypotheses and confirmed with a held-out model evaluation below.", "",
          "![Segment lift](figures/06_segment_lift.png)", "",
          "## Part B - T-learner uplift model", "",
          f"- Random sample of {meta['n_sample']:,} users, split 50/50 into train "
          f"({meta['n_train']:,}) and held-out test ({meta['n_test']:,}), stratified by treatment.",
          "- Two LightGBM classifiers per outcome (treated / control); "
          "uplift(x) = P(y=1 | x, T) − P(y=1 | x, C).",
          "- Evaluation on the test set only. Qini coefficient = area between model and random "
          "Qini curves, normalised by total incremental outcomes (0 = random).", "",
          "| Outcome | Qini coef. | Top 10% | Top 20% | **Top 30%** | Top 50% |",
          "|---|---|---|---|---|---|"]
    for m in METRICS:
        s, ci = model_res[m]["summary"], model_res[m]["capture_ci95"]
        cells = [f"{s['capture'][k]:.0%} [{ci[k][0]:.0%}, {ci[k][1]:.0%}]" for k in
                 ["10%", "20%", "30%", "50%"]]
        cells[2] = f"**{cells[2]}**"
        L.append(f"| {m} | {s['qini_coefficient']:.3f} | " + " | ".join(cells) + " |")
    L += ["", "Cells = share of all incremental outcomes captured by targeting only that top "
          "share of users (bootstrap 95% CI). Random targeting would capture exactly the share "
          "targeted (10%, 20%, ...).", ""]
    for m in METRICS:
        d = pd.DataFrame(model_res[m]["deciles"])
        L += [f"**{m.capitalize()} uplift by decile (test set)**", "",
              "| decile | predicted | observed | 95% CI |", "|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {int(r.decile)} | {r.predicted_uplift * 100:+.3f} pp | "
                     f"{r.observed_uplift * 100:+.3f} pp | [{r.ci_lo * 100:+.3f}, "
                     f"{r.ci_hi * 100:+.3f}] |")
        L.append("")
    v30 = model_res["visit"]["summary"]["capture"]["30%"]
    c30 = model_res["conversion"]["summary"]["capture"]["30%"]
    c30ci = model_res["conversion"]["capture_ci95"]["30%"]
    dv = pd.DataFrame(model_res["visit"]["deciles"])
    L += ["## Takeaways", "",
          f"- **Targeting recommendation:** serving ads only to the 30% of users with the highest "
          f"predicted uplift captures **{c30:.0%}** of incremental conversions (95% CI "
          f"{c30ci[0]:.0%}–{c30ci[1]:.0%}) and **{v30:.0%}** of incremental visits, versus 30% "
          "under random targeting - at 30% of the media cost.",
          *([f"- A capture above 100% means the remaining users have zero or *negative* "
             "estimated uplift in aggregate: showing them ads adds cost without adding "
             "outcomes."] if max(c30, v30) > 1 else []),
          "- The effect is ITT (per assigned user), which is exactly the lever a targeting "
          "policy controls: who is *eligible* for ads.",
          f"- **Use the model for ranking, not for absolute values.** Top-decile predicted "
          f"visit uplift is {dv.predicted_uplift.iloc[0] * 100:+.2f} pp vs "
          f"{dv.observed_uplift.iloc[0] * 100:+.2f} pp observed. T-learner predictions are "
          "over-dispersed because the two models' errors do not cancel; the Qini curve, which "
          "only depends on the ranking, is the right evaluation.",
          "- Caveats: features are anonymised, so segments cannot be named in business terms; "
          "the model is trained on a sample; conversion uplift is estimated from few control "
          "conversions and is the noisier of the two.", "",
          "![Qini curves](figures/06_qini_curves.png)", "",
          "![Uplift by decile](figures/06_uplift_by_decile.png)", "",
          f"_Runtime: {meta['runtime_sec']:.0f}s_", ""]
    return "\n".join(L)


# =========================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--n-buckets", type=int, default=5)
    ap.add_argument("--sample-rows", type=int, default=2_000_000)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()

    spark = get_spark("06-uplift")
    df = spark.read.parquet(args.input).cache()
    n_total = df.count()

    # ---- Part A
    print("Part A: segment analysis ...")
    edges = bucket_edges(df, args.n_buckets)
    seg = segment_table(df, edges)
    for m in METRICS:
        seg[f"{m}_p_bh"] = benjamini_hochberg(seg[f"{m}_p"].values)
    het = heterogeneity_tests(seg)

    # ---- Part B
    print("Part B: T-learner ...")
    frac = min(1.0, args.sample_rows / n_total)
    # Spark writes the sample to Parquet and pandas reads it back: fast, and avoids the
    # Arrow/JVM compatibility issues that toPandas() can hit on newer Java versions.
    sample_path = ROOT / "data/processed/uplift_sample.parquet"
    df.sample(fraction=frac, seed=args.seed).select(*FEATURES, "treatment", *METRICS) \
        .write.mode("overwrite").parquet(str(sample_path))
    sample = pd.read_parquet(sample_path)
    train, test = train_test_split(sample, test_size=0.5, random_state=args.seed,
                                   stratify=sample.treatment)
    rng = np.random.default_rng(args.seed)
    model_res, curves, deciles = {}, {}, {}
    for m in METRICS:
        models = fit_t_learner(train, m, args.seed)
        u = predict_uplift(models, test[FEATURES])
        y, t = test[m].values, test.treatment.values
        phi, q = qini_curve(u, y, t)
        summ = qini_summary(phi, q)
        ci = bootstrap_capture(u, y, t, args.n_boot, rng)
        dec = uplift_by_decile(u, y, t)
        model_res[m] = {"summary": summ, "capture_ci95": ci, "deciles": dec.to_dict("records"),
                        "qini_curve": {"phi": phi.tolist(), "q": q.tolist()}}
        curves[m] = (phi, q, summ)
        deciles[m] = dec
        print(f"  {m:<10} Qini={summ['qini_coefficient']:.3f}  "
              f"top30%→{summ['capture']['30%']:.0%} of incremental")

    FIGS.mkdir(parents=True, exist_ok=True)
    plot_segments(seg, het, FIGS / "06_segment_lift.png")
    plot_qini(curves, FIGS / "06_qini_curves.png")
    plot_deciles(deciles, FIGS / "06_uplift_by_decile.png")

    meta = {"n_buckets": args.n_buckets, "n_sample": len(sample), "n_train": len(train),
            "n_test": len(test), "runtime_sec": time.time() - t0}
    (REPORTS / "06_heterogeneity_uplift.json").write_text(json.dumps(
        {"meta": meta, "edges": edges, "segments": seg.to_dict("records"),
         "heterogeneity": het.to_dict("records"), "model": model_res}, indent=2, default=float))
    (REPORTS / "06_heterogeneity_uplift.md").write_text(render_md(seg, het, model_res, meta))
    print(f"done in {meta['runtime_sec']:.0f}s -> reports/06_heterogeneity_uplift.md")
    spark.stop()


if __name__ == "__main__":
    main()
