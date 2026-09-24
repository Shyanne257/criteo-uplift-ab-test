"""Step 7 - Build the static, interactive results dashboard (GitHub Pages).

Reads the JSON reports written by Steps 1-6, keeps only what the page needs, and
injects it into an HTML template. The result is ONE self-contained file with no
server and no external requests, so it never "sleeps" and loads instantly.

Usage:
    python scripts/07_build_dashboard.py
    # then commit docs/ and enable GitHub Pages: Settings -> Pages -> main branch, /docs

Output:
    docs/index.html
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TEMPLATE = ROOT / "src/criteo_ab/dashboard_template.html"
OUT = ROOT / "docs/index.html"
METRICS = ["visit", "conversion"]


def load(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        raise SystemExit(f"missing {path} - run the step that creates it first")
    return json.loads(path.read_text())


def clean(obj):
    """Round floats (smaller file) and turn NaN/inf into null (valid JSON)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(f"{obj:.6g}")
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def build_data() -> dict:
    p1 = load("01_data_profile.json")
    p2 = load("02_validity.json")
    p3 = load("03_ab_results.json")
    p4 = load("04_power_mde.json")
    p5 = load("05_itt_cace.json")
    p6 = load("06_heterogeneity_uplift.json")

    ab = {}
    for m in METRICS:
        z, b = p3["results"][m]["z_test"], p3["results"][m]["bootstrap"]
        ab[m] = {"rate_c": z["rate_control"], "rate_t": z["rate_treatment"],
                 "abs": z["abs_lift"], "abs_ci": z["abs_lift_ci95"],
                 "rel": z["rel_lift"], "rel_ci": z["rel_lift_ci95"],
                 "rel_ci_boot": b["rel_lift_ci95"], "z": z["z"], "p": z["p_value"]}

    cace = {"compliance": p5["results"]["visit"]["estimates"]["compliance"]}
    for m in METRICS:
        e, b, d = (p5["results"][m][k] for k in ("estimates", "bootstrap", "delta"))
        cace[m] = {"itt": e["itt"], "itt_ci": d["itt_ci95"],
                   "cace": e["cace"], "cace_ci": b["cace_ci95"],
                   "naive": e["naive_exposed_vs_control"],
                   "naive_ci": b["naive_exposed_vs_control_ci95"],
                   "rate_exposed": e["rate_treated_exposed"],
                   "rate_without_ad": e["complier_rate_without_ad"],
                   "cace_rel": e["cace_rel"]}

    uplift = {}
    for m in METRICS:
        r = p6["model"][m]
        q, phi = r["qini_curve"]["q"], r["qini_curve"]["phi"]
        total = q[-1] or 1
        uplift[m] = {"qini_coef": r["summary"]["qini_coefficient"],
                     "capture": r["summary"]["capture"], "capture_ci": r["capture_ci95"],
                     "curve": [[x, y / total] for x, y in zip(phi, q)],
                     "deciles": r["deciles"]}

    segments = {}
    for s in p6["segments"]:
        segments.setdefault(s["feature"], []).append({
            "bucket": s["bucket"], "range": s["range"], "n": s["n"],
            **{f"{m}_{k}": s[f"{m}_{k}"] for m in METRICS
               for k in ("rate_c", "rate_t", "lift", "se", "p_bh")}})
    het = [{"feature": h["feature"], "n_buckets": h["n_buckets"],
            **{f"{m}_{k}": h[f"{m}_{k}"] for m in METRICS
               for k in ("Q", "p_bonf", "lift_range")}} for h in p6["heterogeneity"]]

    return clean({
        "profile": {"n": p1["rows_written"], "groups": p1["group_counts"],
                    "means": p1["overall_means"],
                    "dup_share": p1["duplicate_rows_full"] / p1["rows_read"],
                    "nulls": sum(p1["null_counts"].values()),
                    "control_exposed": p1["consistency"]["control_exposed"]},
        "srm": p2["srm"],
        "balance": [{"feature": r["feature"], "smd": r["smd"], "vr": r["variance_ratio"],
                     "p": r["welch_p"]} for r in p2["balance"]],
        "ab": ab,
        "power": {"alpha": p4["alpha"], "target": p4["power"],
                  "rows": [{"f": r["fraction"], "n": r["n_total"],
                            **{f"{m}_mde": r[m]["mde_rel"] for m in METRICS},
                            **{f"{m}_power": r[m]["power_observed_effect"] for m in METRICS}}
                           for r in p4["by_fraction"]],
                  "empirical": [{"f": e["fraction"], "k": e["k"],
                                 **{f"{m}_emp": e[m]["empirical_power"] for m in METRICS},
                                 **{f"{m}_ana": e[m]["analytic_power"] for m in METRICS}}
                                for e in p4["empirical"]],
                  "allocation": p4["allocation"]},
        "cace": cace,
        "uplift": uplift,
        "segments": segments,
        "heterogeneity": het,
        "sample_rows": p6["meta"]["n_sample"],
    })


def main() -> None:
    data = build_data()
    html = TEMPLATE.read_text()
    marker = "/*__DATA__*/null"
    if marker not in html:
        raise SystemExit("data marker not found in template")
    html = html.replace(marker, json.dumps(data, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    (OUT.parent / ".nojekyll").write_text("")  # serve the file as-is on GitHub Pages
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
