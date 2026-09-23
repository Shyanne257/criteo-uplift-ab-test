# Does Ad Exposure Drive Incremental Visits? — A/B Test & Uplift Analysis on 14M Users (PySpark)

> **Business question:** Criteo ran a randomized ad-incrementality test. How many *extra* visits and
> conversions do ads actually cause, which users respond best, and who should we target?

**Status:** 🚧 in progress — Step 1 of 7 done.

| Step | What | Status |
|---|---|---|
| 1 | PySpark ingestion, data-quality profile, Parquet | ✅ |
| 2 | Experiment validity: SRM test, covariate balance | ⏳ |
| 3 | Core A/B analysis: z-test + bootstrap CIs for visit & conversion | ⏳ |
| 4 | Power & minimum detectable effect | ⏳ |
| 5 | Assignment vs. exposure: ITT vs. CACE | ⏳ |
| 6 | Heterogeneous effects & T-learner uplift model (Qini) | ⏳ |
| 7 | Streamlit dashboard & write-up | ⏳ |

## Key results

_Filled in as each step is completed._

## Data

[Criteo Uplift Prediction Dataset v2.1](https://huggingface.co/datasets/criteo/criteo-uplift):
~13.98M users, 12 anonymised features (`f0`–`f11`), `treatment` (≈85% treated),
`visit`, `conversion`, `exposure`. License: CC-BY-NC-SA 4.0 (non-commercial). The data is not
redistributed in this repo — `scripts/00_download_data.sh` fetches it.

## Project structure

```
├── scripts/
│   ├── 00_download_data.sh        # fetch raw data from Hugging Face
│   ├── make_sample_data.py        # tiny synthetic file with the same schema (for quick runs / CI)
│   └── 01_ingest_and_profile.py   # Step 1: PySpark ingest -> profile -> Parquet
├── src/criteo_ab/spark.py         # SparkSession factory + explicit schema
├── reports/                       # generated reports and figures
└── data/{raw,processed}/          # git-ignored
```

## How to run

Requires Python 3.10+ and Java 17 (for Spark).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# real data (~311 MB download)
bash scripts/00_download_data.sh
python scripts/01_ingest_and_profile.py

# or: synthetic sample, runs in ~30s
python scripts/make_sample_data.py
python scripts/01_ingest_and_profile.py --input data/raw/sample.csv.gz
```

### Why Spark for 14M rows?
Pandas could hold this dataset, but the pipeline is written against the Spark DataFrame API so the
same code scales to the original 25M-row release or to production-sized experiment logs without a
rewrite.

## Citation

Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). *A Large Scale Benchmark for Uplift
Modeling.* AdKDD & TargetAd Workshop, KDD 2018.
