#!/usr/bin/env bash
# Step 0 - download the Criteo Uplift v2.1 dataset (~311 MB) from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
URL="https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
OUT="data/raw/criteo-research-uplift-v2.1.csv.gz"
if [ -f "$OUT" ]; then
  echo "already downloaded: $OUT"
else
  curl -L --fail -o "$OUT" "$URL"
fi
ls -lh "$OUT"
