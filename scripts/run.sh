#!/bin/bash
# run.sh — Linux/Mac one-command reproduction
# Usage: bash scripts/run.sh [candidates_path] [output_path]

set -e

CANDIDATES="${1:-data/output/candidates.jsonl}"
OUT="${2:-submission/submission.csv}"

echo "============================================"
echo " Redrob Track 1 — Candidate Ranking System"
echo "============================================"

# Step 1: Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt --quiet

# Step 2: Download model (first time only)
echo ""
echo "[2/4] Downloading model (skipped if already exists)..."
python src/download_model.py

# Step 3: Precompute embeddings (one-time, uses GPU if available)
echo ""
echo "[3/4] Precomputing embeddings..."
python src/precompute.py --candidates "$CANDIDATES" --model-dir models/minilm --skip-existing

# Step 4: Generate submission CSV (CPU only, <5 min)
echo ""
echo "[4/4] Ranking candidates..."
mkdir -p "$(dirname "$OUT")"
python src/rank.py --candidates "$CANDIDATES" --out "$OUT"

echo ""
echo "Done. Submission at $OUT"
