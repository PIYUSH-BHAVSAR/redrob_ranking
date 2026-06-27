# run.ps1 — Windows one-command reproduction
# Usage: .\scripts\run.ps1

param(
    [string]$Candidates = "data/output/candidates.jsonl",
    [string]$Out = "submission/submission.csv",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Redrob Track 1 — Candidate Ranking System" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Step 1: Install dependencies
Write-Host "`n[1/4] Installing dependencies..." -ForegroundColor Yellow
& $Python -m pip install -r requirements.txt --quiet

# Step 2: Download model (first time only)
Write-Host "`n[2/4] Downloading model (skipped if already exists)..." -ForegroundColor Yellow
& $Python src/download_model.py

# Step 3: Precompute embeddings (one-time, uses GPU if available)
Write-Host "`n[3/4] Precomputing embeddings..." -ForegroundColor Yellow
& $Python src/precompute.py --candidates $Candidates --model-dir models/minilm --skip-existing

# Step 4: Generate submission CSV (CPU only, <5 min)
Write-Host "`n[4/4] Ranking candidates..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
& $Python src/rank.py --candidates $Candidates --out $Out

Write-Host "`nDone. Submission at $Out" -ForegroundColor Green
