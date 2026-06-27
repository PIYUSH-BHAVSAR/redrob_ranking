# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

## Team

- Piyush Bhavsar — your@email.com

---

## One-Command Reproduction

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download model (first time only)
```bash
python src/download_model.py
```

### Step 3 — Precompute embeddings (one-time, not counted in 5-min budget)
```bash
python src/precompute.py --candidates data/output/candidates.jsonl --model-dir models/minilm --skip-existing
```

### Step 4 — Rank (produces submission CSV, runs in <5 min on CPU)
```bash
python src/rank.py --candidates data/output/candidates.jsonl --out submission/submission.csv
```

---

## System Requirements

- Python 3.10+
- 16 GB RAM
- CPU only for ranking step
- GPU optional for precompute (significantly faster)
- **Runtime:** precompute ~16 min (GPU) / ranking ~46 sec (CPU)

---

## Architecture Overview

### Problem

Rank the top 100 best-fit candidates from a 100K pool for a Senior AI Engineer role
focused on retrieval, ranking, and embedding systems.

### Approach

#### 1. JD Parsing

The job description is split into 3 sections — responsibilities, requirements, full text —
and each is embedded separately. This enables targeted similarity scoring:
- Responsibilities section → what the person will actually DO
- Requirements section → what they need to HAVE
- Full text → overall career arc match

#### 2. Feature Groups (6 total)

| Feature | Weight | What it captures |
|---|---|---|
| Responsibility similarity | 0.30 | Per-role max cosine sim vs JD responsibilities |
| Narrative similarity | 0.20 | Career arc vs full JD |
| Skills score | 0.12 | Proficiency + endorsement + assessment trust |
| Domain keyword coverage | 0.08 | JD keyword hits in career text |
| Trajectory score | 0.18 | Exp gate + seniority + product company score |
| Behavioral score | 0.07 | Activity decay + response rate + GitHub |
| Location + notice | 0.05 | Pune/Noida preference, notice period |

#### 3. Key Design Decisions

**Per-role max aggregation** — solves the Tier-5 problem. A candidate who built a
recommendation system at role 2 but doesn't use RAG buzzwords still scores high on
responsibility similarity because we embed each role description separately and take
the max similarity, not the mean.

**Tiered experience gate** — not binary. `3-5 yrs = 0.3x`, `5-9 yrs = 1.0x`,
`9-12 yrs = 0.85x`, `12+ yrs = 0.6x`. Reflects the JD's nuance that the 5-9 band
is ideal but outliers can still qualify.

**Time-weighted product company score** — months spent at product vs services companies,
not just the current employer. Penalizes pure services careers; doesn't penalize someone
who spent 2 years at TCS then 5 at Flipkart.

**Assessment score trust multiplier** — candidates with verified Redrob assessment scores
on JD-relevant skills get a trust boost; unverified claims get a 0.7x discount.

**Behavioral signals as multiplier** — activity decay (`exp(-days/90)`) means a
6-month-inactive candidate loses ~85% of behavioral score regardless of profile quality.

#### 4. Trap Detection (2 layers)

**Layer 1 — Hard deterministic rules:**
1. Timeline anomaly — sum of role months > claimed experience + buffer
2. Perfect signal cluster — 3+ behavioral signals at theoretical max
3. Expert-zero-endorsement — 8+ expert skills, 0 endorsements
4. Keyword stuffer — AI buzzwords in surface text with thin career descriptions
5. Impossible experience — more years than since graduation date
6. Job function mismatch — non-technical career (HR/Marketing/Ops) + inflated AI skill claims

**Layer 2 — IsolationForest** statistical anomaly detection on behavioral feature matrix.
Bottom 0.3% flagged as statistical outliers.

Candidates with 2+ flags → score pushed to 0.03-0.07 (bottom of pool).
Candidates with 1 flag → 0.80x soft penalty.

#### 5. Compute Design

- **Precompute:** all embeddings saved as `.npy` files — one-time cost
- **Rank step:** pure numpy dot products, no model inference at rank time
- **Rank step runtime:** ~46 seconds on CPU
- **No external API calls** at any stage

---

## Project Structure

```
redrob-ranking/
├── src/
│   ├── parse_jd.py          ← structured JD extraction
│   ├── features.py          ← all 6 feature groups
│   ├── trap_detector.py     ← honeypot + keyword stuffer detection
│   ├── scorer.py            ← composite scoring formula
│   ├── reasoner.py          ← evidence-first reasoning generation
│   ├── precompute.py        ← GPU embedding pre-computation
│   ├── rank.py              ← CPU ranking step (≤5 min)
│   └── download_model.py    ← one-time model download
├── submission/
│   └── submission.csv       ← final ranked output
├── scripts/
│   ├── run.sh               ← Linux/Mac reproduction
│   └── run.ps1              ← Windows reproduction
├── README.md
├── requirements.txt
└── submission_metadata.yaml
```

---

## AI Tools Used

- **Claude (Anthropic)** — architecture review, code review, debugging

No candidate data was fed to any LLM. All scoring logic, feature weights, and
trap detection rules were implemented and validated manually.

---

## Compute Environment

Windows 11, Intel i7-12700H, 16GB RAM, NVIDIA RTX 3050 6GB Laptop GPU, Python 3.10.11
