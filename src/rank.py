"""
rank.py — Main ranking step. Must complete in ≤5 minutes on CPU.

Loads all pre-computed artifacts from data/processed/ and produces
the submission CSV. Does NOT call any model — only numpy dot products.

Usage:
  python src/rank.py --candidates data/raw/candidates.jsonl --out data/output/submission.csv

Requirements:
  - data/processed/ must contain all artifacts from precompute.py
  - No network calls
  - No GPU
  - ≤ 5 min wall-clock
"""

import argparse
import csv
import json
import gzip
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from features import (
    responsibility_similarity,
    narrative_similarity,
    requirements_similarity,
    skills_score,
    domain_keyword_coverage,
    trajectory_features,
    behavioral_features,
    location_score,
    notice_score,
    achievement_depth,
)
from scorer import score_candidate, rank_candidates
from reasoner import build_reasoning

PROCESSED_DIR = Path("data/processed")


def load_candidates(path: str) -> list:
    path = Path(path)
    candidates = []
    opener = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" \
             else open(path, "r", encoding="utf-8")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_artifacts():
    """Load all pre-computed numpy artifacts."""
    print("Loading pre-computed artifacts...")

    candidate_ids = np.load(PROCESSED_DIR / "candidate_ids.npy")
    jd_embs_npz = np.load(PROCESSED_DIR / "jd_embs.npz")
    jd_embs = {
        "resp": jd_embs_npz["resp"],
        "full": jd_embs_npz["full"],
        "req":  jd_embs_npz["req"],
    }

    pct_npz = np.load(PROCESSED_DIR / "percentiles.npz")
    percentiles = {
        "github_activity_score":      pct_npz["github_activity_score"],
        "recruiter_response_rate":    pct_npz["recruiter_response_rate"],
        "interview_completion_rate":  pct_npz["interview_completion_rate"],
        "offer_acceptance_rate":      pct_npz["offer_acceptance_rate"],
    }

    trap_scores = np.load(PROCESSED_DIR / "trap_scores.npy")
    narrative_embs = np.load(PROCESSED_DIR / "narrative_embs.npy")
    profile_embs = np.load(PROCESSED_DIR / "profile_embs.npy")
    role_embs_arr = np.load(PROCESSED_DIR / "role_embs.npy")
    role_mask = np.load(PROCESSED_DIR / "role_embs_mask.npy")

    print(f"  Loaded embeddings: narrative={narrative_embs.shape}, "
          f"profile={profile_embs.shape}, role={role_embs_arr.shape}")

    return (candidate_ids, jd_embs, percentiles, trap_scores,
            narrative_embs, profile_embs, role_embs_arr, role_mask)


def get_role_embs_for_candidate(i: int, role_embs_arr: np.ndarray,
                                role_mask: np.ndarray) -> list:
    """Extract valid role embeddings for candidate i as a list."""
    mask = role_mask[i]   # shape (MAX_ROLES,)
    valid = role_embs_arr[i][mask]  # shape (n_valid_roles, dim)
    return list(valid)    # list of 1-D arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t0 = time.time()

    # ── Load candidates ──
    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    N = len(candidates)
    print(f"  {N:,} candidates loaded")

    # ── Load artifacts ──
    (candidate_ids, jd_embs, percentiles, trap_scores,
     narrative_embs, profile_embs, role_embs_arr, role_mask) = load_artifacts()

    # Build ID → index map for verification
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

    # ── Score all candidates ──
    print(f"\nScoring {N:,} candidates...")
    scored = []

    for i, candidate in enumerate(tqdm(candidates, desc="Scoring")):
        cid = candidate["candidate_id"]

        # Resolve index (handles any ordering differences)
        idx = id_to_idx.get(cid, i)

        role_embs = get_role_embs_for_candidate(idx, role_embs_arr, role_mask)

        result = score_candidate(
            candidate=candidate,
            idx=idx,
            role_embs=role_embs,
            narrative_emb=narrative_embs[idx],
            profile_emb=profile_embs[idx],
            jd_embs=jd_embs,
            percentiles=percentiles,
            trap_score=int(trap_scores[idx]),
        )

        scored.append({
            "candidate_id": cid,
            "score": result["score"],
            "components": result,
            "candidate": candidate,
        })

    # ── Rank ──
    print("\nRanking...")
    ranked = rank_candidates(scored)
    top100 = ranked[:100]

    # ── Generate reasoning for top 100 ──
    print("Generating reasoning for top 100...")
    rows = []
    for rank_pos, item in enumerate(top100, start=1):
        if item["components"].get("is_disqualified"):
            reasoning = "Excluded: profile flagged by automated quality checks."
        else:
            reasoning = build_reasoning(
                candidate=item["candidate"],
                score_components=item["components"],
                rank=rank_pos,
            )
        rows.append({
            "candidate_id": item["candidate_id"],
            "rank": rank_pos,
            "score": item["score"],
            "reasoning": reasoning,
        })

    # ── Validate before writing ──
    _validate(rows, set(candidate_ids))

    # ── Write CSV ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n✓ Submission written to {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Top-5 scores: {[r['score'] for r in rows[:5]]}")
    print(f"  Rank 100 score: {rows[-1]['score']}")
    trap_in_top100 = sum(1 for item in top100 if item["components"].get("trap_score", 0) >= 2)
    print(f"  Traps in top 100: {trap_in_top100} (limit: 10)")

    # Print top 10 for quick sanity check
    print("\n── Top 10 ──")
    for r in rows[:10]:
        c = next(x for x in top100 if x["candidate_id"] == r["candidate_id"])
        title = c["candidate"]["profile"].get("current_title", "?")
        yrs = c["candidate"]["profile"].get("years_of_experience", 0)
        print(f"  #{r['rank']:3d}  {r['candidate_id']}  score={r['score']}  "
              f"{title} | {yrs}y")


def _validate(rows: list, valid_ids: set):
    """Pre-write validation — raises on any spec violation."""
    n = len(rows)
    assert n == 100 or n == len(valid_ids), \
        f"Expected 100 rows (or all candidates for test runs), got {n}"
    # For actual submission, enforce exactly 100
    if n != 100:
        print(f"  ⚠ Test run: {n} candidates (full submission requires exactly 100)")

    ranks = [r["rank"] for r in rows]
    expected_ranks = list(range(1, n + 1))
    assert ranks == expected_ranks, f"Ranks must be 1–{n} exactly"

    ids = [r["candidate_id"] for r in rows]
    assert len(set(ids)) == n, "Duplicate candidate_ids detected"
    assert all(cid in valid_ids for cid in ids), "Unknown candidate_id(s) in top 100"

    scores = [r["score"] for r in rows]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], \
            f"Score not non-increasing: rank {i+1} ({scores[i]}) > rank {i+2} ({scores[i+1]})"

    print("  ✓ All validation checks passed")


if __name__ == "__main__":
    main()
