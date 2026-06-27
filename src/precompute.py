"""
precompute.py — One-time pre-computation of all embeddings.

Runs ONCE before the competition clock. Saves:
  data/processed/role_embs.npy        — shape (N, max_roles, emb_dim), padded
  data/processed/role_embs_mask.npy   — bool mask of valid role slots
  data/processed/narrative_embs.npy   — shape (N, emb_dim)
  data/processed/profile_embs.npy     — shape (N, emb_dim)
  data/processed/jd_embs.npz         — resp, full, req embeddings
  data/processed/candidate_ids.npy    — candidate_id strings in order
  data/processed/percentiles.npz     — behavioral signal percentile arrays
  data/processed/trap_scores.npy      — int array of trap scores

This file does NOT count toward the 5-min ranking budget.
The ranking step (rank.py) only loads these cached files.

Usage:
  python src/precompute.py --candidates data/raw/candidates.jsonl
"""

import argparse
import json
import gzip
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from parse_jd import get_jd
from features import (
    build_role_texts,
    build_career_narrative_text,
    build_profile_text,
    compute_behavioral_percentiles,
)
from trap_detector import compute_all_trap_scores

# ── Config ──
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~90MB, fast on CPU
MAX_ROLES = 5          # embed up to 5 most recent roles per candidate
BATCH_SIZE = 256       # safe for 6GB VRAM with MiniLM
PROCESSED_DIR = Path("data/processed")


def load_candidates(path: str) -> list:
    """Load candidates from .jsonl or .jsonl.gz"""
    path = Path(path)
    print(f"Loading candidates from {path}...")
    candidates = []

    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, "r", encoding="utf-8")

    with opener as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates):,} candidates")
    return candidates


def embed_texts_batched(texts: list, model, desc: str = "Encoding") -> np.ndarray:
    """Encode a flat list of texts in batches. Returns (N, dim) array."""
    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i: i + BATCH_SIZE]
        embs = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
        )
        all_embs.append(embs)
    return np.vstack(all_embs).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz]")
    parser.add_argument("--model-dir", default="models/minilm", help="Local model path")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stages whose output files already exist (resume mode)")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def should_skip(filename):
        if not args.skip_existing:
            return False
        exists = (PROCESSED_DIR / filename).exists()
        if exists:
            print(f"  ↩ Skipping {filename} (already exists)")
        return exists

    # ── Load model — use GPU if available ──
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model_path = Path(args.model_dir)
    if model_path.exists():
        print(f"Loading model from {model_path}...")
        model = SentenceTransformer(str(model_path), device=device)
    else:
        print(f"Downloading model {MODEL_NAME} (will save to {model_path})...")
        model = SentenceTransformer(MODEL_NAME, device=device)
        model.save(str(model_path))
        print(f"Model saved to {model_path}")

    # ── Load candidates ──
    candidates = load_candidates(args.candidates)
    N = len(candidates)

    # ── Save candidate IDs ──
    candidate_ids = np.array([c["candidate_id"] for c in candidates])
    np.save(PROCESSED_DIR / "candidate_ids.npy", candidate_ids)
    print(f"Saved {N} candidate IDs")

    # ── JD embeddings ──
    if not should_skip("jd_embs.npz"):
        print("\nEmbedding JD sections...")
        jd = get_jd()
        jd_texts = [jd["responsibilities_text"], jd["full_text"], jd["requirements_text"]]
        jd_embs_arr = model.encode(jd_texts, normalize_embeddings=True)
        np.savez(
            PROCESSED_DIR / "jd_embs.npz",
            resp=jd_embs_arr[0].astype(np.float32),
            full=jd_embs_arr[1].astype(np.float32),
            req=jd_embs_arr[2].astype(np.float32),
        )
        print("JD embeddings saved")

    # ── Behavioral percentiles ──
    if not should_skip("percentiles.npz"):
        print("\nComputing behavioral percentiles...")
        percentiles = compute_behavioral_percentiles(candidates)
        np.savez(
            PROCESSED_DIR / "percentiles.npz",
            github_activity_score=percentiles["github_activity_score"],
            recruiter_response_rate=percentiles["recruiter_response_rate"],
            interview_completion_rate=percentiles["interview_completion_rate"],
            offer_acceptance_rate=percentiles["offer_acceptance_rate"],
        )
        print("Percentiles saved")

    # ── Trap scores ──
    if not should_skip("trap_scores.npy"):
        print("\nComputing trap scores...")
        percentiles_loaded = dict(np.load(PROCESSED_DIR / "percentiles.npz"))
        trap_scores = compute_all_trap_scores(candidates, percentiles_loaded)
        np.save(PROCESSED_DIR / "trap_scores.npy", np.array(trap_scores, dtype=np.int8))
        print(f"Trap scores saved. Traps (2+ flags): {sum(1 for t in trap_scores if t >= 2)}")

    # ── Single-pass text building for all embedding types ──
    print(f"\nBuilding all texts for {N:,} candidates in single pass...")
    narrative_texts = []
    profile_texts = []
    flat_texts = []
    flat_indices = []

    for i, c in enumerate(tqdm(candidates, desc="Building texts")):
        narrative_texts.append(build_career_narrative_text(c))
        profile_texts.append(build_profile_text(c))
        for j, rt in enumerate(build_role_texts(c)[:MAX_ROLES]):
            if rt.strip():
                flat_texts.append(rt)
                flat_indices.append((i, j))

    print(f"  Narrative texts: {len(narrative_texts):,}")
    print(f"  Profile texts:   {len(profile_texts):,}")
    print(f"  Role texts:      {len(flat_texts):,}")

    # ── Narrative embeddings ──
    if not should_skip("narrative_embs.npy"):
        print("\nEncoding narrative embeddings...")
        narrative_embs = embed_texts_batched(narrative_texts, model, desc="Narrative embs")
        np.save(PROCESSED_DIR / "narrative_embs.npy", narrative_embs)
        print(f"Narrative embeddings saved: {narrative_embs.shape}")
        del narrative_embs
        torch.cuda.empty_cache()

    del narrative_texts  # free RAM

    # ── Profile embeddings ──
    if not should_skip("profile_embs.npy"):
        print("\nEncoding profile embeddings...")
        profile_embs = embed_texts_batched(profile_texts, model, desc="Profile embs")
        np.save(PROCESSED_DIR / "profile_embs.npy", profile_embs)
        print(f"Profile embeddings saved: {profile_embs.shape}")
        del profile_embs
        torch.cuda.empty_cache()

    del profile_texts  # free RAM

    # ── Per-role embeddings ──
    if not should_skip("role_embs.npy"):
        narrative_embs_tmp = np.load(PROCESSED_DIR / "narrative_embs.npy")
        dim = narrative_embs_tmp.shape[1]

        role_embs_arr = np.zeros((N, MAX_ROLES, dim), dtype=np.float32)
        role_mask = np.zeros((N, MAX_ROLES), dtype=bool)

        print(f"\nEncoding {len(flat_texts):,} role descriptions...")
        flat_embs = embed_texts_batched(flat_texts, model, desc="Role embs")

        for (ci, ri), emb in zip(flat_indices, flat_embs):
            role_embs_arr[ci, ri] = emb
            role_mask[ci, ri] = True

        np.save(PROCESSED_DIR / "role_embs.npy", role_embs_arr)
        np.save(PROCESSED_DIR / "role_embs_mask.npy", role_mask)
        print(f"Per-role embeddings saved: {role_embs_arr.shape}")

    elapsed = time.time() - t0
    print(f"\n✓ Pre-computation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  All artifacts saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
