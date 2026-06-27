"""
app.py — HuggingFace Spaces sandbox for Redrob Track 1.

Accepts up to 100 candidate records as JSON/JSONL,
runs the full ranking pipeline, and returns a ranked CSV.
No GPU needed — runs on CPU within 5 minutes for ≤100 candidates.
"""

import gradio as gr
import json
import csv
import io
import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

from parse_jd import get_jd
from features import (
    build_role_texts, build_career_narrative_text, build_profile_text,
    compute_behavioral_percentiles, responsibility_similarity,
    narrative_similarity, requirements_similarity, skills_score,
    domain_keyword_coverage, trajectory_features, behavioral_features,
    location_score, notice_score, achievement_depth,
)
from trap_detector import compute_all_trap_scores
from scorer import score_candidate, rank_candidates
from reasoner import build_reasoning

from sentence_transformers import SentenceTransformer

# Load model once at startup
MODEL_PATH = "models/minilm"
print("Loading model...")
model = SentenceTransformer(MODEL_PATH)
print("Model loaded.")

# Pre-compute JD embeddings once
print("Embedding JD...")
jd = get_jd()
jd_texts = [jd["responsibilities_text"], jd["full_text"], jd["requirements_text"]]
jd_embs_arr = model.encode(jd_texts, normalize_embeddings=True)
JD_EMBS = {
    "resp": jd_embs_arr[0],
    "full": jd_embs_arr[1],
    "req":  jd_embs_arr[2],
}
print("JD embedded.")


def rank_candidates_fn(file_input, jsonl_text):
    """Main ranking function called by Gradio."""
    try:
        # Parse input — file upload takes priority over text input
        candidates = []
        if file_input is not None:
            content = file_input.decode("utf-8") if isinstance(file_input, bytes) else open(file_input).read()
            for line in content.strip().splitlines():
                if line.strip():
                    candidates.append(json.loads(line))
        elif jsonl_text and jsonl_text.strip():
            # Try JSON array first, then JSONL
            text = jsonl_text.strip()
            if text.startswith("["):
                candidates = json.loads(text)
            else:
                for line in text.splitlines():
                    if line.strip():
                        candidates.append(json.loads(line))

        if not candidates:
            return "No candidates provided.", None

        if len(candidates) > 100:
            candidates = candidates[:100]
            note = f"Truncated to 100 candidates."
        else:
            note = f"Processing {len(candidates)} candidates."

        print(f"{note}")

        # Embed candidates
        narrative_texts = [build_career_narrative_text(c) for c in candidates]
        profile_texts = [build_profile_text(c) for c in candidates]

        narrative_embs = model.encode(narrative_texts, normalize_embeddings=True)
        profile_embs = model.encode(profile_texts, normalize_embeddings=True)

        # Per-role embeddings
        MAX_ROLES = 5
        dim = narrative_embs.shape[1]
        role_embs_arr = np.zeros((len(candidates), MAX_ROLES, dim), dtype=np.float32)
        role_mask = np.zeros((len(candidates), MAX_ROLES), dtype=bool)

        flat_texts, flat_indices = [], []
        for i, c in enumerate(candidates):
            for j, rt in enumerate(build_role_texts(c)[:MAX_ROLES]):
                if rt.strip():
                    flat_texts.append(rt)
                    flat_indices.append((i, j))

        if flat_texts:
            flat_embs = model.encode(flat_texts, normalize_embeddings=True)
            for (ci, ri), emb in zip(flat_indices, flat_embs):
                role_embs_arr[ci, ri] = emb
                role_mask[ci, ri] = True

        # Behavioral percentiles
        percentiles = compute_behavioral_percentiles(candidates)

        # Trap scores
        trap_scores = compute_all_trap_scores(candidates, percentiles)

        # Score all candidates
        scored = []
        for i, candidate in enumerate(candidates):
            role_embs = list(role_embs_arr[i][role_mask[i]])
            result = score_candidate(
                candidate=candidate,
                idx=i,
                role_embs=role_embs,
                narrative_emb=narrative_embs[i],
                profile_emb=profile_embs[i],
                jd_embs=JD_EMBS,
                percentiles=percentiles,
                trap_score=int(trap_scores[i]),
            )
            scored.append({
                "candidate_id": candidate["candidate_id"],
                "score": result["score"],
                "components": result,
                "candidate": candidate,
            })

        # Rank and take top N
        ranked = rank_candidates(scored)
        top_n = ranked[:min(100, len(ranked))]

        # Generate output
        rows = []
        for rank_pos, item in enumerate(top_n, start=1):
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

        # Build CSV string for display
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
        csv_str = output.getvalue()

        # Save to temp file for download
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write(csv_str)
        tmp.close()

        summary = f"{note}\nRanked {len(rows)} candidates.\nTop score: {rows[0]['score']} | Rank 1: {rows[0]['candidate_id']}\n\nPreview (top 5):\n"
        for r in rows[:5]:
            summary += f"  #{r['rank']}  {r['candidate_id']}  {r['score']}  {r['reasoning'][:80]}...\n"

        return summary, tmp.name

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", None


# ── Gradio UI ──
with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown("""
    # Redrob Track 1 — Intelligent Candidate Ranker
    Upload a JSONL file or paste candidate JSON to rank candidates against the Senior AI Engineer JD.
    - Accepts up to 100 candidates
    - Runs on CPU, no GPU needed
    - Returns ranked CSV with scores and reasoning
    """)

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload candidates.jsonl (≤100 candidates)", file_types=[".jsonl", ".json"])
            jsonl_text = gr.Textbox(
                label="Or paste JSONL / JSON array here",
                placeholder='[{"candidate_id": "CAND_0000001", ...}]',
                lines=5,
            )
            run_btn = gr.Button("Rank Candidates", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Results summary", lines=12)
            output_file = gr.File(label="Download ranked CSV")

    run_btn.click(
        fn=rank_candidates_fn,
        inputs=[file_input, jsonl_text],
        outputs=[output_text, output_file],
    )

    gr.Markdown("""
    **Note:** This sandbox runs on a small sample. Full 100K ranking runs locally per the README.
    """)

if __name__ == "__main__":
    demo.launch()
