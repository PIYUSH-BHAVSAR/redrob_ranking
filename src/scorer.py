"""
scorer.py — Composite scoring with all 7 fixes applied.

Weight breakdown:
  0.30  responsibility_similarity  (per-role max, solves Tier-5)
  0.20  narrative_similarity       (career arc vs full JD)
  0.12  skills_score               (proficiency + assessment trust)
  0.08  domain_keyword_coverage    (fast interpretable keyword signal)
  0.18  trajectory_score           (tiered exp gate + seniority + product_score)
  0.07  behavioral_score           (activity decay + response + github)
  0.05  location_notice_score      (location + notice period)

Multipliers applied after weighted sum:
  * achievement_boost: 0.85 + 0.15 * achievement_depth
  * trap_penalty: 1.0 / 0.85 / 0.05

The exp_gate is applied as:
  trajectory_component * exp_gate_factor
  where exp_gate_factor uses the tiered curve (not binary 0/1)
"""

import random
import numpy as np

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
    cosine_sim,
)


def score_candidate(
    candidate: dict,
    idx: int,
    role_embs: list,
    narrative_emb: np.ndarray,
    profile_emb: np.ndarray,
    jd_embs: dict,
    percentiles: dict,
    trap_score: int,
) -> dict:
    """
    Compute composite score for one candidate.

    Args:
        candidate:      raw candidate dict
        idx:            index in dataset (for percentile lookup)
        role_embs:      list of per-role embeddings (pre-computed)
        narrative_emb:  career narrative embedding (pre-computed)
        profile_emb:    full profile text embedding (pre-computed)
        jd_embs:        dict with keys 'resp', 'full', 'req'
        percentiles:    behavioral signal percentile arrays
        trap_score:     int from trap_detector

    Returns:
        dict with 'score' and all component scores for reasoning
    """

    # ── Hard trap disqualification ──
    if trap_score >= 2:
        noise = random.uniform(0, 0.04)
        return {
            "score": round(0.03 + noise, 4),
            "is_disqualified": True,
            "trap_score": trap_score,
        }

    # ── Semantic similarities ──
    resp_sim = responsibility_similarity(role_embs, jd_embs["resp"])
    narr_sim = narrative_similarity(narrative_emb, jd_embs["full"])
    req_sim = requirements_similarity(profile_emb, jd_embs["req"])

    # ── Skills ──
    sk = skills_score(candidate)
    jd_skill = sk["jd_skill_score"]
    domain_cov = domain_keyword_coverage(candidate)

    # ── Career trajectory ──
    traj = trajectory_features(candidate)
    exp_gate_val = traj["exp_gate"]
    seniority = traj["seniority"]
    progression = traj["progression"]
    product = traj["product_score"]

    # Services background penalty — explicit JD disqualifier
    services_penalty = 0.70 if product <= 0.1 else 1.0

    # Trajectory composite (before exp gate)
    traj_raw = (
        0.45 * seniority +
        0.30 * product +
        0.25 * progression
    )

    # Apply exp gate — not binary, tiered
    # exp_gate 0 = hard exclude, 0.3 = partial, 1.0 = ideal
    if exp_gate_val == 0.0:
        # Under 3 years — can't make top 100
        return {
            "score": round(0.02 + random.uniform(0, 0.02), 4),
            "is_disqualified": True,
            "trap_score": trap_score,
            "reason": "under_experience",
        }
    traj_score = traj_raw * exp_gate_val

    # ── Behavioral ──
    beh = behavioral_features(candidate, percentiles, idx)
    behavioral_raw = (
        0.40 * beh["activity_decay"] +
        0.25 * beh["github_pct"] +
        0.20 * beh["response_pct"] +
        0.10 * beh["open_to_work"] +
        0.05 * beh["verified"]
    )

    # ── Location + notice ──
    loc = location_score(candidate)
    notice = notice_score(candidate)
    loc_notice = 0.60 * loc + 0.40 * notice

    # ── Weighted composite ──
    raw = (
        0.30 * resp_sim +
        0.20 * narr_sim +
        0.12 * jd_skill +
        0.08 * domain_cov +
        0.18 * traj_score +
        0.07 * behavioral_raw +
        0.05 * loc_notice
    )

    # ── Achievement multiplier ──
    ach = achievement_depth(candidate)
    achievement_multiplier = 0.85 + 0.15 * ach
    boosted = raw * achievement_multiplier

    # ── Soft trap penalty ──
    trap_penalty = 0.80 if trap_score == 1 else 1.0
    final = round(max(0.0, min(1.0, boosted * trap_penalty * services_penalty)), 4)

    return {
        "score": final,
        "is_disqualified": False,
        "trap_score": trap_score,
        "services_penalty": services_penalty,
        # Component scores — used for reasoning generation
        "resp_sim": round(resp_sim, 4),
        "narr_sim": round(narr_sim, 4),
        "req_sim": round(req_sim, 4),
        "jd_skill": round(jd_skill, 4),
        "domain_cov": round(domain_cov, 4),
        "traj_score": round(traj_score, 4),
        "exp_gate": exp_gate_val,
        "seniority": round(seniority, 4),
        "product_score": round(product, 4),
        "behavioral": round(behavioral_raw, 4),
        "activity_decay": round(beh["activity_decay"], 4),
        "open_to_work": beh["open_to_work"],
        "days_since_active": beh["days_since_active"],
        "response_pct": round(beh["response_pct"], 4),
        "github_pct": round(beh["github_pct"], 4),
        "loc_score": round(loc, 4),
        "notice_score": round(notice, 4),
        "achievement": round(ach, 4),
    }


def rank_candidates(scored: list) -> list:
    """
    Sort by score descending. For equal scores, sort by candidate_id ascending
    (deterministic tiebreak per spec).
    """
    return sorted(scored, key=lambda x: (-x["score"], x["candidate_id"]))
