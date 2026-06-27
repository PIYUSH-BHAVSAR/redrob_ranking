"""
trap_detector.py — Honeypot and keyword stuffer detection.

Two-layer approach:
  1. Hard deterministic rules — mechanical impossibilities
     (timeline anomalies, impossible skill claims, etc.)
  2. IsolationForest — statistical anomaly detection using
     score_samples() + percentile threshold, NOT contamination=0.01

Returns a trap_score per candidate:
  0 = clean
  1 = single flag (soft penalty 0.85x)
  2+ = likely trap (push to bottom 0.05–0.10)
"""

import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest

from features import days_since, REFERENCE_DATE, JD_CORE_SKILLS

# AI buzzwords for keyword stuffer detection
AI_BUZZWORDS = [
    "llm", "gpt", "chatgpt", "transformer", "diffusion", "rag",
    "vector", "embedding", "rlhf", "fine-tuning", "fine tuning",
    "neural", "agi", "foundation model", "langchain", "openai",
    "anthropic", "claude", "gemini", "stable diffusion",
    "attention mechanism", "bert", "gpt-4", "llama", "mistral",
]

# Non-technical job functions — career in these with AI skill claims = mismatch
NON_TECHNICAL_TITLES = [
    "hr", "human resources", "marketing", "sales", "accountant",
    "operations manager", "civil engineer", "mechanical engineer",
    "graphic designer", "content writer", "project manager",
    "business analyst", "customer support", "recruiter",
]

# Technical role keywords — any of these = technical career evidence
JD_TECHNICAL_ROLES = [
    "engineer", "scientist", "developer", "researcher", "architect",
    "analyst", "ml", "ai", "nlp", "data", "software",
]


# ─────────────────────────────────────────────
# LAYER 1 — Hard deterministic rules
# ─────────────────────────────────────────────

def check_timeline_anomaly(candidate: dict) -> bool:
    """
    Flag if sum of all role durations > years_of_experience * 12 + 30 months.
    Catches '8 yrs at a 3-yr-old company' style impossible profiles.
    """
    history = candidate.get("career_history", [])
    years_exp = candidate["profile"].get("years_of_experience", 0)
    total_months = sum(r.get("duration_months", 0) for r in history)
    claimed_months = years_exp * 12

    # Allow 30-month overlap buffer (legitimate parallel roles, rounding)
    if total_months > claimed_months + 30:
        return True

    # Also check: current role start_date vs years_of_experience
    # If someone claims 8 yrs exp but their oldest role started only 2 yrs ago
    if history:
        sorted_h = sorted(history, key=lambda r: r.get("start_date", "9999"))
        oldest_start = sorted_h[-1].get("start_date", "")
        if oldest_start:
            try:
                oldest_date = datetime.strptime(oldest_start[:10], "%Y-%m-%d").date()
                actual_years = (REFERENCE_DATE - oldest_date).days / 365
                # If claimed exp is more than 2x actual career span → suspicious
                if years_exp > actual_years * 2 + 2 and years_exp > 3:
                    return True
            except Exception:
                pass

    return False


def check_perfect_signal_cluster(candidate: dict) -> bool:
    """
    Real candidates have variance. Multiple signals all at maximum = synthetic.
    Flags if 3+ behavioral signals are at or near their theoretical max.
    """
    rs = candidate.get("redrob_signals", {})

    maxed = 0
    if rs.get("github_activity_score", 0) >= 98:
        maxed += 1
    if rs.get("recruiter_response_rate", 0) >= 0.99:
        maxed += 1
    if rs.get("offer_acceptance_rate", 0) >= 0.99:
        maxed += 1
    if rs.get("interview_completion_rate", 0) >= 0.99:
        maxed += 1
    if rs.get("profile_completeness_score", 0) >= 99:
        maxed += 1

    return maxed >= 3


def check_expert_zero_endorsement(candidate: dict) -> bool:
    """
    Claiming 'expert' in 8+ skills but zero endorsements across all of them.
    Real experts accumulate at least some social validation.
    """
    skills = candidate.get("skills", [])
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 8:
        total_endorsements = sum(s.get("endorsements", 0) for s in expert_skills)
        if total_endorsements == 0:
            return True
    return False


def check_keyword_stuffer(candidate: dict) -> bool:
    """
    High AI buzzword density in summary/headline but thin career descriptions.
    Also catches skills-only keyword stuffing.
    """
    summary = candidate["profile"].get("summary", "")
    headline = candidate["profile"].get("headline", "")
    surface_text = (summary + " " + headline).lower()

    surface_hits = sum(1 for w in AI_BUZZWORDS if w in surface_text)

    career_text = " ".join(
        r.get("description", "") for r in candidate.get("career_history", [])
    )
    career_words = len(career_text.split())

    # Many buzzwords but almost no career description text
    if surface_hits >= 7 and career_words < 80:
        return True

    # Buzzwords only in skills list, not in any actual work description
    if surface_hits >= 10 and career_words < 150:
        return True

    # NEW: skills list loaded with AI buzzwords but career text is thin
    skills_text = " ".join(s.get("name", "").lower() for s in candidate.get("skills", []))
    skills_hits = sum(1 for w in AI_BUZZWORDS if w in skills_text)
    if skills_hits >= 8 and career_words < 200:
        return True

    return False


def check_impossible_experience(candidate: dict) -> bool:
    """
    Years of experience is impossible given education end year.
    E.g., graduated 2022 but claims 10 years experience.
    """
    years_exp = candidate["profile"].get("years_of_experience", 0)
    education = candidate.get("education", [])

    if not education or years_exp < 3:
        return False

    latest_grad = max(
        (e.get("end_year", 0) for e in education if e.get("end_year")),
        default=0
    )
    if latest_grad == 0:
        return False

    # How many years since latest graduation?
    years_since_grad = REFERENCE_DATE.year - latest_grad

    # Can't have more than years_since_grad + 2 years experience (gap year etc)
    if years_exp > years_since_grad + 2 and years_exp > 5:
        return True

    return False


def check_job_function_mismatch(candidate: dict) -> bool:
    """
    Candidate claims AI skills but entire career is non-technical.
    Title is HR/Marketing/Operations but skills list is full of ML buzzwords.
    Catches the exact pattern the JD describes: 'a candidate who has all the
    AI keywords listed as skills but whose title is Marketing Manager is not a fit.'
    """
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Measure career time in technical vs non-technical roles
    non_tech_months = 0
    tech_months = 0
    for role in history:
        title = role.get("title", "").lower()
        months = role.get("duration_months", 1)
        if any(t in title for t in NON_TECHNICAL_TITLES):
            non_tech_months += months
        if any(t in title for t in JD_TECHNICAL_ROLES):
            tech_months += months

    total = non_tech_months + tech_months
    if total == 0:
        return False

    # Only flag if career is predominantly non-technical (70%+)
    if non_tech_months / total < 0.70:
        return False  # has enough technical history — not a mismatch

    # Count advanced/expert AI skill claims
    ai_skill_claims = sum(
        1 for s in skills
        if any(kw in s.get("name", "").lower() for kw in JD_CORE_SKILLS)
        and s.get("proficiency") in ("advanced", "expert")
    )

    # Total endorsements on AI skill claims — credibility signal
    ai_endorsements = sum(
        s.get("endorsements", 0) for s in skills
        if any(kw in s.get("name", "").lower() for kw in JD_CORE_SKILLS)
    )

    # Non-technical career + many uncredentialed AI skill claims = stuffer
    if ai_skill_claims >= 4 and ai_endorsements < 20:
        return True

    return False


def hard_trap_flags(candidate: dict) -> int:
    """
    Returns count of hard rule violations.
    Each violation = 1 flag.
    """
    flags = 0
    flags += int(check_timeline_anomaly(candidate))
    flags += int(check_perfect_signal_cluster(candidate))
    flags += int(check_expert_zero_endorsement(candidate))
    flags += int(check_keyword_stuffer(candidate))
    flags += int(check_impossible_experience(candidate))
    flags += int(check_job_function_mismatch(candidate))
    return flags


# ─────────────────────────────────────────────
# LAYER 2 — IsolationForest statistical anomaly
# ─────────────────────────────────────────────

def build_anomaly_feature_matrix(all_candidates: list, percentiles: dict) -> np.ndarray:
    """
    Build feature matrix for IsolationForest.
    Uses behavioral signals + experience + skill count.
    """
    rows = []
    for i, c in enumerate(all_candidates):
        p = c.get("profile", {})
        rs = c.get("redrob_signals", {})
        skills = c.get("skills", [])

        yrs = p.get("years_of_experience", 0)
        history = c.get("career_history", [])
        total_career_months = sum(r.get("duration_months", 0) for r in history)
        expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")

        github = float(percentiles["github_activity_score"][i])
        response = float(percentiles["recruiter_response_rate"][i])
        offer = float(percentiles["offer_acceptance_rate"][i])
        interview = float(percentiles["interview_completion_rate"][i])
        completeness = rs.get("profile_completeness_score", 50) / 100.0

        days_active = days_since(rs.get("last_active_date", ""))
        active_norm = np.exp(-days_active / 180.0)

        rows.append([
            yrs,
            total_career_months / max(yrs * 12, 1),  # ratio: career months / claimed
            expert_count,
            len(skills),
            github,
            response,
            offer,
            interview,
            completeness,
            active_norm,
        ])

    return np.array(rows, dtype=float)


def fit_isolation_forest(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Fit IsolationForest and return anomaly scores.
    Uses score_samples() not contamination — we pick threshold manually.
    Lower score = more anomalous.
    """
    clf = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(feature_matrix)
    raw_scores = clf.score_samples(feature_matrix)
    return raw_scores


def compute_isolation_flags(raw_scores: np.ndarray) -> np.ndarray:
    """
    Flag bottom 0.3% of anomaly scores as statistical honeypots.
    0.3% of 100K = ~300 candidates flagged, well above the ~80 real honeypots
    but conservative enough to avoid hurting real candidates.
    Returns boolean array.
    """
    threshold = np.percentile(raw_scores, 0.3)
    return raw_scores < threshold


# ─────────────────────────────────────────────
# COMBINED TRAP SCORE
# ─────────────────────────────────────────────

def compute_all_trap_scores(
    all_candidates: list,
    percentiles: dict,
) -> list:
    """
    Run both layers and return list of trap_score ints (one per candidate).
    Called once during pre-computation.
    """
    print("  Building anomaly feature matrix...")
    feat_matrix = build_anomaly_feature_matrix(all_candidates, percentiles)

    print("  Fitting IsolationForest...")
    raw_scores = fit_isolation_forest(feat_matrix)
    iso_flags = compute_isolation_flags(raw_scores)

    print("  Running hard rule checks...")
    trap_scores = []
    for i, c in enumerate(all_candidates):
        hard_flags = hard_trap_flags(c)
        iso_flag = int(iso_flags[i])
        total = hard_flags + iso_flag
        trap_scores.append(total)

    n_flagged = sum(1 for t in trap_scores if t >= 1)
    n_hard = sum(1 for t in trap_scores if t >= 2)
    print(f"  Trap detection: {n_flagged} candidates with 1+ flags, "
          f"{n_hard} with 2+ flags (likely traps)")

    return trap_scores
