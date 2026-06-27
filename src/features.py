"""
features.py — All feature extraction logic.

Feature groups:
  A. Semantic similarities (responsibility_sim, narrative_sim, requirements_sim)
     - Per-role embedding with max aggregation (solves Tier-5 problem)
  B. Skills analysis (proficiency-weighted + assessment score trust)
  C. Career trajectory (tiered exp gate, seniority, product company score)
  D. Behavioral signals (percentile-normalized, activity decay)
  E. Location + notice period
  F. Achievement depth (regex multiplier)

All embeddings are pre-computed. At rank time this module only does
numpy dot products and rule-based scoring — no model inference.
"""

import re
import numpy as np
from datetime import datetime, date

# ── Reference date for recency calculations ──
REFERENCE_DATE = date(2026, 6, 25)

# ── Proficiency weights ──
PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.45,
    "beginner": 0.15,
}

# ── Services company names (exact match after lower()) ──
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "hcl technologies", "tech mahindra", "mphasis", "hexaware",
    "mindtree", "l&t infotech", "ltimindtree", "niit technologies",
    "kpit", "cyient", "zensar", "persistent systems", "sonata software",
    "tata consultancy", "hindustan computers",
}

PRODUCT_INDUSTRIES = {
    "software", "saas", "technology", "fintech", "edtech",
    "healthtech", "ai", "internet", "e-commerce", "marketplace",
    "platform", "startup",
}

PRODUCT_SIZES = {"1-10", "11-50", "51-200", "201-500"}

SENIOR_TITLES = [
    "senior", "lead", "principal", "staff", "head",
    "director", "architect", "founding", "manager", "vp", "chief",
]

# JD core skills for keyword matching
JD_CORE_SKILLS = [
    "embeddings", "retrieval", "ranking", "vector search", "semantic search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "sentence-transformer", "rag", "llm", "fine-tuning",
    "fine tuning", "recommendation", "information retrieval", "hybrid search",
    "ann", "approximate nearest neighbor", "dense retrieval", "reranking",
    "re-ranking", "ndcg", "mrr", "a/b testing", "pytorch", "transformers",
    "huggingface", "bert", "nlp", "machine learning", "deep learning",
    "vector database", "vector db", "lora", "qlora",
]

# Achievement patterns — quantified impact signals
ACHIEVEMENT_PATTERNS = [
    r"\d+\s*[x×]\s*(improvement|faster|speedup|reduction|increase)",
    r"\d+\s*%\s*(improvement|reduction|increase|accuracy|faster|better)",
    r"(served|handles?|processing)\s+\d+[mk]?\+?\s*(users?|requests?|qps|rps)",
    r"reduced\s+.{0,30}\d+",
    r"improved\s+.{0,30}\d+",
    r"scaled?\s+(to|from)\s+\d+",
    r"latency\s+.{0,20}\d+\s*(ms|sec|s\b)",
    r"deployed\s+to\s+prod",
    r"\d+\s*(million|billion|M|B)\s*(users?|records?|requests?|documents?)",
    r"(built|shipped|launched|delivered|owned)\s+.{0,40}(system|pipeline|service|platform|engine)",
]


# ─────────────────────────────────────────────
# A. SEMANTIC SIMILARITIES  (uses pre-computed embeddings)
# ─────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def responsibility_similarity(role_embs: list, jd_resp_emb: np.ndarray) -> float:
    """
    Max cosine similarity of any individual role description against
    JD responsibilities. Solves Tier-5 problem: a candidate who built
    'recommendation engine' scores high even without RAG buzzwords.
    Also gives a small bonus if 2+ roles are relevant (breadth signal).
    """
    if not role_embs:
        return 0.0
    sims = [cosine_sim(emb, jd_resp_emb) for emb in role_embs]
    sims_sorted = sorted(sims, reverse=True)
    best = sims_sorted[0]
    second = sims_sorted[1] if len(sims_sorted) > 1 else 0.0
    # 80% best role + 20% second best — slight breadth bonus
    return 0.80 * best + 0.20 * second


def narrative_similarity(narrative_emb: np.ndarray, jd_full_emb: np.ndarray) -> float:
    """
    Cosine similarity of the candidate's career narrative (last 3 roles
    concatenated) against the full JD. Catches overall career arc fit.
    """
    if narrative_emb is None:
        return 0.0
    return cosine_sim(narrative_emb, jd_full_emb)


def requirements_similarity(profile_emb: np.ndarray, jd_req_emb: np.ndarray) -> float:
    """
    Cosine similarity of candidate's full profile text against the
    JD requirements section. Rewards explicit skill/experience claims.
    """
    if profile_emb is None:
        return 0.0
    return cosine_sim(profile_emb, jd_req_emb)


# ─────────────────────────────────────────────
# B. SKILLS ANALYSIS
# ─────────────────────────────────────────────

def skills_score(candidate: dict) -> dict:
    """
    Proficiency-weighted, endorsement-boosted, assessment-verified skill score.

    Key fix vs original plan: uses skill_assessment_scores as a trust
    multiplier. Claimed 'expert' with verified score 91 >> claimed 'expert'
    with no assessment.
    """
    skills = candidate.get("skills", [])
    rs = candidate.get("redrob_signals", {})
    assessment_scores = rs.get("skill_assessment_scores", {})

    jd_skill_total = 0.0
    jd_skill_count = 0
    total_expert = 0
    bonus_skill_total = 0.0

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        endorsements = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)

        prof_weight = PROFICIENCY_WEIGHT.get(prof, 0.15)

        # Endorsement boost: capped at 30% extra
        endorse_boost = min(0.30, endorsements / 50 * 0.30)

        # Duration signal: skill used for 12+ months is more credible
        duration_factor = min(1.0, duration / 12) if duration > 0 else 0.5

        # Assessment trust multiplier
        # Find assessment for this skill (fuzzy: check if skill name is in key)
        assessment_val = None
        for k, v in assessment_scores.items():
            if k.lower() in name or name in k.lower():
                assessment_val = v
                break
        if assessment_val is not None:
            # Verified score: 0-100 → 0.5-1.0 multiplier
            trust = 0.5 + (assessment_val / 100) * 0.5
        else:
            # Unverified claim: moderate discount
            trust = 0.70

        raw_skill_val = prof_weight * (1 + endorse_boost) * duration_factor * trust

        # Check if JD core skill
        is_jd_skill = any(jd_kw in name or name in jd_kw for jd_kw in JD_CORE_SKILLS)
        if is_jd_skill:
            jd_skill_total += raw_skill_val
            jd_skill_count += 1

        # Bonus skills
        bonus_kws = ["lora", "qlora", "peft", "learning to rank", "xgboost",
                     "mlops", "distributed", "open source"]
        if any(bk in name for bk in bonus_kws):
            bonus_skill_total += raw_skill_val * 0.3

        if prof == "expert":
            total_expert += 1

    # Normalize JD skill score to 0-1 range (saturates at 5 strong skill matches)
    normalized_jd = min(1.0, jd_skill_total / 5.0)

    return {
        "jd_skill_score": normalized_jd,
        "jd_skill_count": jd_skill_count,
        "expert_skill_count": total_expert,
        "bonus_skill_score": min(0.5, bonus_skill_total),
        "raw_jd_skill_total": jd_skill_total,
    }


def domain_keyword_coverage(candidate: dict) -> float:
    """
    Fraction of JD core domain keywords found anywhere in the candidate's
    full career text. Fast, interpretable, complements embeddings.
    """
    full_text = _build_full_career_text(candidate).lower()
    hits = sum(1 for kw in JD_CORE_SKILLS if kw in full_text)
    return min(1.0, hits / max(len(JD_CORE_SKILLS) * 0.3, 1))


# ─────────────────────────────────────────────
# C. CAREER TRAJECTORY
# ─────────────────────────────────────────────

def experience_gate(years: float) -> float:
    """
    Tiered experience gate per JD requirements.
    < 3 yrs  → 0.0   hard exclude
    3–5 yrs  → 0.3   possible exception
    5–9 yrs  → 1.0   ideal band
    9–12 yrs → 0.85  slightly over, still valid
    12+ yrs  → 0.60  likely wrong fit for founding team stage
    """
    if years < 3:
        return 0.0
    elif years < 5:
        return 0.30
    elif years <= 9:
        return 1.0
    elif years <= 12:
        return 0.85
    else:
        return 0.60


def seniority_score(candidate: dict) -> float:
    """Score for current role seniority."""
    title = candidate["profile"].get("current_title", "").lower()
    hits = sum(1 for t in SENIOR_TITLES if t in title)
    return min(1.0, hits * 0.5)


def career_progression(career_history: list) -> float:
    """
    Is seniority increasing over time? 
    Returns 1.0 if yes, 0.5 if flat, 0.2 if declining.
    """
    if len(career_history) < 2:
        return 0.5
    sorted_roles = sorted(career_history, key=lambda r: r.get("start_date", ""))
    seniority_vals = [
        sum(1 for t in SENIOR_TITLES if t in r.get("title", "").lower())
        for r in sorted_roles
    ]
    if seniority_vals[-1] > seniority_vals[0]:
        return 1.0
    elif seniority_vals[-1] == seniority_vals[0]:
        return 0.5
    else:
        return 0.2


def product_company_score(career_history: list) -> float:
    """
    Score based on product vs services company background.
    Uses structured fields (company name, industry, size) not text keywords.
    Returns -1.0 (pure services) to +1.0 (pure product).
    Normalized to 0–1 for scoring.
    """
    weighted_score = 0.0
    total_months = 0

    for role in career_history:
        months = max(role.get("duration_months", 1), 1)
        company = role.get("company", "").lower().strip()
        industry = role.get("industry", "").lower()
        size = role.get("company_size", "")

        role_score = 0.0

        # Check explicit services company names
        is_services = any(s in company for s in SERVICES_COMPANIES)
        if is_services:
            role_score = -1.5  # explicit JD disqualifier
        elif size in PRODUCT_SIZES:
            role_score = 1.0   # small/mid company = likely product
        elif any(ind in industry for ind in PRODUCT_INDUSTRIES):
            role_score = 0.75  # product industry signal
        else:
            role_score = 0.0   # neutral

        weighted_score += role_score * months
        total_months += months

    if total_months == 0:
        return 0.3  # no data, mild neutral

    raw = weighted_score / total_months  # -1.5 to +1.0
    # Normalize to 0–1
    normalized = (raw + 1.5) / 2.5
    return max(0.0, min(1.0, normalized))


def trajectory_features(candidate: dict) -> dict:
    history = candidate.get("career_history", [])
    years = candidate["profile"].get("years_of_experience", 0)

    return {
        "exp_gate": experience_gate(years),
        "years_exp": years,
        "seniority": seniority_score(candidate),
        "progression": career_progression(history),
        "product_score": product_company_score(history),
    }


# ─────────────────────────────────────────────
# D. BEHAVIORAL SIGNALS
# ─────────────────────────────────────────────

def days_since(date_str: str) -> int:
    """Days between a date string (YYYY-MM-DD) and REFERENCE_DATE."""
    if not date_str:
        return 365
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return max(0, (REFERENCE_DATE - d).days)
    except Exception:
        return 365


def activity_decay(last_active_date: str) -> float:
    """
    Exponential decay based on days since last active.
    30 days → 0.78,  60 days → 0.51,  90 days → 0.36,  180 days → 0.13
    """
    d = days_since(last_active_date)
    return float(np.exp(-d / 90.0))


def behavioral_features(candidate: dict, percentiles: dict, idx: int) -> dict:
    """
    Percentile-normalized behavioral signals.
    percentiles: dict of signal_name → np.array of per-candidate percentiles
    idx: index of this candidate in the percentiles arrays
    """
    rs = candidate.get("redrob_signals", {})

    decay = activity_decay(rs.get("last_active_date", ""))
    open_to_work = float(rs.get("open_to_work_flag", False))
    verified = float(rs.get("verified_email", False)) * 0.5 + \
               float(rs.get("verified_phone", False)) * 0.5

    github_pct = float(percentiles["github_activity_score"][idx])
    response_pct = float(percentiles["recruiter_response_rate"][idx])
    interview_pct = float(percentiles["interview_completion_rate"][idx])
    offer_pct = float(percentiles["offer_acceptance_rate"][idx])

    # Profile completeness normalized
    completeness = rs.get("profile_completeness_score", 50) / 100.0

    return {
        "activity_decay": decay,
        "open_to_work": open_to_work,
        "github_pct": github_pct,
        "response_pct": response_pct,
        "interview_pct": interview_pct,
        "offer_pct": offer_pct,
        "verified": verified,
        "completeness": completeness,
        "days_since_active": days_since(rs.get("last_active_date", "")),
        "saved_by_recruiters": min(1.0, rs.get("saved_by_recruiters_30d", 0) / 20),
    }


def compute_behavioral_percentiles(all_candidates: list) -> dict:
    """
    Compute per-signal percentile ranks across the full 100K dataset.
    Called once, results passed to behavioral_features() per candidate.
    """
    from scipy.stats import rankdata

    signals = {
        "github_activity_score": [],
        "recruiter_response_rate": [],
        "interview_completion_rate": [],
        "offer_acceptance_rate": [],
    }

    for c in all_candidates:
        rs = c.get("redrob_signals", {})
        for s in signals:
            val = rs.get(s, None)
            # -1 means no data (e.g. no github, no offer history)
            if val is None or val == -1:
                val = 0.0
            signals[s].append(float(val))

    percentiles = {}
    for s, arr in signals.items():
        arr = np.array(arr, dtype=float)
        ranks = rankdata(arr, method="average")
        percentiles[s] = ranks / len(ranks)

    return percentiles


# ─────────────────────────────────────────────
# E. LOCATION + NOTICE PERIOD
# ─────────────────────────────────────────────

TIER1_TARGET = {"pune", "noida", "delhi", "ncr", "gurgaon", "gurugram", "new delhi"}
TIER1_NEARBY = {"mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata", "ahmedabad"}


def location_score(candidate: dict) -> float:
    """
    Location scoring per JD preference.
    Pune/Noida → 1.0, other Tier-1 India + relocate → 0.75, etc.
    """
    profile = candidate.get("profile", {})
    rs = candidate.get("redrob_signals", {})

    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = rs.get("willing_to_relocate", False)

    # Outside India — JD says case-by-case, no visa sponsorship
    if country not in ("india", "in"):
        return 0.10 if relocate else 0.05

    # Already in target city
    if any(city in loc for city in TIER1_TARGET):
        return 1.0

    # Nearby Tier-1 city
    if any(city in loc for city in TIER1_NEARBY):
        return 0.75 if relocate else 0.55

    # Elsewhere in India
    return 0.45 if relocate else 0.25


def notice_score(candidate: dict) -> float:
    """
    JD: sub-30 day preferred, up to 30 buyout, 30+ bar gets higher.
    """
    rs = candidate.get("redrob_signals", {})
    days = rs.get("notice_period_days", 60)

    if days <= 15:
        return 1.0
    elif days <= 30:
        return 0.90
    elif days <= 60:
        return 0.65
    elif days <= 90:
        return 0.40
    else:
        return 0.15  # 90+ days is a real problem for founding team


# ─────────────────────────────────────────────
# F. ACHIEVEMENT DEPTH  (multiplier on responsibility_sim)
# ─────────────────────────────────────────────

def achievement_depth(candidate: dict) -> float:
    """
    Regex-based detection of quantified impact in career descriptions.
    Returns 0–1. Used as a multiplier boost on responsibility_similarity.
    Candidates with 3+ achievement signals get max boost.
    """
    full_text = " ".join(
        r.get("description", "") for r in candidate.get("career_history", [])
    )
    hits = sum(
        1 for pattern in ACHIEVEMENT_PATTERNS
        if re.search(pattern, full_text, re.IGNORECASE)
    )
    return min(1.0, hits / 3.0)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _build_full_career_text(candidate: dict) -> str:
    """All career text concatenated: headline + summary + all role descriptions."""
    p = candidate.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(p for p in parts if p)


def build_career_narrative_text(candidate: dict, n_roles: int = 3) -> str:
    """
    Last N roles concatenated as the 'career narrative' for embedding.
    Recent roles weighted more — early irrelevant roles should not dilute.
    """
    history = candidate.get("career_history", [])
    sorted_history = sorted(history, key=lambda r: r.get("start_date", ""), reverse=True)
    recent = sorted_history[:n_roles]

    parts = [candidate["profile"].get("headline", ""),
             candidate["profile"].get("summary", "")]
    for role in recent:
        parts.append(f"{role.get('title','')} at {role.get('company','')}: "
                     f"{role.get('description','')}")
    return " ".join(p for p in parts if p)


def build_role_texts(candidate: dict) -> list:
    """
    List of individual role description texts for per-role embedding.
    Each entry = role title + description (enough context for the model).
    Returns at most 5 most recent roles.
    """
    history = candidate.get("career_history", [])
    sorted_history = sorted(history, key=lambda r: r.get("start_date", ""), reverse=True)
    texts = []
    for role in sorted_history[:5]:
        title = role.get("title", "")
        desc = role.get("description", "")
        company = role.get("company", "")
        if desc.strip():
            texts.append(f"{title} at {company}. {desc}")
    return texts


def build_profile_text(candidate: dict) -> str:
    """Full profile text for requirements_similarity embedding."""
    return _build_full_career_text(candidate)
