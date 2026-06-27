"""
reasoner.py — Evidence-first reasoning generation.

Each reasoning string:
  Sentence 1: Technical fit — specific role + company + years + top skill/work.
  Sentence 2: Reachability + one differentiator or concern.

Rules:
  - Every claim must reference actual data from the candidate profile.
  - No hallucination — don't mention skills not in their profile.
  - Honest concerns for lower-ranked candidates (notice, inactivity, etc.).
  - Reasoning tone must match the rank.
"""

from features import days_since, JD_CORE_SKILLS


def build_reasoning(candidate: dict, score_components: dict, rank: int) -> str:
    """
    Build 1-2 sentence evidence-first reasoning string.
    """
    p = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    rs = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    # ── Most recent role ──
    sorted_history = sorted(history, key=lambda r: r.get("start_date", ""), reverse=True)
    recent = sorted_history[0] if sorted_history else {}
    title = recent.get("title", p.get("current_title", "Unknown"))
    company = recent.get("company", p.get("current_company", "Unknown"))
    yrs = p.get("years_of_experience", 0)

    # ── Find strongest JD-relevant skill ──
    top_skill = _get_top_jd_skill(skills, rs)

    # ── Activity string ──
    days = score_components.get("days_since_active", 365)
    if days <= 7:
        active_str = f"active {days}d ago"
    elif days <= 30:
        active_str = f"active {days}d ago"
    elif days <= 90:
        active_str = f"last active {days}d ago"
    else:
        active_str = f"last active {days}d ago — note engagement lag"

    open_str = ", open to work" if rs.get("open_to_work_flag") else ""

    # ── Experience band note ──
    exp_gate = score_components.get("exp_gate", 1.0)
    if exp_gate < 0.5:
        exp_note = f" ({yrs} yrs exp — below preferred 5-yr band)"
    elif yrs > 12:
        exp_note = f" ({yrs} yrs — senior; founding team fit to verify)"
    else:
        exp_note = ""

    # ── Product company note ──
    product = score_components.get("product_score", 0.5)
    if product >= 0.7:
        product_note = " Product company background."
    elif product <= 0.3:
        product_note = " Predominantly services/consulting background."
    else:
        product_note = ""

    # ── Notice period note ──
    notice_days = rs.get("notice_period_days", 60)
    if notice_days <= 30:
        notice_note = f" Notice: {notice_days}d."
    elif notice_days > 90:
        notice_note = f" Notice: {notice_days}d (long)."
    else:
        notice_note = ""

    # ── Location note ──
    location = p.get("location", "")
    country = p.get("country", "")
    relocate = rs.get("willing_to_relocate", False)
    loc_score = score_components.get("loc_score", 0.5)
    if loc_score >= 0.9:
        loc_note = f" Based in {location}."
    elif country != "India" and country != "india":
        loc_note = f" Based in {location}, {country}" + (" (willing to relocate)" if relocate else " (relocation unclear)")  + "."
    else:
        loc_note = ""

    # ── Construct sentences ──

    # Sentence 1: technical fit
    if top_skill:
        s1 = (f"{title} at {company} with {yrs:.1f} yrs exp{exp_note}; "
              f"{top_skill} is the strongest JD-relevant signal.")
    else:
        s1 = (f"{title} at {company} with {yrs:.1f} yrs exp{exp_note}; "
              f"limited direct match to core retrieval/ranking requirements.")

    # Sentence 2: reachability + differentiator + concerns
    resp_sim = score_components.get("resp_sim", 0)
    if resp_sim >= 0.55:
        work_note = " Career descriptions closely match JD responsibilities."
    elif resp_sim >= 0.40:
        work_note = " Some career overlap with JD responsibilities."
    else:
        work_note = " Limited career evidence matching JD responsibilities."

    s2 = f"{active_str.capitalize()}{open_str}.{work_note}{product_note}{notice_note}{loc_note}"

    # ── For low-ranked candidates, add honest concern ──
    if rank >= 80:
        concerns = _get_concerns(candidate, score_components)
        if concerns:
            s2 += f" Concern: {concerns}"

    return f"{s1} {s2}".strip()


def _get_top_jd_skill(skills: list, rs: dict) -> str:
    """
    Find the highest-proficiency JD-relevant skill the candidate has,
    preferring assessment-verified ones.
    """
    assessment_scores = rs.get("skill_assessment_scores", {})
    best_skill = None
    best_val = -1

    proficiency_rank = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")

        # Check if JD-relevant
        is_relevant = any(jd_kw in name or name in jd_kw for jd_kw in JD_CORE_SKILLS)
        if not is_relevant:
            continue

        prof_val = proficiency_rank.get(prof, 1)

        # Assessment boost
        for k, v in assessment_scores.items():
            if k.lower() in name or name in k.lower():
                prof_val += v / 100  # up to +1 for a score of 100
                break

        if prof_val > best_val:
            best_val = prof_val
            best_skill = s.get("name", "")

    return best_skill or ""


def _get_concerns(candidate: dict, score_components: dict) -> str:
    """Generate honest concern string for lower-ranked candidates."""
    concerns = []

    exp_gate = score_components.get("exp_gate", 1.0)
    yrs = candidate["profile"].get("years_of_experience", 0)
    if exp_gate < 0.5:
        concerns.append(f"{yrs} yrs exp is below 5-yr minimum")

    days = score_components.get("days_since_active", 0)
    if days > 180:
        concerns.append(f"inactive for {days} days")

    product = score_components.get("product_score", 0.5)
    if product < 0.3:
        concerns.append("primarily services/consulting background")

    resp = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 1.0)
    if resp < 0.2:
        concerns.append(f"low recruiter response rate ({resp:.0%})")

    notice = candidate.get("redrob_signals", {}).get("notice_period_days", 60)
    if notice > 90:
        concerns.append(f"{notice}d notice period")

    return "; ".join(concerns[:2]) + "." if concerns else ""
