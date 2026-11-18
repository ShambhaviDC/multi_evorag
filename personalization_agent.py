"""
personalization_agent.py — Dynamic User Personalization Agent for MultiPersona EvoRAG.

Combines:
  1. Explicit Preferences: Directly provided by user (categories, perspectives, language, length)
  2. Implicit Preferences: Inferred from conversation queries using MiniLM embeddings + recency decay
  3. Persona Affinity Scoring: Decoupled 20% personalization weighting + 80% objective peer review
  4. User Control & Privacy: View, edit, manual add, exclude, toggle, and reset interests
"""

import json
import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlmodel import Session, select

from config import PERSONAS
from database import get_session
from models import (
    PersonalizationProfile,
    PersonalizationProfileRead,
    PersonalizationProfileUpdate,
    User,
    UserPreference,
    utc_now,
)

log = logging.getLogger(__name__)

# ── Predefined Interest Categories (14 Categories) ───────────────────────────
PREDEFINED_CATEGORIES: List[str] = [
    "Politics",
    "Technology",
    "Artificial Intelligence",
    "Business and Startups",
    "Economy and Finance",
    "Sports",
    "Science",
    "Environment",
    "Healthcare",
    "International Affairs",
    "Education",
    "Employment",
    "Cybersecurity",
    "Innovation",
]

# Rich keywords per category for semantic augmentation and offline matching
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Politics": [
        "politics", "political", "government", "election", "parliament", "congress",
        "senate", "policy", "legislation", "democracy", "diplomat", "bill", "court",
        "law", "minister", "president", "voting", "campaign",
    ],
    "Technology": [
        "technology", "tech", "software", "hardware", "computing", "computer", "cloud",
        "semiconductor", "chip", "chips", "robotics", "device", "internet", "gpu",
        "processor", "telecom", "architecture", "microelectronics",
    ],
    "Artificial Intelligence": [
        "artificial intelligence", "ai", "machine learning", "ml", "deep learning",
        "llm", "neural network", "openai", "anthropic", "chatgpt", "gemini",
        "transformer", "nlp", "computer vision", "diffusion", "rag", "phi-3",
        "agent", "autonomous",
    ],
    "Business and Startups": [
        "business", "startup", "startups", "venture capital", "vc", "founder",
        "entrepreneur", "ipo", "acquisition", "merger", "revenue", "valuation",
        "unicorn", "corporate", "commercial", "enterprise",
    ],
    "Economy and Finance": [
        "economy", "economic", "finance", "financial", "inflation", "interest rates",
        "gdp", "stock market", "banking", "treasury", "recession", "fed", "fiscal",
        "stocks", "wall street", "monetary", "debt", "tariff",
    ],
    "Sports": [
        "sports", "football", "soccer", "basketball", "cricket", "tennis", "olympics",
        "championship", "tournament", "athlete", "player", "fifa", "nba", "match",
        "league", "coach", "world cup",
    ],
    "Science": [
        "science", "scientific", "physics", "astronomy", "space", "biology",
        "chemistry", "quantum", "genetics", "nasa", "telescope", "laboratory",
        "experiment", "matter", "cosmology", "particles",
    ],
    "Environment": [
        "environment", "climate", "climate change", "global warming", "carbon",
        "renewable", "solar", "wind", "sustainability", "emissions", "pollution",
        "green energy", "biodiversity", "conservation",
    ],
    "Healthcare": [
        "healthcare", "health", "medicine", "medical", "disease", "vaccine",
        "clinical trial", "hospital", "fda", "doctor", "pharmaceutical", "pharma",
        "therapy", "pandemic", "cancer", "treatment",
    ],
    "International Affairs": [
        "international affairs", "geopolitics", "foreign policy", "diplomacy",
        "united nations", "un", "nato", "sanctions", "treaty", "ambassador",
        "bilateral", "global conflict", "treaties", "summit", "relations",
    ],
    "Education": [
        "education", "educational", "school", "university", "college", "student",
        "teacher", "curriculum", "academic", "scholarship", "learning", "tuition",
        "literacy", "pedagogy",
    ],
    "Employment": [
        "employment", "jobs", "job", "career", "labor", "workforce", "hiring",
        "unemployment", "layoffs", "wages", "salary", "remote work", "occupations",
    ],
    "Cybersecurity": [
        "cybersecurity", "cyber", "hacking", "hacker", "ransomware", "malware",
        "data breach", "vulnerability", "encryption", "firewall", "phishing",
        "zero-day", "infosec", "threat", "ddos",
    ],
    "Innovation": [
        "innovation", "innovative", "breakthrough", "patents", "cutting-edge",
        "next-gen", "disruption", "futuristic", "r&d", "pioneering", "discovery",
    ],
}

# Persona affinity mapping: which personas specialize in which topics and angles
PERSONA_AFFINITIES: Dict[str, Dict[str, Any]] = {
    "analytical-critical": {
        "categories": ["Technology", "Artificial Intelligence", "Innovation", "Economy and Finance", "Cybersecurity", "Science"],
        "perspectives": ["Analytical", "Critical", "Technical", "In-depth", "Architecture"],
        "description": "Rigorous logical decomposition, architectural assessment, critical inquiry",
    },
    "empirical-evidential": {
        "categories": ["Science", "Healthcare", "Economy and Finance", "Artificial Intelligence", "Technology"],
        "perspectives": ["Empirical", "Data-driven", "Evidence-based", "Factual", "Quantitative"],
        "description": "Grounding conclusions in verifiable data, benchmarks, empirical findings",
    },
    "adversarial-skeptical": {
        "categories": ["Politics", "International Affairs", "Cybersecurity", "Business and Startups"],
        "perspectives": ["Skeptical", "Counter-argument", "Adversarial", "Risk-aware", "Devil's Advocate"],
        "description": "Stress-testing assumptions, highlighting vulnerabilities, identifying edge cases",
    },
    "domain-expert": {
        "categories": ["Artificial Intelligence", "Technology", "Science", "Healthcare", "Cybersecurity"],
        "perspectives": ["Technical", "Specialized", "Expert", "Deep Dive", "Domain-Specific"],
        "description": "Deep domain terminology, authoritative specialized insight, precise technical context",
    },
    "synthesis-integrative": {
        "categories": ["International Affairs", "Education", "Environment", "Politics", "Sports"],
        "perspectives": ["Balanced", "Synthesized", "Holistic", "Summary", "Cross-disciplinary"],
        "description": "Harmonizing multifaceted viewpoints into cohesive, actionable syntheses",
    },
    "temporal-contextual": {
        "categories": ["Politics", "International Affairs", "Innovation", "Economy and Finance", "Education"],
        "perspectives": ["Historical", "Timeline", "Evolutionary", "Contextual", "Chronological"],
        "description": "Tracing historical developments, trajectory shifts, future projections",
    },
    "creative-lateral": {
        "categories": ["Innovation", "Business and Startups", "Environment", "Technology", "Sports"],
        "perspectives": ["Creative", "Lateral", "Strategic", "Alternative", "Unconventional"],
        "description": "Unconventional connections, creative lateral thinking, novel solutions",
    },
    "risk-evaluative": {
        "categories": ["Economy and Finance", "Cybersecurity", "Healthcare", "Business and Startups", "Environment"],
        "perspectives": ["Risk", "Impact", "Safety", "Vulnerability", "Mitigation"],
        "description": "Systemic risk appraisal, second-order effects, threat mitigation",
    },
}

# ── Lazy Category Embeddings Cache ───────────────────────────────────────────
_category_embeddings: Optional[np.ndarray] = None


def _get_category_embeddings() -> Optional[np.ndarray]:
    """Precompute or return cached normalized embeddings for all 14 categories."""
    global _category_embeddings
    if _category_embeddings is not None:
        return _category_embeddings

    try:
        from embedder import _get_model
        model = _get_model()
        category_prompts = [
            f"{cat}: {' '.join(CATEGORY_KEYWORDS.get(cat, [])[:10])}"
            for cat in PREDEFINED_CATEGORIES
        ]
        raw_vecs = model.encode(category_prompts, convert_to_numpy=True)
        # Normalize to unit length for fast cosine similarity via dot product
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        _category_embeddings = raw_vecs / norms
        log.info("Initialized semantic embeddings for %d personalization categories.", len(PREDEFINED_CATEGORIES))
        return _category_embeddings
    except Exception as exc:
        log.warning("Could not initialize category embeddings via SentenceTransformer: %s. Using keyword fallback.", exc)
        return None


# ── Interest Extraction and Decay ─────────────────────────────────────────────

def _keyword_similarity(text: str, category: str) -> float:
    """Compute normalized keyword matching score between text and a category."""
    lower_text = text.lower()
    keywords = CATEGORY_KEYWORDS.get(category, [])
    if not keywords:
        return 0.0

    matches = 0
    total_weight = 0.0
    key_anchors = {"ai", "ml", "llm", "gpu", "chip", "chips", "gdp", "ipo", "vc", "fda", "un", "nato", "cyber", "tech"}
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, lower_text):
            matches += 1
            # Primary category name or key acronym carries high weight
            if kw == category.lower() or kw in key_anchors:
                total_weight += 1.8
            else:
                total_weight += 1.0

    if matches == 0:
        return 0.0

    score = min(1.0, total_weight / 2.0)
    return round(score, 3)


def extract_query_interests(
    query: str,
    excluded_categories: Optional[List[str]] = None,
    min_confidence_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Extract candidate interest categories from user query.
    Combines MiniLM cosine similarity with keyword frequency.
    Strictly filters out any excluded categories.
    """
    if not query or not query.strip():
        return []

    excluded_set = {c.strip().lower() for c in (excluded_categories or [])}
    results: List[Dict[str, Any]] = []

    cat_embeds = _get_category_embeddings()
    query_vec = None
    if cat_embeds is not None:
        try:
            from embedder import _get_model
            model = _get_model()
            qv = model.encode([query], convert_to_numpy=True)[0]
            norm = np.linalg.norm(qv)
            if norm > 0:
                query_vec = qv / norm
        except Exception as e:
            log.debug("Query vector encoding skipped: %s", e)
            query_vec = None

    for idx, category in enumerate(PREDEFINED_CATEGORIES):
        if category.lower() in excluded_set:
            continue

        kw_score = _keyword_similarity(query, category)
        cos_score = 0.0
        if query_vec is not None and cat_embeds is not None:
            cos_score = float(np.dot(query_vec, cat_embeds[idx]))
            cos_score = max(0.0, min(1.0, (cos_score + 0.1) / 1.1))

        # Hybrid fusion: prioritize keyword match if present, else semantic embedding
        if kw_score > 0.0:
            combined_score = max(kw_score, 0.7 * kw_score + 0.3 * cos_score)
        else:
            combined_score = cos_score

        if combined_score >= min_confidence_threshold:
            results.append({
                "category": category,
                "score": round(combined_score, 3),
                "confidence": round(min(1.0, combined_score * 0.9 + 0.1), 3),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def calculate_recency_decay(
    old_score: float,
    last_interaction_iso: Optional[str],
    half_life_days: float = 30.0,
) -> float:
    """
    Calculate recency decay using exponential half-life formula:
      S_decayed = S_old * exp(-lambda * delta_days), where lambda = ln(2) / half_life_days
    """
    if not last_interaction_iso:
        return old_score

    try:
        last_dt = datetime.fromisoformat(last_interaction_iso.replace("Z", "+00:00"))
        now_dt = datetime.utcnow()
        if last_dt.tzinfo is not None:
            last_dt = last_dt.replace(tzinfo=None)
        delta_days = max(0.0, (now_dt - last_dt).total_seconds() / 86400.0)
    except Exception:
        delta_days = 0.0

    decay_lambda = math.log(2) / half_life_days
    decay_factor = math.exp(-decay_lambda * delta_days)
    return max(0.05, round(old_score * decay_factor, 4))


# ── Profile Lifecycle & Database Sync ────────────────────────────────────────

def get_or_create_profile(user_id: int, session: Session) -> PersonalizationProfile:
    """Fetch existing PersonalizationProfile for user or initialize a new one with synced preferences."""
    profile = session.exec(
        select(PersonalizationProfile).where(PersonalizationProfile.user_id == user_id)
    ).first()

    if profile:
        return profile

    # Create new profile; seed with existing explicit UserPreference if available
    pref = session.exec(
        select(UserPreference).where(UserPreference.user_id == user_id)
    ).first()

    preferred_cats = []
    preferred_persp = []
    pref_lang = "English"
    pref_len = "detailed"

    if pref:
        try:
            preferred_cats = json.loads(pref.preferred_categories_json or "[]")
        except Exception:
            pass
        try:
            preferred_persp = json.loads(pref.preferred_perspectives_json or "[]")
        except Exception:
            pass
        pref_lang = pref.preferred_language or "English"
        pref_len = pref.preferred_length or "detailed"

    profile = PersonalizationProfile(
        user_id=user_id,
        enabled=True,
        inferred_interests_json="[]",
        excluded_interests_json="[]",
        preferred_categories_json=json.dumps(preferred_cats),
        preferred_perspectives_json=json.dumps(preferred_persp),
        preferred_language=pref_lang,
        preferred_length=pref_len,
        last_profile_update=utc_now(),
    )
    session.add(profile)
    session.commit()
    session.refresh(profile)
    log.info("Created new PersonalizationProfile for user_id=%d", user_id)
    return profile


def update_user_interests_incremental(
    user_id: int,
    query: str,
    session: Optional[Session] = None,
) -> None:
    """
    Incrementally update inferred user interests after a query.
    Applies recency decay, updates frequencies, and updates lifecycle statuses:
      - 'emerging': 1-2 interactions, confidence < 0.45
      - 'recurring': >= 2 interactions, score >= 0.45
      - 'strong': score >= 0.70 and confidence >= 0.70
      - 'declining': inactive > 45 days, score < 0.35
    Respects excluded_interests strictly (never re-infers excluded topics).
    """
    should_close_session = False
    if session is None:
        session = next(get_session())
        should_close_session = True

    try:
        profile = get_or_create_profile(user_id, session)
        if not profile.enabled:
            return

        try:
            excluded = set(json.loads(profile.excluded_interests_json or "[]"))
        except Exception:
            excluded = set()

        try:
            current_inferred = json.loads(profile.inferred_interests_json or "[]")
        except Exception:
            current_inferred = []

        inferred_map: Dict[str, Dict[str, Any]] = {
            item["category"]: item for item in current_inferred if item.get("category")
        }

        detected = extract_query_interests(
            query=query,
            excluded_categories=list(excluded),
            min_confidence_threshold=0.38,
        )

        now_iso = utc_now().isoformat()
        detected_categories = {d["category"] for d in detected}

        # 1. Update detected interests
        for d in detected:
            cat = d["category"]
            if cat in excluded:
                continue

            match_score = d["score"]

            if cat in inferred_map:
                existing = inferred_map[cat]
                decayed_score = calculate_recency_decay(
                    existing.get("score", 0.5),
                    existing.get("last_interaction"),
                    half_life_days=30.0,
                )
                new_interactions = existing.get("supporting_interactions", 1) + 1
                new_score = min(1.0, round(decayed_score + match_score * 0.32, 3))
                new_confidence = min(
                    1.0,
                    round(0.20 + 0.15 * min(new_interactions, 5) + 0.25 * new_score, 3),
                )
                existing["score"] = new_score
                existing["confidence"] = new_confidence
                existing["supporting_interactions"] = new_interactions
                existing["last_interaction"] = now_iso

                # Classify status
                if new_score >= 0.70 and new_confidence >= 0.70:
                    existing["status"] = "strong"
                elif new_interactions >= 2 and new_score >= 0.45:
                    existing["status"] = "recurring"
                else:
                    existing["status"] = "emerging"
            else:
                inferred_map[cat] = {
                    "category": cat,
                    "score": round(match_score * 0.65, 3),
                    "confidence": round(d["confidence"] * 0.60, 3),
                    "supporting_interactions": 1,
                    "first_detected": now_iso,
                    "last_interaction": now_iso,
                    "status": "emerging",
                    "source": "chat_history",
                }

        # 2. Check for declining status on undetected categories
        for cat, item in list(inferred_map.items()):
            if cat not in detected_categories:
                decayed = calculate_recency_decay(
                    item.get("score", 0.5),
                    item.get("last_interaction"),
                    half_life_days=30.0,
                )
                item["score"] = decayed
                # Check inactivity duration
                try:
                    last_dt = datetime.fromisoformat(item["last_interaction"].replace("Z", "+00:00")).replace(tzinfo=None)
                    inactive_days = (datetime.utcnow() - last_dt).total_seconds() / 86400.0
                except Exception:
                    inactive_days = 0.0

                if inactive_days > 45.0 or decayed < 0.25:
                    item["status"] = "declining"

        # Filter out negligible declining interests with 1 interaction
        updated_list = [
            v for v in inferred_map.values()
            if not (v.get("status") == "declining" and v.get("supporting_interactions", 1) <= 1 and v.get("score", 0) < 0.20)
        ]
        # Sort by score descending
        updated_list.sort(key=lambda x: (x.get("score", 0), x.get("confidence", 0)), reverse=True)

        profile.inferred_interests_json = json.dumps(updated_list)
        profile.last_profile_update = utc_now()
        session.add(profile)
        session.commit()
        log.info(
            "Updated user_id=%d personalization: %d active interests (detected: %s)",
            user_id,
            len(updated_list),
            list(detected_categories),
        )
    except Exception as exc:
        log.exception("Failed to update user interests incrementally: %s", exc)
        if session:
            session.rollback()
    finally:
        if should_close_session and session:
            session.close()


# ── Decoupled Personalization Scoring ─────────────────────────────────────────

def compute_personalization_scores(
    query: str,
    profile: Optional[PersonalizationProfile],
) -> Dict[str, float]:
    """
    Compute decoupled personalization scores S_pers(p) in [0.0, 1.0] for all 8 personas:
      S_pers(p) = 0.50 * S_explicit(p) + 0.35 * S_historical(p) + 0.15 * S_perspective(p)

    If profile is None or disabled, returns neutral scores (0.5 for all personas).
    """
    neutral_scores = {p["id"]: 0.50 for p in PERSONAS}
    if not profile or not profile.enabled:
        return neutral_scores

    try:
        preferred_cats = set(json.loads(profile.preferred_categories_json or "[]"))
    except Exception:
        preferred_cats = set()

    try:
        preferred_persp = set(json.loads(profile.preferred_perspectives_json or "[]"))
    except Exception:
        preferred_persp = set()

    try:
        inferred = json.loads(profile.inferred_interests_json or "[]")
    except Exception:
        inferred = []

    # Map inferred interests with weights
    historical_weights: Dict[str, float] = {}
    for item in inferred:
        cat = item.get("category")
        status = item.get("status", "emerging")
        if status == "declining":
            continue
        sc = float(item.get("score", 0.5))
        conf = float(item.get("confidence", 0.5))
        # Strong and recurring interests get amplified
        multiplier = 1.3 if status == "strong" else (1.1 if status == "recurring" else 0.8)
        historical_weights[cat] = min(1.0, sc * conf * multiplier)

    # Current query category detections
    query_interests = extract_query_interests(query)
    query_cat_weights = {d["category"]: d["score"] for d in query_interests}

    persona_scores: Dict[str, float] = {}

    for persona in PERSONAS:
        pid = persona["id"]
        affinity = PERSONA_AFFINITIES.get(pid, {})
        aff_cats = set(affinity.get("categories", []))
        aff_persp = set(affinity.get("perspectives", []))

        # 1. S_explicit(p): Explicit categories + perspectives alignment
        cat_overlap = len(preferred_cats.intersection(aff_cats))
        persp_overlap = len(preferred_persp.intersection(aff_persp))
        s_explicit = 0.5
        if preferred_cats or preferred_persp:
            c_score = (cat_overlap / max(1, len(preferred_cats))) if preferred_cats else 0.5
            p_score = (persp_overlap / max(1, len(preferred_persp))) if preferred_persp else 0.5
            s_explicit = 0.5 * c_score + 0.5 * p_score

        # 2. S_historical(p): Alignment with learned recurring interests
        s_historical = 0.5
        if historical_weights:
            cat_hits = [historical_weights[c] for c in aff_cats if c in historical_weights]
            if cat_hits:
                s_historical = min(1.0, sum(cat_hits) / len(cat_hits) + 0.2)
            else:
                s_historical = 0.35

        # 3. S_perspective(p): Alignment with the immediate query's topics
        s_perspective = 0.5
        if query_cat_weights:
            q_hits = [query_cat_weights[c] for c in aff_cats if c in query_cat_weights]
            if q_hits:
                s_perspective = min(1.0, max(q_hits) + 0.1)
            else:
                s_perspective = 0.40

        # Weighted combination: 50% explicit + 35% historical + 15% immediate perspective
        s_pers = 0.50 * s_explicit + 0.35 * s_historical + 0.15 * s_perspective
        persona_scores[pid] = round(max(0.1, min(1.0, s_pers)), 4)

    return persona_scores


def build_personalization_context(
    profile: Optional[PersonalizationProfile],
    query: str,
    persona_scores: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """
    Build a compact context injection block for persona prompts.
    Provides personal focus instructions without overriding factual grounding.
    """
    if not profile or not profile.enabled:
        return None

    try:
        preferred_cats = json.loads(profile.preferred_categories_json or "[]")
    except Exception:
        preferred_cats = []

    try:
        preferred_persp = json.loads(profile.preferred_perspectives_json or "[]")
    except Exception:
        preferred_persp = []

    try:
        inferred = json.loads(profile.inferred_interests_json or "[]")
    except Exception:
        inferred = []

    # Get active recurring or strong interests
    active_interests = [
        item["category"]
        for item in inferred
        if item.get("status") in ("strong", "recurring")
    ][:4]

    # If no explicit or implicit preferences exist yet, omit injection
    if not preferred_cats and not preferred_persp and not active_interests and profile.preferred_language == "English":
        return None

    lines = ["[USER PERSONALIZATION CONTEXT]"]

    if preferred_persp:
        lines.append(f"- Preferred Analytical Angles: {', '.join(preferred_persp[:4])}")

    combined_interests = list(dict.fromkeys(preferred_cats + active_interests))
    if combined_interests:
        lines.append(f"- User Topic Interests: {', '.join(combined_interests[:5])}")

    if profile.preferred_language and profile.preferred_language.lower() != "english":
        lines.append(f"- Preferred Language: {profile.preferred_language}")

    if profile.preferred_length and profile.preferred_length != "detailed":
        lines.append(f"- Preferred Response Depth: {profile.preferred_length}")

    lines.append(
        "- DIRECTIVE: Calibrate analysis emphasis, technical depth, and synthesis angle "
        "toward the user's preferred perspectives while maintaining strict factual grounding in retrieved evidence."
    )

    return "\n".join(lines)


def generate_personalization_summary(
    profile: Optional[PersonalizationProfile],
    query_categories: Optional[List[str]] = None,
) -> str:
    """Generate a clean, user-facing summary badge text."""
    if not profile or not profile.enabled:
        return ""

    try:
        inferred = json.loads(profile.inferred_interests_json or "[]")
    except Exception:
        inferred = []

    strong_cats = [
        item["category"]
        for item in inferred
        if item.get("status") in ("strong", "recurring")
    ]

    try:
        explicit_cats = json.loads(profile.preferred_categories_json or "[]")
    except Exception:
        explicit_cats = []

    relevant = []
    if query_categories:
        for qc in query_categories:
            if qc in strong_cats or qc in explicit_cats:
                relevant.append(qc)

    if not relevant:
        relevant = (strong_cats + explicit_cats)[:2]

    if relevant:
        return f"Personalized for your interests in {', '.join(relevant)}"
    elif profile.preferred_perspectives_json and profile.preferred_perspectives_json != "[]":
        try:
            persp = json.loads(profile.preferred_perspectives_json)[:2]
            return f"Personalized with {', '.join(persp)} perspective"
        except Exception:
            pass

    return "Personalized analysis enabled"
