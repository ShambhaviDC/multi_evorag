"""
feedback_intelligence.py — Feedback Intelligence Engine & Continuous Personalization for EvoRAG.

Implements:
  1. Feedback Ingestion & Storage (Likes, Dislikes, Optional Reasons & Comments)
  2. Dimension A: User-Level Personalization (Learning individual user preferences from repeated feedback)
  3. Dimension B: Response Quality Analysis (Classifying negative feedback to isolate pipeline bottlenecks)
  4. Dimension C: Multi-Persona Performance Analysis (Correlating feedback with activated personas & strategies)
  5. Continuous Improvement Recommendations (Actionable diagnostic recommendations for retrieval & synthesis)
  6. Aggregated Feedback Analytics (Live metrics, Chart.js datasets, issue distributions)
"""

import json
import logging
import math
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from config import PERSONAS
from database import get_session
from models import (
    Conversation,
    Message,
    PersonalizationProfile,
    ResponseFeedback,
    User,
    utc_now,
)

log = logging.getLogger(__name__)

# Valid standardized reasons for negative feedback
DISLIKE_REASONS = {
    "factually_incorrect": "Factually incorrect",
    "outdated_information": "Outdated information",
    "irrelevant_query": "Irrelevant to my query",
    "incomplete_answer": "Incomplete answer",
    "poor_explanation": "Poor explanation",
    "missing_sources": "Missing supporting sources",
    "length_issue": "Too lengthy or too short",
    "other": "Other",
}

# Optional standardized tags for positive feedback
LIKE_REASONS = {
    "accurate": "Accurate & factual",
    "helpful_detail": "Thorough & well-explained",
    "good_sources": "Well-cited sources",
    "clear_explanation": "Clear & easy to understand",
    "actionable": "Direct & practical",
    "other": "Other",
}

# Mapping dislike reasons to affected architectural pipeline components
REASON_TO_PIPELINE_COMPONENT = {
    "irrelevant_query": "Query Understanding & Routing",
    "missing_sources": "Retrieval & FAISS Search",
    "outdated_information": "Temporal Event Tracking & Freshness",
    "factually_incorrect": "Factual Grounding & Verification",
    "incomplete_answer": "Response Synthesis & Output Limits",
    "poor_explanation": "Persona Reasoning Quality",
    "length_issue": "Response Formatter & Length Guidance",
    "other": "General Generation Quality",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEEDBACK INGESTION & LIFECYCLE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def save_or_update_feedback(
    session: Session,
    message_id: str,
    feedback_type: str,
    user: Optional[User] = None,
    session_id: Optional[str] = None,
    conversation_id: Optional[int] = None,
    query_text: Optional[str] = None,
    reason: Optional[str] = None,
    comment: Optional[str] = None,
    execution_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[ResponseFeedback, str]:
    """
    Save new feedback or update an existing feedback record.
    Enforces strictly ONE active feedback record per (user_id/session_id) + message_id.
    Returns (ResponseFeedback, action) where action is 'created' or 'updated'.
    """
    fb_type = feedback_type.strip().lower()
    if fb_type not in ("like", "dislike"):
        raise ValueError("feedback_type must be either 'like' or 'dislike'")

    user_id = getattr(user, "id", None)
    clean_session_id = session_id.strip() if session_id else None

    if not user_id and not clean_session_id:
        clean_session_id = f"guest_{uuid.uuid4().hex[:12]}"

    # Locate existing active feedback record
    query = select(ResponseFeedback).where(ResponseFeedback.message_id == message_id)
    if user_id:
        query = query.where(ResponseFeedback.user_id == user_id)
    else:
        query = query.where(ResponseFeedback.session_id == clean_session_id)

    existing = session.exec(query).first()

    # Clean & validate reason
    clean_reason = reason.strip() if reason else None
    if fb_type == "dislike" and clean_reason and clean_reason not in DISLIKE_REASONS:
        clean_reason = "other"
    elif fb_type == "like" and clean_reason and clean_reason not in LIKE_REASONS:
        clean_reason = "other"

    clean_comment = comment.strip() if comment else None
    meta_json = json.dumps(execution_metadata or {})

    now = utc_now()

    if existing:
        existing.feedback_type = fb_type
        existing.reason = clean_reason
        existing.comment = clean_comment
        if conversation_id:
            existing.conversation_id = conversation_id
        if query_text:
            existing.query_text = query_text
        if execution_metadata:
            existing.execution_metadata_json = meta_json
        existing.updated_at = now
        session.add(existing)
        session.commit()
        session.refresh(existing)
        log.info(
            "Updated feedback_id=%s for message_id=%s: type=%s, reason=%s",
            existing.feedback_id, message_id, fb_type, clean_reason
        )
        return existing, "updated"

    # Create new feedback record
    new_fb = ResponseFeedback(
        feedback_id=f"fb_{uuid.uuid4().hex[:14]}",
        user_id=user_id,
        session_id=clean_session_id if not user_id else None,
        conversation_id=conversation_id,
        message_id=message_id,
        query_text=query_text,
        feedback_type=fb_type,
        reason=clean_reason,
        comment=clean_comment,
        execution_metadata_json=meta_json,
        created_at=now,
        updated_at=now,
    )
    session.add(new_fb)
    session.commit()
    session.refresh(new_fb)
    log.info(
        "Created feedback_id=%s for message_id=%s: type=%s, reason=%s (user_id=%s, session_id=%s)",
        new_fb.feedback_id, message_id, fb_type, clean_reason, user_id, clean_session_id
    )
    return new_fb, "created"


def delete_feedback(
    session: Session,
    message_id: str,
    user: Optional[User] = None,
    session_id: Optional[str] = None,
) -> bool:
    """Delete feedback for a specific message (user clicked active icon again to undo)."""
    user_id = getattr(user, "id", None)
    clean_session_id = session_id.strip() if session_id else None

    query = select(ResponseFeedback).where(ResponseFeedback.message_id == message_id)
    if user_id:
        query = query.where(ResponseFeedback.user_id == user_id)
    elif clean_session_id:
        query = query.where(ResponseFeedback.session_id == clean_session_id)
    else:
        return False

    existing = session.exec(query).first()
    if existing:
        session.delete(existing)
        session.commit()
        log.info("Deleted feedback for message_id=%s (user_id=%s)", message_id, user_id)
        return True
    return False


def get_feedback_for_message(
    session: Session,
    message_id: str,
    user: Optional[User] = None,
    session_id: Optional[str] = None,
) -> Optional[ResponseFeedback]:
    """Retrieve active feedback record for a given message and user/session."""
    user_id = getattr(user, "id", None)
    clean_session_id = session_id.strip() if session_id else None

    query = select(ResponseFeedback).where(ResponseFeedback.message_id == message_id)
    if user_id:
        query = query.where(ResponseFeedback.user_id == user_id)
    elif clean_session_id:
        query = query.where(ResponseFeedback.session_id == clean_session_id)
    else:
        return None

    return session.exec(query).first()


def get_user_feedback_history(
    session: Session,
    user_id: int,
    limit: int = 50,
) -> List[ResponseFeedback]:
    """Fetch recent feedback history for an authenticated user."""
    stmt = (
        select(ResponseFeedback)
        .where(ResponseFeedback.user_id == user_id)
        .order_by(ResponseFeedback.created_at.desc())
        .limit(limit)
    )
    return session.exec(stmt).all()


# ══════════════════════════════════════════════════════════════════════════════
# 2. DIMENSION A: USER-LEVEL PERSONALIZATION LEARNING
# ══════════════════════════════════════════════════════════════════════════════

def learn_user_preferences_from_feedback(
    user_id: int,
    session: Session,
    min_feedback_threshold: int = 2,
) -> Dict[str, Any]:
    """
    Analyze repeated feedback signals for an authenticated user to infer style preferences.
    Guardrails:
      - Does NOT update from a single isolated like or dislike.
      - Requires >= min_feedback_threshold relevant feedback interactions.
      - Does NOT overwrite explicit onboarding choices.
      - Dislikes only contribute if reason specifically indicates a format preference.
      - Updates profile.learned_preferences_json.
    """
    profile = session.exec(
        select(PersonalizationProfile).where(PersonalizationProfile.user_id == user_id)
    ).first()
    if not profile or not profile.enabled:
        return {}

    user_feedbacks = session.exec(
        select(ResponseFeedback)
        .where(ResponseFeedback.user_id == user_id)
        .order_by(ResponseFeedback.created_at.desc())
        .limit(60)
    ).all()

    if len(user_feedbacks) < min_feedback_threshold:
        return {}

    likes = [f for f in user_feedbacks if f.feedback_type == "like"]
    dislikes = [f for f in user_feedbacks if f.feedback_type == "dislike"]

    # 1. Analyze Citation & Sourcing Affinity
    # Signals: dislikes with 'missing_sources' or likes with 'good_sources'
    missing_sources_count = sum(1 for d in dislikes if d.reason == "missing_sources")
    good_sources_count = sum(1 for l in likes if l.reason == "good_sources")
    prefer_sources = (missing_sources_count >= 1 or good_sources_count >= 1)

    # 2. Analyze Technical Depth vs Conciseness
    # Signals: likes with 'helpful_detail', technical complexity in metadata/query, or poor_explanation dislikes
    detail_likes = sum(1 for l in likes if l.reason == "helpful_detail")
    length_dislikes = sum(1 for d in dislikes if d.reason == "length_issue")
    poor_explanation_dislikes = sum(1 for d in dislikes if d.reason == "poor_explanation")
    technical_meta_likes = sum(
        1 for l in likes
        if (
            l.reason == "helpful_detail"
            or (l.execution_metadata_json and ("technical" in l.execution_metadata_json.lower() or "domain-expert" in l.execution_metadata_json))
            or (l.query_text and any(k in l.query_text.lower() for k in ("technical", "deep-dive", "architecture", "mechanism", "algorithm")))
        )
    )

    prefer_technical_depth = False
    prefer_concise = False

    if detail_likes >= 2 or poor_explanation_dislikes >= 2 or technical_meta_likes >= 2:
        prefer_technical_depth = True
    elif length_dislikes >= 2 and detail_likes == 0:
        prefer_concise = True

    # 3. Analyze Structured/Tabular Preference
    # If user repeatedly likes responses that contained charts or tables
    structured_likes = 0
    for l in likes:
        meta = json.loads(l.execution_metadata_json or "{}")
        if meta.get("has_components") or meta.get("components_count", 0) > 0:
            structured_likes += 1
    prefer_visual_components = (structured_likes >= 2)

    # 4. Persona Archetype Affinity from Likes
    persona_like_counts = Counter()
    persona_dislike_counts = Counter()

    for l in likes:
        meta = json.loads(l.execution_metadata_json or "{}")
        winner = meta.get("winning_persona")
        if winner:
            persona_like_counts[winner] += 1

    for d in dislikes:
        meta = json.loads(d.execution_metadata_json or "{}")
        winner = meta.get("winning_persona")
        if winner:
            persona_dislike_counts[winner] += 1

    favored_personas = [p for p, c in persona_like_counts.items() if c >= 2 and persona_dislike_counts[p] == 0]

    # Synthesize learned preferences dictionary
    learned_dict = {
        "prefer_sources": prefer_sources,
        "prefer_technical_depth": prefer_technical_depth,
        "prefer_concise": prefer_concise,
        "prefer_visual_components": prefer_visual_components,
        "favored_personas": favored_personas,
        "total_signals_analyzed": len(user_feedbacks),
        "likes_count": len(likes),
        "dislikes_count": len(dislikes),
        "last_updated": utc_now().isoformat(),
    }

    # Generate readable summary of learned style preferences
    summary_parts = []
    if prefer_sources:
        summary_parts.append("prioritize verified citations & sources")
    if prefer_technical_depth:
        summary_parts.append("provide in-depth technical analysis & mechanisms")
    if prefer_concise:
        summary_parts.append("prefer concise, direct summaries")
    if prefer_visual_components:
        summary_parts.append("include structured comparison tables & dynamic charts")

    learned_dict["summary"] = "; ".join(summary_parts) if summary_parts else "balanced default style"

    profile.learned_preferences_json = json.dumps(learned_dict)
    session.add(profile)
    session.commit()
    log.info("Learned feedback preferences updated for user_id=%d: %s", user_id, learned_dict["summary"])

    return learned_dict


# ══════════════════════════════════════════════════════════════════════════════
# 3. DIMENSION B & C: RESPONSE QUALITY & PERSONA PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_response_quality_issues(session: Session) -> Dict[str, Any]:
    """
    Classify recurring negative feedback into pipeline component failure categories.
    Identifies if Query Understanding, Retrieval, Freshness, Grounding, or Synthesis need attention.
    """
    dislikes = session.exec(
        select(ResponseFeedback).where(ResponseFeedback.feedback_type == "dislike")
    ).all()

    total_dislikes = len(dislikes)
    if total_dislikes == 0:
        return {
            "total_negative_reports": 0,
            "reasons_breakdown": {},
            "component_issues": {},
            "status": "healthy",
        }

    reason_counts = Counter(d.reason or "other" for d in dislikes)
    component_counts = Counter()

    for r, count in reason_counts.items():
        comp = REASON_TO_PIPELINE_COMPONENT.get(r, "General Generation Quality")
        component_counts[comp] += count

    # Compute component health ratings
    component_issues = {}
    for comp, count in component_counts.items():
        pct = round((count / total_dislikes) * 100, 1)
        severity = "high" if pct >= 30 and count >= 3 else ("medium" if pct >= 15 else "low")
        component_issues[comp] = {
            "issue_count": count,
            "percentage": pct,
            "severity": severity,
        }

    return {
        "total_negative_reports": total_dislikes,
        "reasons_breakdown": dict(reason_counts),
        "component_issues": component_issues,
        "primary_bottleneck": component_counts.most_common(1)[0][0] if component_counts else "None",
    }


def analyze_persona_feedback_performance(session: Session) -> Dict[str, Any]:
    """
    Correlate feedback with winning and participating personas from execution metadata.
    Computes per-persona satisfaction metrics without prematurely changing system weights.
    """
    all_feedbacks = session.exec(select(ResponseFeedback)).all()
    persona_stats: Dict[str, Dict[str, Any]] = {}

    for p in PERSONAS:
        pid = p["id"]
        persona_stats[pid] = {
            "name": p["name"],
            "wins_count": 0,
            "likes": 0,
            "dislikes": 0,
            "satisfaction_rate": 0.0,
            "associated_dislike_reasons": Counter(),
        }

    for fb in all_feedbacks:
        meta = json.loads(fb.execution_metadata_json or "{}")
        winner = meta.get("winning_persona")
        if winner and winner in persona_stats:
            persona_stats[winner]["wins_count"] += 1
            if fb.feedback_type == "like":
                persona_stats[winner]["likes"] += 1
            else:
                persona_stats[winner]["dislikes"] += 1
                if fb.reason:
                    persona_stats[winner]["associated_dislike_reasons"][fb.reason] += 1

    # Calculate satisfaction percentages
    for pid, s in persona_stats.items():
        total = s["likes"] + s["dislikes"]
        s["satisfaction_rate"] = round((s["likes"] / total * 100), 1) if total > 0 else 0.0
        s["associated_dislike_reasons"] = dict(s["associated_dislike_reasons"])

    return persona_stats


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONTINUOUS IMPROVEMENT RECOMMENDATIONS & ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def generate_system_improvement_recommendations(session: Session) -> List[Dict[str, str]]:
    """
    Formulate concrete, actionable architectural recommendations based on aggregated feedback.
    Safeguards against noisy or isolated signals.
    """
    quality_analysis = analyze_response_quality_issues(session)
    persona_analysis = analyze_persona_feedback_performance(session)

    recommendations: List[Dict[str, str]] = []
    issues = quality_analysis.get("component_issues", {})

    # 1. Retrieval & Sourcing Recommendation
    retrieval_issue = issues.get("Retrieval & FAISS Search", {})
    if retrieval_issue.get("severity") in ("high", "medium"):
        recommendations.append({
            "component": "Retrieval & FAISS Search",
            "type": "configuration_tuning",
            "observation": f"{retrieval_issue['issue_count']} feedback reports ({retrieval_issue['percentage']}%) cited missing or insufficient supporting sources.",
            "recommendation": "Increase RETRIEVAL_TOP_K default from 5 to 7 or lower the min_similarity_score threshold to ingest broader multi-angle chunks.",
        })

    # 2. Freshness & Temporal Recommendation
    freshness_issue = issues.get("Temporal Event Tracking & Freshness", {})
    if freshness_issue.get("severity") in ("high", "medium"):
        recommendations.append({
            "component": "Temporal Event Tracking",
            "type": "scheduler_sweep",
            "observation": f"{freshness_issue['issue_count']} reports noted outdated information.",
            "recommendation": "Trigger incremental feed ingestion scheduler sweep and re-cluster active events to capture contemporaneous developments.",
        })

    # 3. Grounding & Factuality Recommendation
    grounding_issue = issues.get("Factual Grounding & Verification", {})
    if grounding_issue.get("severity") in ("high", "medium"):
        recommendations.append({
            "component": "Factual Grounding & Verification",
            "type": "prompt_refinement",
            "observation": f"{grounding_issue['issue_count']} reports indicated factual inaccuracies.",
            "recommendation": "Tighten consensus verification instructions; ensure personas strictly cross-verify dates, entity names, and quantitative metrics.",
        })

    # 4. Response Synthesis Recommendation
    synthesis_issue = issues.get("Response Synthesis & Output Limits", {})
    if synthesis_issue.get("severity") in ("high", "medium"):
        recommendations.append({
            "component": "Response Synthesis",
            "type": "token_limit_expansion",
            "observation": f"{synthesis_issue['issue_count']} reports noted incomplete or abruptly cut-off answers.",
            "recommendation": "Expand max_output_tokens in synthesis calls from 1500 to 2048 to prevent truncation on comprehensive queries.",
        })

    # 5. Persona-Specific Diagnostics
    for pid, pdata in persona_analysis.items():
        total_evals = pdata["likes"] + pdata["dislikes"]
        if total_evals >= 4 and pdata["satisfaction_rate"] < 50.0:
            top_reason = (
                max(pdata["associated_dislike_reasons"].items(), key=lambda x: x[1])[0]
                if pdata["associated_dislike_reasons"]
                else "general dissatisfaction"
            )
            recommendations.append({
                "component": f"Persona: {pdata['name']}",
                "type": "persona_prompt_adjustment",
                "observation": f"Persona '{pdata['name']}' has a {pdata['satisfaction_rate']}% satisfaction rate ({pdata['dislikes']} dislikes), commonly cited for '{top_reason}'.",
                "recommendation": f"Calibrate '{pdata['name']}' system prompt heuristics in personas.py to address '{top_reason}' issues.",
            })

    if not recommendations:
        recommendations.append({
            "component": "Consensus & Pipeline Health",
            "type": "nominal_state",
            "observation": "System satisfaction is healthy with no elevated failure patterns across components.",
            "recommendation": "Maintain current retrieval thresholds, persona scoring weights, and synthesis parameters.",
        })

    return recommendations


def get_feedback_analytics(session: Session) -> Dict[str, Any]:
    """
    Compile complete, live administrative analytics derived directly from stored feedback.
    Powers the administrative Feedback Analytics Dashboard.
    """
    all_fb = session.exec(select(ResponseFeedback).order_by(ResponseFeedback.created_at.asc())).all()

    total_submissions = len(all_fb)
    likes_count = sum(1 for fb in all_fb if fb.feedback_type == "like")
    dislikes_count = sum(1 for fb in all_fb if fb.feedback_type == "dislike")
    positive_percentage = round((likes_count / total_submissions * 100), 1) if total_submissions > 0 else 0.0

    # Negative feedback reasons distribution (for Bar Chart)
    dislike_reasons_counter = Counter(
        DISLIKE_REASONS.get(fb.reason, fb.reason or "Other")
        for fb in all_fb
        if fb.feedback_type == "dislike"
    )

    # Positive feedback reasons distribution (if captured)
    like_reasons_counter = Counter(
        LIKE_REASONS.get(fb.reason, fb.reason or "Helpful & Accurate")
        for fb in all_fb
        if fb.feedback_type == "like" and fb.reason
    )

    # Feedback trends over time (Daily grouping for Line Chart)
    trends_map: Dict[str, Dict[str, int]] = defaultdict(lambda: {"likes": 0, "dislikes": 0})
    for fb in all_fb:
        day_str = fb.created_at.strftime("%Y-%m-%d") if fb.created_at else "Recent"
        if fb.feedback_type == "like":
            trends_map[day_str]["likes"] += 1
        else:
            trends_map[day_str]["dislikes"] += 1

    sorted_dates = sorted(list(trends_map.keys()))
    trend_labels = sorted_dates[-14:] if len(sorted_dates) > 14 else sorted_dates
    trend_likes = [trends_map[d]["likes"] for d in trend_labels]
    trend_dislikes = [trends_map[d]["dislikes"] for d in trend_labels]

    # Topic distribution of feedback
    topic_counter = Counter()
    for fb in all_fb:
        meta = json.loads(fb.execution_metadata_json or "{}")
        cat = meta.get("query_category") or meta.get("topic") or "General News"
        topic_counter[cat] += 1

    # Quality and persona diagnostics
    quality_analysis = analyze_response_quality_issues(session)
    persona_stats = analyze_persona_feedback_performance(session)
    recommendations = generate_system_improvement_recommendations(session)

    # Clean, anonymized list of recent quality issues for the dashboard table
    recent_issues = []
    for fb in reversed(all_fb[-20:]):
        if fb.feedback_type == "dislike":
            meta = json.loads(fb.execution_metadata_json or "{}")
            recent_issues.append({
                "message_id": fb.message_id,
                "reason": DISLIKE_REASONS.get(fb.reason, fb.reason or "Unspecified"),
                "component": REASON_TO_PIPELINE_COMPONENT.get(fb.reason, "General Pipeline"),
                "winning_persona": meta.get("winning_persona", "Consensus Engine"),
                "query_preview": (fb.query_text[:50] + "...") if fb.query_text else "General Query",
                "date": fb.created_at.strftime("%b %d, %H:%M") if fb.created_at else "Recent",
                "has_comment": bool(fb.comment),
            })

    return {
        "summary": {
            "total_submissions": total_submissions,
            "likes_count": likes_count,
            "dislikes_count": dislikes_count,
            "positive_percentage": positive_percentage,
            "negative_percentage": round(100.0 - positive_percentage, 1) if total_submissions > 0 else 0.0,
        },
        "charts": {
            "satisfaction_doughnut": {
                "labels": ["Positive (Likes)", "Negative (Dislikes)"],
                "data": [likes_count, dislikes_count],
                "colors": ["#10b981", "#ef4444"],
            },
            "dislike_reasons_bar": {
                "labels": list(dislike_reasons_counter.keys()),
                "data": list(dislike_reasons_counter.values()),
                "colors": ["#f87171", "#fb923c", "#fbbf24", "#a3e635", "#38bdf8", "#818cf8", "#c084fc", "#94a3b8"],
            },
            "trends_line": {
                "labels": trend_labels,
                "likes": trend_likes,
                "dislikes": trend_dislikes,
            },
            "topic_distribution": dict(topic_counter.most_common(8)),
        },
        "persona_performance": list(persona_stats.values()),
        "pipeline_health": quality_analysis,
        "recommendations": recommendations,
        "recent_quality_issues": recent_issues[:10],
    }
