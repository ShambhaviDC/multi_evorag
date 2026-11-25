"""
persona_evolution/fitness.py — Multi-Factor Persona Fitness Function

Implements the multi-dimensional evolutionary fitness function:
  Fitness = 0.30 * Feedback + 0.25 * Evidence + 0.20 * Relevance + 0.15 * Completeness + 0.10 * Consistency

Key Features:
- User feedback weighting with penalty attribution (factually_incorrect, missing_sources, etc.)
- Claim-level evidence attribution grounding from XAI records & AnalysisSnapshots
- Peer scoring integration across factual_grounding, reasoning_quality, completeness
- Top-K consensus consistency metric
- Sample-size uncertainty discounting when N < 8
- Detailed breakdown report per persona
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from sqlmodel import Session, select

from database import engine
from models import AnalysisSnapshot, ResponseFeedback

log = logging.getLogger(__name__)

SCORE_STORE_PATH = Path("score_store.json")
MIN_SAMPLE_THRESHOLD = 8  # Below this, apply square-root uncertainty discount
PRIOR_FITNESS = 5.0       # Neutral Bayesian prior for unproven personas


class FitnessReport(BaseModel):
    """Detailed multi-factor fitness calculation report for a persona."""
    persona_id: str
    composite_fitness: float = Field(..., ge=0.0, le=10.0)
    raw_fitness: float = Field(..., ge=0.0, le=10.0)
    sample_count: int
    is_statistically_valid: bool
    uncertainty_discount_applied: bool
    discount_factor: float
    dimensions: Dict[str, float] = Field(
        ...,
        description="Individual score components: feedback, evidence, relevance, completeness, consistency"
    )
    penalized_reasons: List[str] = Field(default_factory=list)
    times_in_top_k: int = 0
    top_k_rate: float = 0.0


def _load_score_store_entries(path: Path = SCORE_STORE_PATH) -> List[Dict[str, Any]]:
    """Load evaluation records from score_store.json."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        log.warning("Could not read %s for fitness evaluation: %s", path, exc)
        return []


def calculate_persona_fitness(
    persona_id: str,
    session: Optional[Session] = None,
    window: int = 25,
) -> FitnessReport:
    """
    Calculate the multi-factor fitness score for a specific persona.

    Weights:
      - 30% User Feedback
      - 25% Evidence Attribution Quality
      - 20% Reasoning Quality (Peer Relevance)
      - 15% Completeness (Peer Evaluation)
      - 10% Consensus Consistency (Top-3 Selection Frequency)
    """
    close_session = False
    if session is None:
        session = Session(engine)
        close_session = True

    try:
        history = _load_score_store_entries()
        
        # 1. Collect peer evaluations for this persona
        persona_records: List[Dict[str, Any]] = []
        for entry in history:
            ps = entry.get("persona_scores", {}).get(persona_id)
            if isinstance(ps, dict):
                persona_records.append(ps)

        sample_count = len(persona_records)
        recent_records = persona_records[-window:] if persona_records else []

        # Peer Scores (Relevance & Completeness)
        relevance_scores: List[float] = []
        completeness_scores: List[float] = []
        factual_scores: List[float] = []
        times_in_top_k = 0

        for r in recent_records:
            if r.get("made_top_k"):
                times_in_top_k += 1
            dims = r.get("dimensions", {})
            if dims:
                if "reasoning_quality" in dims:
                    relevance_scores.append(float(dims["reasoning_quality"]))
                if "completeness" in dims:
                    completeness_scores.append(float(dims["completeness"]))
                if "factual_grounding" in dims:
                    factual_scores.append(float(dims["factual_grounding"]))
            elif r.get("avg_score") is not None:
                relevance_scores.append(float(r["avg_score"]))
                completeness_scores.append(float(r["avg_score"]))

        s_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 6.5
        s_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 6.5
        top_k_rate = times_in_top_k / len(recent_records) if recent_records else 0.0
        s_consistency = min(10.0, top_k_rate * 10.0)

        # 2. Evidence Attribution Component (from Snapshots / XAI records)
        s_evidence = 7.0
        try:
            snapshots = session.exec(
                select(AnalysisSnapshot).order_by(AnalysisSnapshot.id.desc()).limit(30)
            ).all()
            supported_claims = 0
            total_claims = 0
            for snap in snapshots:
                try:
                    top_p = json.loads(snap.top_personas_json or "[]")
                    if persona_id in top_p:
                        xai = json.loads(snap.xai_record_json or "{}")
                        claims = xai.get("evidence_attribution", [])
                        for c in claims:
                            total_claims += 1
                            if c.get("verification_status") in ("Supported", "Partially supported"):
                                supported_claims += 1
                except Exception:
                    pass
            if total_claims > 0:
                s_evidence = (supported_claims / total_claims) * 10.0
            elif factual_scores:
                s_evidence = sum(factual_scores) / len(factual_scores)
        except Exception as snap_err:
            log.debug("Snapshot query skipped during fitness calc: %s", snap_err)

        # 3. User Feedback Component (from ResponseFeedback)
        s_feedback = 7.5  # Neutral default
        penalized_reasons: List[str] = []
        try:
            feedbacks = session.exec(select(ResponseFeedback).limit(80)).all()
            persona_fb = []
            for fb in feedbacks:
                meta = json.loads(fb.execution_metadata_json or "{}")
                if meta.get("winning_persona") == persona_id or persona_id in meta.get("top_personas", []):
                    persona_fb.append(fb)

            if persona_fb:
                likes = sum(1 for f in persona_fb if f.feedback_type == "like")
                dislikes = sum(1 for f in persona_fb if f.feedback_type == "dislike")
                base_ratio = likes / len(persona_fb)
                s_feedback = base_ratio * 10.0

                # Penalty deductions for specific reasons
                for d in persona_fb:
                    if d.feedback_type == "dislike" and d.reason:
                        penalized_reasons.append(d.reason)
                        if d.reason in ("factually_incorrect", "missing_sources"):
                            s_feedback = max(0.0, s_feedback - 0.5)
                        elif d.reason == "outdated_information":
                            s_feedback = max(0.0, s_feedback - 0.4)
                        elif d.reason == "poor_explanation":
                            s_feedback = max(0.0, s_feedback - 0.3)
        except Exception as fb_err:
            log.debug("Feedback query skipped during fitness calc: %s", fb_err)

        # 4. Multi-Factor Formula
        raw_fitness = (
            0.30 * s_feedback +
            0.25 * s_evidence +
            0.20 * s_relevance +
            0.15 * s_completeness +
            0.10 * s_consistency
        )
        raw_fitness = max(0.0, min(10.0, raw_fitness))

        # 5. Uncertainty Discounting (N < 8)
        is_statistically_valid = sample_count >= MIN_SAMPLE_THRESHOLD
        discount_factor = 1.0
        uncertainty_discount_applied = False
        composite_fitness = raw_fitness

        if not is_statistically_valid:
            discount_factor = math.sqrt(sample_count / MIN_SAMPLE_THRESHOLD) if sample_count > 0 else 0.2
            # Shrink towards neutral prior (5.0)
            composite_fitness = PRIOR_FITNESS + (raw_fitness - PRIOR_FITNESS) * discount_factor
            uncertainty_discount_applied = True

        composite_fitness = round(max(0.0, min(10.0, composite_fitness)), 3)
        raw_fitness = round(raw_fitness, 3)

        return FitnessReport(
            persona_id=persona_id,
            composite_fitness=composite_fitness,
            raw_fitness=raw_fitness,
            sample_count=sample_count,
            is_statistically_valid=is_statistically_valid,
            uncertainty_discount_applied=uncertainty_discount_applied,
            discount_factor=round(discount_factor, 3),
            dimensions={
                "feedback": round(s_feedback, 2),
                "evidence": round(s_evidence, 2),
                "relevance": round(s_relevance, 2),
                "completeness": round(s_completeness, 2),
                "consistency": round(s_consistency, 2),
            },
            penalized_reasons=list(set(penalized_reasons)),
            times_in_top_k=times_in_top_k,
            top_k_rate=round(top_k_rate, 3),
        )
    finally:
        if close_session and session:
            session.close()


def calculate_all_fitness(session: Optional[Session] = None) -> Dict[str, FitnessReport]:
    """Calculate fitness reports for all active personas."""
    from persona_evolution.config_schema import get_active_personas
    active = get_active_personas()
    reports = {}
    for p in active:
        reports[p.persona_id] = calculate_persona_fitness(p.persona_id, session=session)
    return reports
