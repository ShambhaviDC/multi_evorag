"""
evolution_api.py — REST API for Evolutionary Persona Adaptation & Governance

Endpoints:
  - GET  /evolution/personas             → Active personas with versions, generations & multi-factor fitness
  - GET  /evolution/candidates           → Staged challenger candidates with prompt diffs
  - GET  /evolution/history              → Complete audit history of mutations, crossovers & rollbacks
  - POST /evolution/trigger              → Trigger an evolutionary evaluation cycle (GA operators + benchmark)
  - POST /evolution/candidate/{id}/approve → Admin approval & hot-swapping into champion slot
  - POST /evolution/candidate/{id}/reject  → Reject staged candidate
  - POST /evolution/rollback/{persona_id} → Instant rollback to earlier version
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlmodel import Session

from auth import get_current_user, get_optional_current_user
from database import get_session
from models import User
from persona_evolution.config_schema import (
    PersonaConfiguration,
    get_active_personas,
    load_persona_registry,
)
from persona_evolution.fitness import calculate_all_fitness, calculate_persona_fitness
from persona_evolution.genetic_ops import (
    crossover_personas,
    identify_underperforming_personas,
    mutate_persona,
    select_elite_parents,
)
from persona_evolution.evaluator import evaluate_champion_vs_challenger
from persona_evolution.admin_service import (
    approve_candidate,
    generate_prompt_diff,
    get_candidate_by_id,
    get_evolution_history,
    list_candidates,
    reject_candidate,
    rollback_persona,
    stage_candidate,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/evolution", tags=["Evolutionary Personas"])


class RollbackRequest(BaseModel):
    target_version: int = Field(default=1, ge=1, description="Target version number to restore")


class TriggerEvolutionRequest(BaseModel):
    force_persona_id: Optional[str] = Field(default=None, description="Force adaptation on a specific persona")
    strategy: str = Field(default="mutation", description="'mutation' or 'crossover'")


@router.get("/personas", summary="List active personas with multi-factor fitness")
def get_personas_with_fitness(session: Session = Depends(get_session)):
    """Retrieve all 8 active champion personas with their version, generation, and multi-factor fitness."""
    active = get_active_personas()
    fitness_reports = calculate_all_fitness(session=session)

    results = []
    for p in active:
        rep = fitness_reports.get(p.persona_id) or calculate_persona_fitness(p.persona_id, session=session)
        p_dict = p.model_dump()
        p_dict["fitness"] = rep.model_dump()
        results.append(p_dict)
    return results


@router.get("/candidates", summary="List staged challenger candidates awaiting review")
def get_staged_candidates():
    """Retrieve all staged challenger candidates with unified prompt diffs against parent prompts."""
    candidates = list_candidates()
    registry = load_persona_registry()

    enriched = []
    for c in candidates:
        entry = dict(c)
        cfg = c.get("persona_config", {})
        parents = cfg.get("parent_ids", [])
        parent_prompt = ""
        if parents and parents[0] in registry:
            parent_prompt = registry[parents[0]].system_prompt
        elif parents:
            # Try finding root id without version suffix
            root_id = parents[0].split("-v")[0]
            if root_id in registry:
                parent_prompt = registry[root_id].system_prompt

        new_prompt = cfg.get("system_prompt", "")
        if parent_prompt and new_prompt:
            entry["prompt_diff"] = generate_prompt_diff(parent_prompt, new_prompt)
        else:
            entry["prompt_diff"] = []

        enriched.append(entry)
    return enriched


@router.get("/history", summary="Chronological audit history of persona evolution")
def get_history():
    """Return chronological log of all promotions, crossovers, mutations, and rollbacks."""
    return get_evolution_history()


@router.post("/trigger", summary="Trigger genetic algorithm evaluation & candidate staging")
def trigger_evolution(
    req: TriggerEvolutionRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
):
    """
    Run GA evaluation:
    1. Computes multi-factor fitness for all personas.
    2. Identifies underperforming personas (or uses force_persona_id).
    3. Applies Mutation or Crossover operator to generate a Challenger.
    4. Runs Champion vs Challenger benchmark evaluation.
    5. Stages Challenger in review queue.
    """
    fitness_reports = calculate_all_fitness(session=session)
    registry = load_persona_registry()

    target_pid = req.force_persona_id
    target_report = None

    if target_pid:
        target_report = fitness_reports.get(target_pid) or calculate_persona_fitness(target_pid, session=session)
    else:
        underperformers = identify_underperforming_personas(fitness_reports)
        if underperformers:
            target_pid, target_report = underperformers[0]
        else:
            # If all are healthy, pick the lowest scoring persona for continuous improvement
            sorted_by_score = sorted(fitness_reports.items(), key=lambda x: x[1].composite_fitness)
            target_pid, target_report = sorted_by_score[0]

    champion = registry.get(target_pid)
    if not champion:
        raise HTTPException(status_code=404, detail=f"Target persona '{target_pid}' not found.")

    challenger: PersonaConfiguration
    if req.strategy == "crossover":
        elites = select_elite_parents(fitness_reports, top_n=2)
        if len(elites) >= 2 and elites[0][0] in registry and elites[1][0] in registry:
            challenger = crossover_personas(registry[elites[0][0]], registry[elites[1][0]])
        else:
            challenger = mutate_persona(champion, target_report)
    else:
        challenger = mutate_persona(champion, target_report)

    # Benchmark Champion vs Challenger
    benchmark = evaluate_champion_vs_challenger(
        champion=champion,
        challenger=challenger,
        champion_baseline_score=target_report.composite_fitness,
    )

    # Stage candidate in review queue
    candidate_id = stage_candidate(challenger, benchmark)

    return {
        "status": "candidate_staged",
        "candidate_id": candidate_id,
        "champion_id": champion.persona_id,
        "challenger_id": challenger.persona_id,
        "benchmark_result": benchmark.model_dump(),
        "challenger_config": challenger.model_dump(),
    }


@router.post("/candidate/{candidate_id}/approve", summary="Approve and promote challenger candidate")
def approve_candidate_endpoint(
    candidate_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Admin approval to promote staged Challenger into active Champion slot."""
    reviewer_name = current_user.username if current_user else "Admin"
    try:
        promoted = approve_candidate(candidate_id, reviewer=reviewer_name)
        return {
            "status": "promoted",
            "message": f"Candidate successfully promoted to Champion for {promoted.persona_id}.",
            "promoted_persona": promoted.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {e}")


@router.post("/candidate/{candidate_id}/reject", summary="Reject staged challenger candidate")
def reject_candidate_endpoint(
    candidate_id: str,
    reason: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Mark a staged candidate as rejected."""
    try:
        reject_candidate(candidate_id, reason=reason or "Rejected by Admin")
        return {"status": "rejected", "candidate_id": candidate_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/rollback/{persona_id}", summary="Rollback persona to target version")
def rollback_persona_endpoint(
    persona_id: str,
    req: RollbackRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Instant 1-click rollback of a persona to earlier version or Gen 0 baseline."""
    try:
        restored = rollback_persona(persona_id, target_version=req.target_version)
        return {
            "status": "rolled_back",
            "message": f"Successfully restored {persona_id} to version {restored.version} (Gen {restored.generation}).",
            "restored_persona": restored.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rollback failed: {e}")
