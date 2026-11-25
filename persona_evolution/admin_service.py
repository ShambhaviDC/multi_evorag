"""
persona_evolution/admin_service.py — Admin Candidate Staging, Review, Prompt Diffs & Safe Rollback

Provides governance controls:
- Staging area for challenger candidates awaiting approval
- Visual line-by-line prompt diff generator
- Promotion execution with automated archival of prior versions
- Instant 1-click rollback to any historical version (including Gen 0 baseline)
- Hot-swapping active personas in memory (config.PERSONAS and personas.PERSONAS)
"""

from __future__ import annotations

import difflib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from persona_evolution.config_schema import (
    PersonaConfiguration,
    get_baseline_personas,
    load_persona_registry,
    save_persona_registry,
)
from persona_evolution.evaluator import BenchmarkResult

log = logging.getLogger(__name__)

CANDIDATES_FILE = Path(__file__).parent / "persona_candidates.json"
HISTORY_FILE = Path(__file__).parent / "persona_versions.json"


def _load_json_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return []


def _save_json_file(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def stage_candidate(
    candidate: PersonaConfiguration,
    benchmark_result: BenchmarkResult,
) -> str:
    """Stage a candidate in the review queue."""
    candidates = _load_json_file(CANDIDATES_FILE)
    candidate_id = f"cand-{uuid.uuid4().hex[:8]}"

    entry = {
        "candidate_id": candidate_id,
        "persona_config": candidate.model_dump(),
        "benchmark_result": benchmark_result.model_dump(),
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "status": "awaiting_review" if benchmark_result.qualified_for_promotion else "benchmark_failed",
    }
    candidates.append(entry)
    _save_json_file(CANDIDATES_FILE, candidates)
    log.info("Staged candidate %s for persona %s", candidate_id, candidate.persona_id)
    return candidate_id


def list_candidates() -> List[Dict[str, Any]]:
    """List all staged challenger candidates."""
    return _load_json_file(CANDIDATES_FILE)


def get_candidate_by_id(candidate_id: str) -> Optional[Dict[str, Any]]:
    """Find candidate record by ID."""
    for c in list_candidates():
        if c.get("candidate_id") == candidate_id:
            return c
    return None


def generate_prompt_diff(old_prompt: str, new_prompt: str) -> List[Dict[str, str]]:
    """
    Generate structured line-by-line diff comparing old vs new system prompt.
    Returns: list of dicts with type ('added', 'removed', 'unchanged') and line text.
    """
    old_lines = old_prompt.splitlines()
    new_lines = new_prompt.splitlines()
    diff = list(difflib.ndiff(old_lines, new_lines))

    result = []
    for line in diff:
        if line.startswith("+ "):
            result.append({"type": "added", "text": line[2:]})
        elif line.startswith("- "):
            result.append({"type": "removed", "text": line[2:]})
        elif line.startswith("  "):
            result.append({"type": "unchanged", "text": line[2:]})
    return result


def _hot_swap_runtime_personas(active_configs: List[PersonaConfiguration]) -> None:
    """Hot-swap runtime config.PERSONAS in memory to immediately apply changes without server restart."""
    runtime_list = [
        {"id": p.persona_id, "name": p.name, "system_prompt": p.system_prompt}
        for p in active_configs
    ]
    config.PERSONAS = runtime_list
    try:
        import personas
        personas.PERSONAS = runtime_list
    except Exception:
        pass
    log.info("Hot-swapped runtime config.PERSONAS with %d personas.", len(runtime_list))


def approve_candidate(candidate_id: str, reviewer: str = "Admin") -> PersonaConfiguration:
    """
    Approve candidate promotion:
    1. Archives existing champion version in persona_versions.json.
    2. Promotes candidate to active champion in persona_registry.json.
    3. Hot-swaps runtime config.PERSONAS.
    4. Marks candidate as approved.
    """
    candidates = list_candidates()
    target_idx = None
    for idx, c in enumerate(candidates):
        if c.get("candidate_id") == candidate_id:
            target_idx = idx
            break

    if target_idx is None:
        raise ValueError(f"Candidate '{candidate_id}' not found.")

    cand_data = candidates[target_idx]
    p_config = PersonaConfiguration(**cand_data["persona_config"])

    registry = load_persona_registry()
    parent_id = p_config.parent_ids[0] if p_config.parent_ids else p_config.persona_id.split("-v")[0]

    # Find incumbent champion to archive
    incumbent = registry.get(parent_id) or registry.get(p_config.persona_id)
    history = _load_json_file(HISTORY_FILE)

    if incumbent:
        # Archive incumbent
        history.append({
            "event": "archived_champion",
            "persona_id": incumbent.persona_id,
            "version": incumbent.version,
            "generation": incumbent.generation,
            "system_prompt": incumbent.system_prompt,
            "replaced_by": p_config.persona_id,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "reviewer": reviewer,
        })
        incumbent.is_active = False
        incumbent.status = "archived"

    # Promote challenger
    p_config.is_active = True
    p_config.status = "champion"
    registry[parent_id] = p_config  # Replace slot with evolved champion

    # Record promotion event in history
    history.append({
        "event": "promoted_champion",
        "persona_id": p_config.persona_id,
        "slot_id": parent_id,
        "version": p_config.version,
        "generation": p_config.generation,
        "system_prompt": p_config.system_prompt,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "benchmark": cand_data.get("benchmark_result"),
    })

    # Save updates
    save_persona_registry(registry)
    _save_json_file(HISTORY_FILE, history)

    cand_data["status"] = "approved"
    cand_data["approved_at"] = datetime.now(timezone.utc).isoformat()
    cand_data["approved_by"] = reviewer
    _save_json_file(CANDIDATES_FILE, candidates)

    _hot_swap_runtime_personas(list(registry.values()))
    log.info("Candidate %s promoted to Champion for slot %s!", candidate_id, parent_id)
    return p_config


def reject_candidate(candidate_id: str, reason: str = "") -> None:
    """Mark a candidate as rejected in the staging queue."""
    candidates = list_candidates()
    for c in candidates:
        if c.get("candidate_id") == candidate_id:
            c["status"] = "rejected"
            c["rejected_at"] = datetime.now(timezone.utc).isoformat()
            c["rejection_reason"] = reason or "Rejected during admin review"
            _save_json_file(CANDIDATES_FILE, candidates)
            log.info("Candidate %s rejected: %s", candidate_id, reason)
            return
    raise ValueError(f"Candidate '{candidate_id}' not found.")


def rollback_persona(persona_id: str, target_version: int = 1) -> PersonaConfiguration:
    """
    Rollback a persona to an earlier version (or baseline Gen 0 v1):
    1. Finds target historical version from persona_versions.json or baseline configs.
    2. Overwrites registry slot with target configuration.
    3. Hot-swaps runtime config.PERSONAS.
    """
    registry = load_persona_registry()
    history = _load_json_file(HISTORY_FILE)

    target_config: Optional[PersonaConfiguration] = None

    if target_version == 1:
        # Restore canonical Gen 0 baseline
        baselines = {p.persona_id: p for p in get_baseline_personas()}
        if persona_id in baselines:
            target_config = baselines[persona_id]

    if not target_config:
        for entry in reversed(history):
            if (entry.get("persona_id") == persona_id or entry.get("slot_id") == persona_id) and entry.get("version") == target_version:
                target_config = PersonaConfiguration(
                    persona_id=persona_id,
                    name=persona_id.replace("-", " ").title(),
                    version=target_version,
                    generation=entry.get("generation", 0),
                    system_prompt=entry["system_prompt"],
                    is_active=True,
                    status="champion",
                )
                break

    if not target_config:
        raise ValueError(f"Version {target_version} not found in history for persona '{persona_id}'.")

    target_config.is_active = True
    target_config.status = "champion"
    registry[persona_id] = target_config

    history.append({
        "event": "rollback_executed",
        "persona_id": persona_id,
        "restored_version": target_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    save_persona_registry(registry)
    _save_json_file(HISTORY_FILE, history)
    _hot_swap_runtime_personas(list(registry.values()))

    log.info("Rolled back persona %s to version %d", persona_id, target_version)
    return target_config


def get_evolution_history() -> List[Dict[str, Any]]:
    """Return complete chronological audit log of all evolutionary persona events."""
    return _load_json_file(HISTORY_FILE)
