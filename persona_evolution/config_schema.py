"""
persona_evolution/config_schema.py — Versioned Persona Configuration Schema

Provides structured schema and atomic persistence for EvoRAG personas:
- Stable persona identifier
- Integer versioning and generation tracking (Gen 0 = baseline)
- Role, explicit responsibilities, domain expertise, and system prompt
- Parentage lineage (parent_ids) and mutation history tags
- Active status and runtime hot-swapping
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from config import PERSONAS

log = logging.getLogger(__name__)

REGISTRY_FILE = Path(__file__).parent / "persona_registry.json"


class PersonaConfiguration(BaseModel):
    """Immutable or versioned representation of an analytical AI persona."""
    persona_id: str = Field(..., description="Unique slug identifier (e.g. 'analytical-critical')")
    name: str = Field(..., description="Human-readable title (e.g. 'Analytical-Critical')")
    version: int = Field(default=1, description="Monotonically increasing version number")
    generation: int = Field(default=0, description="Genetic algorithm generation (0 = baseline)")
    role: str = Field(default="", description="High-level analytical role")
    responsibilities: List[str] = Field(default_factory=list, description="Explicit analytical duties")
    domain_expertise: List[str] = Field(default_factory=list, description="Specialized domain areas")
    system_prompt: str = Field(..., description="Full system instruction prompt")
    parent_ids: List[str] = Field(default_factory=list, description="Lineage tracking for mutations/crossovers")
    mutation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Audit trail of applied mutations")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = Field(default=True, description="Whether this persona is in the active 8-roster")
    status: str = Field(default="champion", description="'champion', 'challenger', 'retired', 'archived'")
    fitness_score: Optional[float] = Field(default=None, description="Latest recorded composite fitness score [0-10]")


# Default metadata mapping for the 8 baseline personas (Generation 0)
BASELINE_METADATA: Dict[str, Dict[str, Any]] = {
    "analytical-critical": {
        "role": "Policy and structural mechanism deconstruction",
        "responsibilities": [
            "Deconstruct reported decisions, institutional policies, and functional mechanics",
            "Explain practical implications and real-world execution details",
            "Avoid robotic or academic meta-critique"
        ],
        "domain_expertise": ["Policy Analysis", "Governance", "Institutional Systems", "Legal Frameworks"]
    },
    "empirical-evidential": {
        "role": "Empirical and factual evidence validation",
        "responsibilities": [
            "Extract verified dates, quantifiable metrics, statistics, and concrete events",
            "Anchor statements strictly in retrieved publisher passages",
            "Eliminate ungrounded speculation"
        ],
        "domain_expertise": ["Data Verification", "Statistics", "Investigative Research", "Fact-Checking"]
    },
    "adversarial-skeptical": {
        "role": "Balanced counter-perspective and stakeholder friction analysis",
        "responsibilities": [
            "Surface substantive dissenting viewpoints and opposing stakeholder arguments",
            "Identify non-obvious conflicts of interest and counter-claims",
            "Avoid meta-critiquing text format; focus on genuine real-world tension"
        ],
        "domain_expertise": ["Counter-Analysis", "Stakeholder Dynamics", "Risk Scrutiny", "Debate Synthesis"]
    },
    "synthesis-integrative": {
        "role": "Cross-source synthesis and systemic convergence",
        "responsibilities": [
            "Connect disparate themes across multiple news articles",
            "Identify common patterns, structural tensions, and macro-trends",
            "Produce cohesive summaries that integrate distinct viewpoints"
        ],
        "domain_expertise": ["Macro Synthesis", "Complex Systems", "Trend Interconnection", "Integrative Summary"]
    },
    "domain-expert": {
        "role": "Specialized technical depth and terminology precision",
        "responsibilities": [
            "Provide deep domain nuance using precise scientific or technical nomenclature",
            "Clarify and correct oversimplified public explanations",
            "Highlight technical feasibility and architectural dependencies"
        ],
        "domain_expertise": ["Specialized Technology", "Scientific Method", "Domain Terminology", "Technical Evaluation"]
    },
    "temporal-contextual": {
        "role": "Chronological evolution and trajectory tracking",
        "responsibilities": [
            "Frame developments along chronological milestones and event sequences",
            "Distinguish historical context from newly breaking developments",
            "Identify trajectory, velocity of change, and evolving milestones"
        ],
        "domain_expertise": ["Chronological Analysis", "Event Evolution", "Trend Trajectories", "Milestone Tracking"]
    },
    "risk-evaluative": {
        "role": "Failure-mode and hazard assessment",
        "responsibilities": [
            "Identify vulnerabilities, threat vectors, and systemic failure modes",
            "Estimate potential impact, likelihood, and affected stakeholders",
            "Highlight what could go wrong and identify mitigation blindspots"
        ],
        "domain_expertise": ["Threat Modeling", "Vulnerability Assessment", "Impact Estimation", "Mitigation Strategy"]
    },
    "creative-lateral": {
        "role": "Unconventional framing and non-obvious connection discovery",
        "responsibilities": [
            "Explore second-order consequences and non-linear implications",
            "Formulate novel framings while staying grounded in source facts",
            "Draw unexpected parallels across adjacent industry domains"
        ],
        "domain_expertise": ["Lateral Thinking", "Second-Order Effects", "Cross-Domain Analogy", "Novel Perspective"]
    }
}


def get_baseline_personas() -> List[PersonaConfiguration]:
    """Generate the 8 canonical Generation 0 baseline persona configurations."""
    baseline_configs: List[PersonaConfiguration] = []
    for p in PERSONAS:
        pid = p["id"]
        meta = BASELINE_METADATA.get(pid, {
            "role": "Analytical News Intelligence",
            "responsibilities": ["Analyze news context thoroughly."],
            "domain_expertise": ["General News Analysis"]
        })
        config = PersonaConfiguration(
            persona_id=pid,
            name=p["name"],
            version=1,
            generation=0,
            role=meta["role"],
            responsibilities=meta["responsibilities"],
            domain_expertise=meta["domain_expertise"],
            system_prompt=p["system_prompt"],
            parent_ids=[],
            mutation_history=[],
            is_active=True,
            status="champion",
        )
        baseline_configs.append(config)
    return baseline_configs


def load_persona_registry() -> Dict[str, PersonaConfiguration]:
    """Load all personas from persona_registry.json. Initializes with baseline Gen 0 if missing."""
    if not REGISTRY_FILE.exists():
        log.info("Initializing baseline Generation 0 personas in registry: %s", REGISTRY_FILE)
        baselines = get_baseline_personas()
        registry = {p.persona_id: p for p in baselines}
        save_persona_registry(registry)
        return registry

    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        registry = {}
        for pid, pdata in data.items():
            registry[pid] = PersonaConfiguration(**pdata)
        return registry
    except Exception as exc:
        log.error("Failed to read persona_registry.json: %s. Falling back to Gen 0 baseline.", exc)
        baselines = get_baseline_personas()
        return {p.persona_id: p for p in baselines}


def save_persona_registry(registry: Dict[str, PersonaConfiguration]) -> None:
    """Safely persist persona registry to disk atomically."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = REGISTRY_FILE.with_suffix(".tmp")
    data = {pid: p.model_dump() for pid, p in registry.items()}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(REGISTRY_FILE)
    log.info("Saved %d personas to registry: %s", len(registry), REGISTRY_FILE)


def get_active_personas() -> List[PersonaConfiguration]:
    """Retrieve the currently active 8 champion personas."""
    registry = load_persona_registry()
    active = [p for p in registry.values() if p.is_active and p.status == "champion"]
    # Fallback to any 8 active if champion tags missing
    if len(active) < 8:
        active = [p for p in registry.values() if p.is_active][:8]
    return active


def get_persona_by_id(persona_id: str) -> Optional[PersonaConfiguration]:
    """Find a specific persona configuration by ID."""
    registry = load_persona_registry()
    return registry.get(persona_id)
