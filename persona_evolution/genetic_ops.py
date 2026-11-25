"""
persona_evolution/genetic_ops.py — Genetic Algorithm Operators for Persona Evolution

Implements GA operators for continuous persona adaptation:
1. Selection: Fitness-proportionate & tournament selection of underperformers and elite parents
2. Mutation: Targeted prompt mutation guided by empirical failure modes (negative feedback, low scores)
3. Crossover: Trait recombination of two elite parents producing a specialized challenger persona
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from persona_evolution.config_schema import PersonaConfiguration
from persona_evolution.fitness import FitnessReport

log = logging.getLogger(__name__)

UNDERPERFORMANCE_THRESHOLD = 5.4  # Composite fitness below this triggers adaptation


def identify_underperforming_personas(
    fitness_reports: Dict[str, FitnessReport],
    threshold: float = UNDERPERFORMANCE_THRESHOLD,
) -> List[Tuple[str, FitnessReport]]:
    """Identify personas that qualify for adaptation or replacement."""
    flagged = []
    for pid, rep in fitness_reports.items():
        # Must have minimal data to be eligible for replacement
        if rep.sample_count >= 5 and rep.composite_fitness < threshold:
            flagged.append((pid, rep))
    # Sort worst first
    flagged.sort(key=lambda x: x[1].composite_fitness)
    return flagged


def select_elite_parents(
    fitness_reports: Dict[str, FitnessReport],
    top_n: int = 2,
) -> List[Tuple[str, FitnessReport]]:
    """Select the highest-performing personas to serve as genetic crossover parents."""
    eligible = [
        (pid, rep) for pid, rep in fitness_reports.items()
        if rep.sample_count >= 5
    ]
    eligible.sort(key=lambda x: x[1].composite_fitness, reverse=True)
    return eligible[:top_n]


def mutate_persona(
    parent: PersonaConfiguration,
    fitness_report: FitnessReport,
) -> PersonaConfiguration:
    """
    Generate a mutated Challenger persona from an underperforming Parent.
    Mutation direction is guided by observed empirical penalties and lowest dimension scores.
    """
    reasons = fitness_report.penalized_reasons
    dims = fitness_report.dimensions
    mutation_tags = []
    prompt_additions = []
    updated_responsibilities = list(parent.responsibilities)

    # 1. Feedback-guided mutation
    if "outdated_information" in reasons:
        mutation_tags.append("temporal_verification_emphasis")
        prompt_additions.append(
            "MANDATORY TEMPORAL FILTER: Explicitly verify publication dates and sequence of events. "
            "Never cite outdated developments as current breaking news."
        )
        updated_responsibilities.append("Verify temporal currency and chronology of reported claims.")

    if "missing_sources" in reasons or "factually_incorrect" in reasons:
        mutation_tags.append("rigorous_evidence_grounding")
        prompt_additions.append(
            "EVIDENCE GROUNDING MANDATE: Every major factual claim must be explicitly anchored in the provided context. "
            "Never make unsupported assertions or speculate beyond verified news reports."
        )
        updated_responsibilities.append("Strict factual evidence anchoring with zero unverified extrapolation.")

    if "poor_explanation" in reasons or dims.get("relevance", 7.0) < 6.0:
        mutation_tags.append("causal_depth_amplification")
        prompt_additions.append(
            "STRUCTURAL CLARITY: Provide deep, step-by-step causal explanations. Explain not only what happened, "
            "but the underlying mechanism, operational drivers, and institutional stakes."
        )
        updated_responsibilities.append("Explain underlying mechanisms and real-world execution dependencies.")

    if dims.get("completeness", 7.0) < 6.0:
        mutation_tags.append("completeness_expansion")
        prompt_additions.append(
            "COMPREHENSIVE MULTI-PERSPECTIVE SCOPE: Ensure coverage includes key secondary stakeholders, "
            "policy trade-offs, and downstream economic or social ramifications."
        )
        updated_responsibilities.append("Cover multi-stakeholder trade-offs and secondary impacts.")

    # Fallback mutation if no specific tag triggered
    if not prompt_additions:
        mutation_tags.append("precision_and_substance_refinement")
        prompt_additions.append(
            "ANALYTICAL RIGOR: Focus on concrete, high-information analysis. Eliminate filler and deliver "
            "authoritative, insight-dense explanations directly answering the user query."
        )

    # Construct evolved system prompt
    base_prompt = parent.system_prompt.strip()
    evolved_prompt = base_prompt + "\n\n" + "\n".join(prompt_additions)

    candidate_id = f"{parent.persona_id}-v{parent.version + 1}"
    candidate_name = f"{parent.name} (Evolved v{parent.version + 1})"

    challenger = PersonaConfiguration(
        persona_id=candidate_id,
        name=candidate_name,
        version=parent.version + 1,
        generation=parent.generation + 1,
        role=f"{parent.role} [Targeted Adaptation]",
        responsibilities=updated_responsibilities,
        domain_expertise=parent.domain_expertise,
        system_prompt=evolved_prompt,
        parent_ids=[parent.persona_id],
        mutation_history=parent.mutation_history + [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parent_version": parent.version,
            "tags": mutation_tags,
            "prior_fitness": fitness_report.composite_fitness,
        }],
        created_at=datetime.now(timezone.utc).isoformat(),
        is_active=False,
        status="challenger",
        fitness_score=None,
    )
    log.info("Generated mutated Challenger: %s (mutations: %s)", candidate_id, mutation_tags)
    return challenger


def crossover_personas(
    parent_a: PersonaConfiguration,
    parent_b: PersonaConfiguration,
) -> PersonaConfiguration:
    """
    Perform genetic crossover between two high-performing parent personas.
    Synthesizes complementary strengths without naive text concatenation.
    """
    hybrid_id = f"hybrid-{parent_a.persona_id[:8]}-{parent_b.persona_id[:8]}"
    hybrid_name = f"{parent_a.name.split('-')[0]}-{parent_b.name.split('-')[0]} Hybrid"
    next_gen = max(parent_a.generation, parent_b.generation) + 1

    # Interleave complementary responsibilities
    combined_resp = []
    for r in parent_a.responsibilities[:2] + parent_b.responsibilities[:2]:
        if r not in combined_resp:
            combined_resp.append(r)

    # Merge domain expertise
    combined_expertise = list(dict.fromkeys(parent_a.domain_expertise + parent_b.domain_expertise))

    # Construct synthesized system prompt
    hybrid_prompt = (
        f"You are the {hybrid_name}. You combine the {parent_a.role.lower()} with the {parent_b.role.lower()}.\n"
        f"Your responsibilities:\n"
        + "\n".join(f"- {r}" for r in combined_resp)
        + "\n\nDeliver authoritative, multi-perspective synthesis grounded strictly in verified news context."
    )

    challenger = PersonaConfiguration(
        persona_id=hybrid_id,
        name=hybrid_name,
        version=1,
        generation=next_gen,
        role=f"Synthesized Hybrid ({parent_a.name} × {parent_b.name})",
        responsibilities=combined_resp,
        domain_expertise=combined_expertise,
        system_prompt=hybrid_prompt,
        parent_ids=[parent_a.persona_id, parent_b.persona_id],
        mutation_history=[{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "crossover",
            "parents": [parent_a.persona_id, parent_b.persona_id],
        }],
        created_at=datetime.now(timezone.utc).isoformat(),
        is_active=False,
        status="challenger",
        fitness_score=None,
    )
    log.info("Generated Crossover Challenger: %s (parents: %s, %s)", hybrid_id, parent_a.persona_id, parent_b.persona_id)
    return challenger
