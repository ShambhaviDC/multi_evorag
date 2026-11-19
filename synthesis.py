"""Top-K persona selection and final answer synthesis for EvoRAG."""

import logging
from typing import Optional

from config import CONSENSUS_TOP_K, PERSONA_TIMEOUT_SECONDS
from utils import call_ollama_async, race_ollama_with_gemini

log = logging.getLogger(__name__)


def select_top_personas(
    final_scores: dict[str, float],
    responses: dict[str, Optional[str]],
    k: int = CONSENSUS_TOP_K,
    personalization_scores: Optional[dict[str, float]] = None,
) -> list[str]:
    """Select the highest-scoring personas that produced non-empty responses.

    When personalization_scores is provided, calculates a decoupled combined score:
      Combined Score = 0.80 * Peer Score + 0.20 * (Personalization Score * 10.0)
    This ensures objective factual correctness dominates ranking while respecting user preferences.
    """
    eligible = [
        persona_id
        for persona_id, response in responses.items()
        if response and response.strip()
    ]

    def _rank_key(pid: str) -> float:
        peer_score = final_scores.get(pid, 0.0)
        if personalization_scores and pid in personalization_scores:
            pers_val = float(personalization_scores[pid]) * 10.0
            return 0.80 * peer_score + 0.20 * pers_val
        return peer_score

    ranked = sorted(eligible, key=_rank_key, reverse=True)

    # Fix 6: detect degenerate case where all scoring calls failed
    if eligible and all(final_scores.get(pid, 0.0) == 0.0 for pid in eligible):
        log.warning(
            "ALL PEER SCORES ARE ZERO — scoring stage produced no valid votes. "
            "Winner selection is falling back to config insertion order, NOT genuine ranking. "
            "Check api_debug.log for RAW SCORING OUTPUT to diagnose the scoring failure."
        )

    return ranked[:k]


def build_synthesis_prompt(
    query: str,
    top_personas: list[str],
    responses: dict[str, Optional[str]],
    final_scores: dict[str, float],
) -> str:
    """Build the final neutral synthesis prompt from selected responses."""
    labels = ("A", "B", "C", "D", "E")
    blocks = []
    for label, persona_id in zip(labels, top_personas):
        blocks.append(
            f"Response {label} (persona: {persona_id}, score: {final_scores.get(persona_id, 0.0):.1f}):\n"
            f"{responses.get(persona_id) or ''}"
        )
    selected = "\n\n".join(blocks)
    return f"""You are a neutral analytical synthesizer.

Strong responses to the question "{query}" have been selected through peer evaluation.
Merge them into one coherent, well-structured answer.

Rules:
- Focus directly on explaining the news and answering the user query clearly and thoroughly
- Explain what happened, who is involved, key details, and real-world significance
- Citing verified news publishers with inline markers [1], [2] is encouraged
- STRICTLY FORBIDDEN: Do not mention internal classifications or scores (e.g. 'Emerging', 'confidence score', 'assessed confidence level')
- STRICTLY FORBIDDEN: Do not generate robotic academic sections like 'Logical Structure and Assumptions', 'Limitations', or 'Implications and Risks'
- Knowledge gaps: Mention knowledge gaps ONLY when the user explicitly asks to predict or forecast future unknown outcomes
- Do not add information that is not present in the selected responses
- Do not mention that this is a synthesis or that multiple perspectives were used
- Write as a single, natural, authoritative news explanation

{selected}

Synthesized answer:"""


async def synthesize_top_k(
    query: str,
    top_personas: list[str],
    responses: dict[str, Optional[str]],
    final_scores: dict[str, float],
) -> str:
    """Call Ollama once to synthesize selected persona responses."""
    if not top_personas:
        return "No persona produced enough information to synthesize a consensus answer."
    prompt = build_synthesis_prompt(query, top_personas, responses, final_scores)
    answer, source = await race_ollama_with_gemini(
        call_ollama_async(prompt, timeout_seconds=PERSONA_TIMEOUT_SECONDS),
        prompt,
        total_timeout_seconds=PERSONA_TIMEOUT_SECONDS,
        log_label="[synthesis] ",
        max_output_tokens=1000,
        temperature=0.5,
    )
    if answer:
        log.info("[synthesis] Done (source=%s)", source)
        return answer
    log.warning("Synthesis call failed; falling back to highest-ranked persona response.")
    return responses.get(top_personas[0]) or ""
