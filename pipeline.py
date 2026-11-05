"""Full EvoRAG consensus pipeline orchestration."""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from config import RETRIEVAL_TOP_K
from personas import run_all_personas
from retriever import retrieve, evaluate_retrieval
from score_store import append_score_entry
from synthesis import select_top_personas, synthesize_top_k
from utils import call_gemini_async, word_count
from voting import build_score_matrix, compute_final_scores, summarize_received_scores

log = logging.getLogger(__name__)


async def evorag_query(query: str, top_k: int = RETRIEVAL_TOP_K) -> dict[str, Any]:
    """Run retrieval, relevance/sufficiency evaluation, persona generation, peer voting, and synthesis.

    Workflow:
      1. Search the latest local FAISS index.
      2. Evaluate relevance, freshness, and sufficiency of retrieved results.
      3. If sufficient: execute the 8-persona consensus pipeline and final synthesis.
      4. If insufficient or outdated: invoke the Gemini API fallback.
    """
    query_id = str(uuid4())

    # Step 1: Search the latest local FAISS index (auto-synchronized with disk)
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Evaluate retrieval relevance, freshness, and sufficiency
    is_sufficient, reason = evaluate_retrieval(query, chunks)

    # Step 3: Handle insufficient local information via Gemini fallback
    if not is_sufficient:
        log.warning(
            "Local knowledge retrieval insufficient for '%s' (%s). Invoking Gemini fallback.",
            query[:60],
            reason,
        )
        fallback_prompt = (
            "You are EvoRAG's fallback knowledge intelligence model.\n"
            f"The local news knowledge base did not contain sufficient relevant or fresh documents for this query: {reason}.\n\n"
            f"Question: {query}\n\n"
            "Instructions:\n"
            "- Provide a thorough, objective, and well-structured answer.\n"
            "- If the question specifically inquires about breaking news or real-time events that occurred after your training cut-off, state clearly that real-time live browsing is unavailable without real-time search grounding."
        )
        gemini_answer = await call_gemini_async(fallback_prompt, max_output_tokens=750, temperature=0.7)
        if not gemini_answer:
            gemini_answer = (
                "The local knowledge base does not contain sufficient information on this topic, "
                "and the external Gemini fallback service was temporarily unavailable."
            )

        sources = [
            {
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "source": c.get("source", ""),
                "score": round(c.get("score", 0.0), 4),
            }
            for c in chunks
        ]

        fallback_responses = {
            "gemini-fallback": gemini_answer,
        }
        top_personas = ["gemini-fallback"]
        final_scores = {"gemini-fallback": 1.0}
        score_matrix = {}

        score_entry = {
            "query_id": query_id,
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "persona_scores": {"gemini-fallback": {"average": 1.0, "made_top_k": True}},
            "winner": "gemini-fallback",
            "synthesis_used": top_personas,
            "user_signal": f"gemini_fallback: {reason}",
        }
        append_score_entry(score_entry)
        log.info("Gemini fallback completed for query_id=%s. reason=%s", query_id, reason)

        return {
            "final_answer": gemini_answer,
            "top_personas": top_personas,
            "scores": final_scores,
            "all_responses": fallback_responses,
            "score_matrix": score_matrix,
            "query_id": query_id,
            "sources": sources,
            "retrieval_status": {
                "sufficient": False,
                "reason": reason,
                "fallback_used": True,
            },
        }

    # Step 4: Context is sufficient — run 8-persona generation on retrieved chunks
    persona_result = await run_all_personas(
        query=query,
        top_k=top_k,
        persona_ids=None,
        pre_retrieved_chunks=chunks,
    )
    responses: dict[str, Optional[str]] = {
        persona["id"]: persona.get("response") or None
        for persona in persona_result["personas"]
    }

    score_matrix = await build_score_matrix(query, responses)
    final_scores = compute_final_scores(score_matrix)
    top_personas = select_top_personas(final_scores, responses)
    final_answer = await synthesize_top_k(query, top_personas, responses, final_scores)

    received = summarize_received_scores(score_matrix)
    persona_scores = {}
    for persona_id, summary in received.items():
        persona_scores[persona_id] = {
            **summary,
            "made_top_k": persona_id in top_personas,
            "response_length": word_count(responses.get(persona_id)),
        }

    winner = top_personas[0] if top_personas else None
    score_entry = {
        "query_id": query_id,
        "query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "persona_scores": persona_scores,
        "winner": winner,
        "synthesis_used": top_personas,
        "user_signal": None,
    }
    append_score_entry(score_entry)
    log.info("Consensus complete. winner=%s top_k=%s", winner, top_personas)

    return {
        "final_answer": final_answer,
        "top_personas": top_personas,
        "scores": final_scores,
        "all_responses": responses,
        "score_matrix": score_matrix,
        "query_id": query_id,
        "sources": persona_result["sources"],
        "retrieval_status": {
            "sufficient": True,
            "reason": reason,
            "fallback_used": False,
        },
    }
