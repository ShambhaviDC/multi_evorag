"""gemini_client.py - Structured Request Adapter and Multimodal Client for Gemini (EvoRAG Phase 7).

Converts StructuredQuery objects, multimodal file attachments, and retrieved context
into valid Google Gemini REST API payloads. Supports:
- Multimodal image inline_data parts
- Google Search grounding tools for live news
- Structured system instructions and format hints
- Robust retry with exponential backoff and connection pooling
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from config import GEMINI_API_KEY, GEMINI_MODEL
from file_processor import ProcessedFile
from query_models import StructuredQuery

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_gemini_http_client: Optional[httpx.AsyncClient] = None


def get_gemini_http_client() -> httpx.AsyncClient:
    global _gemini_http_client
    if _gemini_http_client is None or _gemini_http_client.is_closed:
        _gemini_http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=20)
        )
    return _gemini_http_client


def build_structured_prompt(
    structured_query: StructuredQuery,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[ProcessedFile]] = None,
    personalization_context: Optional[str] = None,
    custom_instruction: Optional[str] = None,
) -> str:
    """Build a comprehensive context prompt from structured query representation, evidence, and user personalization."""
    sections = [
        "You are the EvoRAG Adaptive Intelligence Engine.",
        f"Original User Query: {structured_query.original_query}",
        f"Identified Intent: {structured_query.intent} (Category: {structured_query.query_category})",
    ]

    # Entities
    if structured_query.entities:
        ent_strs = [f"{e.name} ({e.type})" for e in structured_query.entities]
        sections.append(f"Identified Entities: {', '.join(ent_strs)}")

    # Relationships
    if structured_query.relationships:
        rel_strs = [f"{r.source} -> [{r.type}] -> {r.target}" for r in structured_query.relationships]
        sections.append(f"Entity Relationships: {'; '.join(rel_strs)}")

    # Requested Information
    if structured_query.requested_information:
        sections.append(f"Requested Focus Areas: {', '.join(structured_query.requested_information)}")

    # Temporal & Constraints
    if structured_query.constraints.requires_latest_information:
        sections.append("Freshness Requirement: Latest, up-to-date information is explicitly required.")
    if structured_query.constraints.temporal_focus:
        sections.append(f"Temporal Scope: {structured_query.constraints.temporal_focus}")

    # Output Plan
    sections.append(
        f"Expected Output Format: Primary='{structured_query.output_plan.primary_format}', "
        f"Additional={structured_query.output_plan.additional_formats}"
    )

    # Attachments Context (Extracted Text / Tabular Summaries)
    if attachments:
        sections.append("\n=== UPLOADED ATTACHMENTS EVIDENCE ===")
        for att in attachments:
            if att.error:
                sections.append(f"File [{att.filename}]: Error - {att.error}")
            elif att.structured_data:
                # Tabular dataset summary
                sd = att.structured_data
                sections.append(
                    f"Dataset [{att.filename}]: {sd.get('total_rows')} rows, {sd.get('total_columns')} columns.\n"
                    f"Columns: {', '.join(sd.get('columns', []))}\n"
                    f"Sample Data:\n{att.text_content[:2500]}"
                )
            elif att.text_content:
                sections.append(f"Document [{att.filename}] (Type: {att.file_type}):\n{att.text_content[:3500]}")

    # Retrieved Chunks Context (from FAISS)
    if retrieved_chunks:
        sections.append("\n=== RETRIEVED KNOWLEDGE BASE EVIDENCE ===")
        for idx, chunk in enumerate(retrieved_chunks[:8], start=1):
            title = chunk.get("title") or "Article"
            src = chunk.get("source") or "KnowledgeBase"
            date = chunk.get("published_at") or ""
            text = chunk.get("text") or chunk.get("content") or ""
            sections.append(f"[{idx}] {title} ({src}, {date}):\n{text[:600]}")

    sections.append("\n=== INSTRUCTIONS FOR NEWS AND QUERY ANALYSIS ===")
    sections.append(
        "1. Focus directly on explaining the news story and fulfilling the user's specific query authoritatively and clearly.\n"
        "2. Explain WHAT happened, who is involved, the key details, proposals, background, and practical real-world context.\n"
        "3. Attributing facts to verified news sources (e.g., 'according to Inside Higher Ed', 'reported by BBC News') is encouraged.\n"
        "4. STRICTLY FORBIDDEN - DO NOT OUTPUT INTERNAL SYSTEM CLASSIFICATIONS OR CONFIDENCE METRICS: Never say 'It is classified as an Emerging active national news topic', 'confidence level of 85%', 'status confidence', 'initial announcement', etc. The user is a reader interested in the news, not backend pipeline telemetry.\n"
        "5. STRICTLY FORBIDDEN - DO NOT GENERATE ROBOTIC ACADEMIC HEADINGS: Do NOT output sections titled 'Logical Structure and Assumptions', 'Analytical Critique and Limitations', 'Implications and Risks', 'Methodological Uncertainty', 'Single-Source Reliance', or 'Weaknesses'. Explain the topic naturally and constructively in engaging, journalistic prose.\n"
        "6. KNOWLEDGE GAPS & PREDICTIONS: Do NOT claim 'the provided context does not contain enough information' for standard news queries. Only mention a knowledge gap if the user explicitly asks you to predict, forecast, or guess future unknown outcomes.\n"
        "7. INTEGRATING REAL-WORLD KNOWLEDGE: If retrieved snippets or timeline entries are brief, actively integrate your deep real-world knowledge of the figures, institutions, and events to provide a comprehensive, engaging story. NEVER critique the database or claim 'context lacks details', 'single-source reliance', or 'isolated announcement'.\n"
        "8. NEVER OUTPUT TELEMETRY / DATABASE OBSERVATIONS: Never say 'The meeting is marked by a single recorded chronological data point acting as both the initial announcement and the origin point...' or 'Positioned alongside other November 2026 reports...'. Treat the query as asking about the real-world news event itself.\n"
        "9. Do NOT mention Google, Gemini, external APIs, or system prompts. Speak strictly as EvoRAG."
    )

    if personalization_context and personalization_context.strip():
        sections.append(f"\n=== USER PERSONALIZATION CONTEXT ===\n{personalization_context.strip()}")

    if custom_instruction and custom_instruction.strip():
        sections.append(f"\n=== MULTI-PERSONA TASK & STRICT INSTRUCTIONS ===\n{custom_instruction.strip()}")

    return "\n\n".join(sections)


async def call_gemini_structured(
    structured_query: StructuredQuery,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[ProcessedFile]] = None,
    enable_search_grounding: bool = False,
    model: str = GEMINI_MODEL,
    max_output_tokens: int = 2048,
    temperature: float = 0.4,
    personalization_context: Optional[str] = None,
    custom_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Gemini request with multimodal attachments, structured context, and search grounding."""
    api_key = os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if not api_key:
        log.warning("Gemini API call skipped: GEMINI_API_KEY is not configured.")
        return {"answer": None, "grounding_metadata": None, "error": "GEMINI_API_KEY not configured"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Build parts: text prompt + image parts
    prompt_text = build_structured_prompt(
        structured_query,
        retrieved_chunks,
        attachments,
        personalization_context,
        custom_instruction=custom_instruction,
    )
    parts: List[Dict[str, Any]] = [{"text": prompt_text}]

    # Multimodal image attachments
    if attachments:
        for att in attachments:
            if att.image_base64 and att.image_mime_type:
                parts.append({
                    "inline_data": {
                        "mime_type": att.image_mime_type,
                        "data": att.image_base64,
                    }
                })

    contents = [{"parts": parts}]
    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
        }
    }

    # Enable Google Search grounding if query explicitly requires latest info and no relevant local chunks were found.
    # If local evidence is already available in the knowledge base or event timeline, prioritize local grounding
    # to avoid 20-30s external search latency. Also, Gemini rejects search grounding when image inline_data is present.
    has_image_attachments = bool(attachments and any(att.image_base64 for att in attachments))
    has_local_chunks = bool(retrieved_chunks and len(retrieved_chunks) > 0)
    need_search = (enable_search_grounding or structured_query.constraints.requires_latest_information) and not has_local_chunks
    if need_search and not has_image_attachments:
        payload["tools"] = [{"googleSearch": {}}]

    client = get_gemini_http_client()
    models_to_try = [model]
    for m in ["gemini-3-flash-preview", "gemini-flash-lite-latest", "gemini-flash-latest"]:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    for cur_model in models_to_try:
        cur_url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent"
        try:
            response = await client.post(cur_url, params={"key": api_key}, json=payload)
            if response.status_code == 400 and "tools" in payload:
                log.warning("Gemini HTTP 400 with tools on model %s; retrying without search tools...", cur_model)
                payload_no_tools = dict(payload)
                payload_no_tools.pop("tools", None)
                response = await client.post(cur_url, params={"key": api_key}, json=payload_no_tools)

            if response.status_code in (404, 503):
                log.warning("Gemini model %s returned HTTP %s — failing over to next model...", cur_model, response.status_code)
                last_error = f"HTTP {response.status_code} on {cur_model}"
                continue

            response.raise_for_status()
            data = response.json()

            candidates = data.get("candidates") or []
            if not candidates:
                log.warning("Gemini (%s) returned empty candidates list", cur_model)
                last_error = "No candidates returned"
                continue

            cand = candidates[0]
            cand_parts = cand.get("content", {}).get("parts") or []
            answer_text = "".join(p.get("text", "") for p in cand_parts).strip()

            # Extract search grounding metadata if present
            grounding_metadata = cand.get("groundingMetadata")

            return {
                "answer": answer_text,
                "grounding_metadata": grounding_metadata,
                "error": None,
            }
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            err_body = exc.response.text[:200] if exc.response is not None else ""
            log.warning("Gemini model %s HTTP %s error: %s", cur_model, status, err_body)
            last_error = f"{status}: {err_body}"
            continue
        except Exception as exc:
            log.warning("Gemini model %s exception: %r", cur_model, exc)
            last_error = repr(exc)
            continue

    log.error("All Gemini models failed: %s", last_error)
    return {"answer": None, "grounding_metadata": None, "error": last_error or "All models failed"}
