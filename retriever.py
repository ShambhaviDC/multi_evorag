"""
retriever.py — EvoRAG Phase 3: Retrieval Pipeline

Public API
----------
    retrieve(query, top_k)    → List[dict]   (top-k chunk dicts with 'score')
    build_prompt(query, chunks) → str        (assembled RAG prompt)
    ask_ollama(prompt)        → str          (LLM answer string)
    answer(query, top_k)      → dict         ({answer, sources, query})

This module is imported by api.py (Phase 3) and by the persona engine (Phase 4).
"""

import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import httpx

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    RETRIEVAL_TOP_K,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    RETRIEVAL_MIN_CANDIDATES,
    PERSONA_TIMEOUT_SECONDS,
    GEMINI_ONLY_MODE,
    FAISS_INDEX_FILE,
    META_FILE,
)
from embedder import query as faiss_query, load_index, build_or_load, release_model
from utils import call_gemini_async

log = logging.getLogger(__name__)

# ── Module-level index cache with auto-reload synchronization ─────────────
_index = None
_metadata = None
_last_index_mtime: float = 0.0


def reload_index_if_updated() -> bool:
    """Check if the FAISS index or metadata on disk was modified, and safely reload if valid.

    Prevents loading corrupted or partially written indices.
    Returns True if a new index was reloaded, False otherwise.
    """
    global _index, _metadata, _last_index_mtime

    if not FAISS_INDEX_FILE.exists() or not META_FILE.exists():
        return False

    try:
        current_mtime = max(FAISS_INDEX_FILE.stat().st_mtime, META_FILE.stat().st_mtime)
        if _index is None or _metadata is None or current_mtime > _last_index_mtime:
            new_index, new_metadata = load_index()
            # Strict safety check: index vector count MUST match metadata entries count
            if new_index.ntotal == len(new_metadata):
                _index = new_index
                _metadata = new_metadata
                _last_index_mtime = current_mtime
                log.info(
                    "Retriever successfully reloaded FAISS index: %d vectors, dim=%d (disk mtime: %s)",
                    _index.ntotal,
                    _index.d,
                    _last_index_mtime,
                )
                return True
            else:
                log.warning(
                    "Index mismatch detected (%d vectors in FAISS vs %d metadata entries). "
                    "Keeping existing valid index in memory.",
                    new_index.ntotal,
                    len(new_metadata),
                )
    except Exception as exc:
        log.error("Failed to reload index from disk: %s. Continuing with existing index.", exc)
    return False


def get_loaded_index_stats() -> Dict:
    """Return in-memory index stats and last reload timestamp."""
    _ensure_index()
    return {
        "vectors": _index.ntotal if _index is not None else 0,
        "dim": _index.d if _index is not None else 0,
        "metadata_count": len(_metadata) if _metadata is not None else 0,
        "last_mtime": _last_index_mtime,
    }


def _ensure_index():
    """Ensure index is loaded and synchronized with disk updates."""
    global _index, _metadata
    if _index is None or _metadata is None:
        reload_index_if_updated()
    else:
        # Check if disk index has been updated by background scheduler
        reload_index_if_updated()


def evaluate_retrieval(
    query: str,
    chunks: List[Dict],
    relevance_threshold: float = 1.35,
    min_chunks: int = 1,
    min_total_words: int = 40,
) -> Tuple[bool, str]:
    """
    Evaluate the relevance, freshness, and sufficiency of retrieved chunks for a user query.

    Returns:
        (is_sufficient: bool, rationale: str)
    """
    if not chunks:
        return False, "No chunks retrieved from the local knowledge base."

    if len(chunks) < min_chunks:
        return False, f"Retrieved only {len(chunks)} chunks, below minimum threshold ({min_chunks})."

    total_words = sum(c.get("word_count", 0) or len(c.get("text", "").split()) for c in chunks)
    if total_words < min_total_words:
        return False, f"Total retrieved context is too sparse ({total_words} words < {min_total_words} words)."

    # 1. Relevance check using FAISS L2 score (lower is more similar)
    best_score = float(chunks[0].get("score", 999.0))
    if best_score > relevance_threshold:
        return False, (
            f"Retrieved context is not sufficiently relevant "
            f"(best L2 score {best_score:.3f} > threshold {relevance_threshold:.2f})."
        )

    # 2. Freshness check for "latest news" queries
    latest_news_pattern = re.compile(
        r"\b(latest|breaking|today|current|recent|newest|this week|yesterday|2026)\b",
        re.IGNORECASE,
    )
    if latest_news_pattern.search(query):
        now = datetime.now(timezone.utc)
        has_fresh_chunk = False
        parsed_dates = []

        for c in chunks:
            pub = c.get("published_at")
            if pub:
                try:
                    dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age_days = (now - dt).total_seconds() / 86400.0
                    parsed_dates.append(age_days)
                    if age_days <= 14:  # within 14 days
                        has_fresh_chunk = True
                except Exception:
                    pass

        if parsed_dates and not has_fresh_chunk:
            min_age = min(parsed_dates)
            return False, (
                f"Query requests latest/recent news, but local information is {min_age:.1f} days old (outdated)."
            )

    return True, f"Sufficient relevant context retrieved (best score {best_score:.3f}, {total_words} words)."


def _document_keys(chunk: Dict) -> List[str]:
    """Return all stable document-level identities for retrieval deduplication."""
    keys = []

    title = " ".join(str(chunk.get("title") or "").lower().split())
    if title:
        keys.append(f"title:{title}")

    url_hash = str(chunk.get("url_hash") or "").strip().lower()
    if url_hash:
        keys.append(f"url_hash:{url_hash}")

    url = str(chunk.get("url") or "").strip().lower()
    if url:
        keys.append(f"url:{url.rstrip('/')}")

    if not keys:
        keys.append(f"chunk:{chunk.get('chunk_id', id(chunk))}")
    return keys


def deduplicate_chunks(chunks: List[Dict], limit: int) -> List[Dict]:
    """Keep the best-scoring chunk from each source document."""
    unique: List[Dict] = []
    seen = set()

    for chunk in chunks:
        keys = _document_keys(chunk)
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        unique.append(chunk)
        if len(unique) >= limit:
            break

    return unique


# ══════════════════════════════════════════════════════════════════════════════
# 1. RETRIEVE — semantic search over FAISS index
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
    """
    Embed `query` and return the top-k most similar chunks from the FAISS index.

    Args:
        query:  The user question / search string.
        top_k:  Number of chunks to return.

    Returns:
        List of chunk dicts (each has 'text', 'title', 'source', 'url', 'score', …).
    """
    _ensure_index()
    candidate_k = min(
        len(_metadata),
        max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, RETRIEVAL_MIN_CANDIDATES),
    )
    candidates = faiss_query(
        query,
        top_k=candidate_k,
        index=_index,
        metadata=_metadata,
        deduplicate=False,
    )
    results = deduplicate_chunks(candidates, limit=top_k)

    if len(results) < min(top_k, len(candidates)):
        log.info(
            "Retrieved %s unique documents from %s candidates for query: '%s'",
            len(results),
            len(candidates),
            query[:60],
        )
    else:
        log.info(
            "Retrieved %s deduplicated chunks from %s candidates for query: '%s'",
            len(results),
            len(candidates),
            query[:60],
        )
    return results


def retrieval_diagnostics(query: str, top_k: int = RETRIEVAL_TOP_K) -> Dict:
    """Return retrieval diversity metrics for a query without changing API contracts."""
    _ensure_index()
    candidate_k = min(
        len(_metadata),
        max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, RETRIEVAL_MIN_CANDIDATES),
    )
    candidates = faiss_query(
        query,
        top_k=candidate_k,
        index=_index,
        metadata=_metadata,
        deduplicate=False,
    )
    unique_results = deduplicate_chunks(candidates, limit=top_k)

    candidate_title_keys = {
        next((key for key in _document_keys(chunk) if key.startswith("title:")), _document_keys(chunk)[0])
        for chunk in candidates
    }
    top_urls = [chunk.get("url", "") for chunk in unique_results]
    scores = [float(chunk.get("score", 0.0)) for chunk in unique_results]

    return {
        "query": query,
        "requested_top_k": top_k,
        "candidate_count": len(candidates),
        "returned_count": len(unique_results),
        "unique_document_count": len({
            next((key for key in _document_keys(chunk) if key.startswith("title:")), _document_keys(chunk)[0])
            for chunk in unique_results
        }),
        "unique_source_count": len({chunk.get("source", "") for chunk in unique_results}),
        "duplicate_candidate_ratio": (
            round(1 - (len(candidate_title_keys) / len(candidates)), 4)
            if candidates else 0.0
        ),
        "average_retrieval_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "top_retrieved_urls": top_urls,
        "results": [
            {
                "rank": rank,
                "score": round(float(chunk.get("score", 0.0)), 4),
                "source": chunk.get("source", ""),
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
            }
            for rank, chunk in enumerate(unique_results, start=1)
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROMPT ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Assemble a RAG prompt from the retrieved chunks.

    Format:
        System instruction
        Numbered context passages (title + source + text)
        User question

    Args:
        query:  The original user question.
        chunks: List of retrieved chunk dicts (must have 'text', 'title', 'source').

    Returns:
        The full prompt string to send to Ollama.
    """
    system = (
        "You are a factual news analysis assistant. "
        "Use ONLY the context passages provided below to answer the question. "
        "Cite the passage number (e.g. [1], [2]) when you use information from it. "
        "If the context does not contain enough information to answer, say so clearly — "
        "do not invent facts."
    )

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        title  = chunk.get("title", "Untitled")[:100]
        source = chunk.get("source", "unknown")
        # Truncate to 150 words so phi3 can respond quickly on CPU
        words  = chunk.get("text", "").strip().split()
        text   = " ".join(words[:150])
        block  = (
            f"[{i}] Source: {source} | Title: {title}\n"
            f"{text}"
        )
        context_blocks.append(block)

    context_section = "\n\n".join(context_blocks)

    prompt = (
        f"{system}\n\n"
        f"--- CONTEXT ---\n"
        f"{context_section}\n"
        f"--- END CONTEXT ---\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# 3. OLLAMA CALL — synchronous HTTP (used by validate_phase3 & CLI)
# ══════════════════════════════════════════════════════════════════════════════

def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Send a prompt to Ollama's /api/generate endpoint (non-streaming).

    Args:
        prompt: The full prompt string.
        model:  Ollama model tag (default from config).

    Returns:
        The model's response text string.

    Raises:
        httpx.HTTPError or ConnectionError if Ollama is unreachable.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    log.info(f"Calling Ollama ({model}) — prompt length: {len(prompt)} chars")
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer_text = data.get("response", "").strip()
            log.info(f"Ollama responded ({len(answer_text)} chars).")
            return answer_text
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. ASYNC OLLAMA CALL — used by FastAPI endpoint (non-blocking)
# ══════════════════════════════════════════════════════════════════════════════

async def ask_ollama_async(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Async version of ask_ollama — used inside FastAPI route handlers.

    Args:
        prompt: The full prompt string.
        model:  Ollama model tag.

    Returns:
        The model's response text string.
    """
    if GEMINI_ONLY_MODE:
        result = await call_gemini_async(prompt)
        if not result:
            raise ConnectionError(
                "Gemini-only mode: Gemini call failed (GEMINI_ONLY_MODE=True, Ollama disabled)."
            )
        return result

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    # Use PERSONA_TIMEOUT_SECONDS (+10 s grace) so the httpx socket deadline
    # is always slightly longer than the asyncio.wait_for deadline in
    # personas.py.  This means asyncio cancels first (clean TimeoutError)
    # rather than httpx raising a ReadTimeout that the caller must handle
    # separately.
    http_timeout = PERSONA_TIMEOUT_SECONDS + 10
    log.info("[async] Calling Ollama (%s) — prompt length: %s chars (timeout=%ss)",
             model, len(prompt), http_timeout)
    try:
        async with httpx.AsyncClient(timeout=http_timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer_text = data.get("response", "").strip()
            if answer_text:
                log.info(
                    "[async] Ollama responded length=%s preview='%s'",
                    len(answer_text),
                    _preview(answer_text),
                )
            else:
                log.warning(
                    "[async] Ollama returned an empty response. keys=%s raw_response_len=%s "
                    "done=%s eval_count=%s",
                    sorted(data.keys()),
                    len(data.get("response", "")),
                    data.get("done"),
                    data.get("eval_count"),
                )
            return answer_text
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500] if exc.response is not None else ""
        log.error(
            "[async] Ollama HTTP %s: %s",
            exc.response.status_code if exc.response is not None else "?",
            body,
        )
        raise


# ══════════════════════════════════════════════════════════════════════════════
# 5. END-TO-END — retrieve + prompt + answer (sync, for CLI / tests)
# ══════════════════════════════════════════════════════════════════════════════

def answer(query: str, top_k: int = RETRIEVAL_TOP_K) -> Dict:
    """
    Full RAG pipeline — synchronous version.

    Steps:
        1. Retrieve top-k chunks from FAISS
        2. Build RAG prompt
        3. Send to Ollama
        4. Return structured result

    Returns:
        {
            "query":   str,
            "answer":  str,
            "sources": [{"title", "url", "source", "score"}, …]
        }
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "query":   query,
            "answer":  "No relevant context found in the index. Try fetching more articles first.",
            "sources": [],
        }

    prompt = build_prompt(query, chunks)
    release_model()
    answer_text = ask_ollama(prompt)

    sources = [
        {
            "title":  c.get("title", ""),
            "url":    c.get("url", ""),
            "source": c.get("source", ""),
            "score":  round(c.get("score", 0.0), 4),
        }
        for c in chunks
    ]

    return {
        "query":   query,
        "answer":  answer_text,
        "sources": sources,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. END-TO-END — async version (used by FastAPI)
# ══════════════════════════════════════════════════════════════════════════════

async def answer_async(query: str, top_k: int = RETRIEVAL_TOP_K) -> Dict:
    """
    Full RAG pipeline — async version used by the FastAPI endpoint.
    Identical logic to answer() but calls ask_ollama_async().
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "query":   query,
            "answer":  "No relevant context found in the index. Try fetching more articles first.",
            "sources": [],
        }

    prompt = build_prompt(query, chunks)
    release_model()
    answer_text = await ask_ollama_async(prompt)

    sources = [
        {
            "title":  c.get("title", ""),
            "url":    c.get("url", ""),
            "source": c.get("source", ""),
            "score":  round(c.get("score", 0.0), 4),
        }
        for c in chunks
    ]

    return {
        "query":   query,
        "answer":  answer_text,
        "sources": sources,
    }
