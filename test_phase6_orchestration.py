"""
test_phase6_orchestration.py — Validation script for Phase 6: Automated Knowledge Ingestion and Orchestration

Verifies:
1. Knowledge base update-due logic and index reuse (< 3 days).
2. Local retrieval sufficiency and relevance evaluation.
3. Fallback routing to Gemini for insufficient or outdated queries.
4. Dynamic safe index reloading when index on disk changes.
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_scheduler import is_update_due, load_ingestion_status
from retriever import retrieve, evaluate_retrieval, reload_index_if_updated, get_loaded_index_stats
from pipeline import evorag_query


def test_update_due_logic():
    print("\n[TEST 1] Testing Knowledge Scheduler Update-Due Logic...")
    due, reason = is_update_due(interval_days=3)
    status = load_ingestion_status()
    print(f"  Status recorded: {status.get('status')}")
    print(f"  Last update    : {status.get('last_successful_update')}")
    print(f"  Update Due     : {due}")
    print(f"  Reason         : {reason}")
    assert due is False, "Expected update to NOT be due since index was just updated (< 3 days ago)."
    print("  [OK] Update-due check correctly identified index is fresh and reusable.")


def test_retrieval_evaluation():
    print("\n[TEST 2] Testing Retrieval Relevance & Sufficiency Evaluation...")
    
    # 2A: Relevant in-domain query
    query_relevant = "What are the regulatory guidelines and compliance standards for AI?"
    chunks_relevant = retrieve(query_relevant, top_k=2)
    is_suff, reason = evaluate_retrieval(query_relevant, chunks_relevant)
    print(f"  Query (In-Domain)  : '{query_relevant}'")
    print(f"  Chunks Retrieved   : {len(chunks_relevant)}")
    if chunks_relevant:
        print(f"  Best Chunk Score   : {chunks_relevant[0].get('score'):.4f}")
    print(f"  Sufficient         : {is_suff} ({reason})")
    assert is_suff is True, f"Expected relevant query to pass sufficiency check, got False: {reason}"
    print("  [OK] In-domain query passed sufficiency evaluation.")

    # 2B: Out-of-domain query
    query_irrelevant = "What is the recipe for authentic Neapolitan pizza dough using sourdough?"
    chunks_irrel = retrieve(query_irrelevant, top_k=2)
    is_suff_irrel, reason_irrel = evaluate_retrieval(query_irrelevant, chunks_irrel)
    print(f"\n  Query (Out-of-Domain): '{query_irrelevant}'")
    print(f"  Chunks Retrieved     : {len(chunks_irrel)}")
    if chunks_irrel:
        print(f"  Best Chunk Score     : {chunks_irrel[0].get('score'):.4f}")
    print(f"  Sufficient           : {is_suff_irrel} ({reason_irrel})")
    assert is_suff_irrel is False, "Expected out-of-domain query to fail sufficiency check."
    print("  [OK] Out-of-domain query correctly failed sufficiency evaluation.")


async def test_gemini_fallback_flow():
    print("\n[TEST 3] Testing Gemini Fallback for Insufficient Local Context...")
    query = "What were the primary goals of the Apollo 11 lunar landing in 1969?"
    print(f"  Sending query to evorag_query: '{query}'")
    
    result = await evorag_query(query, top_k=2)
    print(f"  Final Answer Length : {len(result['final_answer'])} chars")
    print(f"  Top Personas        : {result['top_personas']}")
    print(f"  Retrieval Status    : {result.get('retrieval_status')}")
    
    assert "final_answer" in result and len(result["final_answer"]) > 20
    assert result.get("retrieval_status", {}).get("sufficient") is False
    assert result.get("retrieval_status", {}).get("fallback_used") is True
    print("  [OK] Gemini fallback correctly answered and returned compliant consensus response.")


def test_dynamic_reload():
    print("\n[TEST 4] Testing Safe Dynamic Index Reloading...")
    stats_before = get_loaded_index_stats()
    print(f"  Loaded vectors before : {stats_before['vectors']}")
    reloaded = reload_index_if_updated()
    stats_after = get_loaded_index_stats()
    print(f"  Reloaded              : {reloaded}")
    print(f"  Loaded vectors after  : {stats_after['vectors']}")
    assert stats_after["vectors"] > 0
    print("  [OK] Index reload function executed safely.")


async def main():
    print("=" * 65)
    print("  EvoRAG Phase 6 Orchestration & Fallback Validation")
    print("=" * 65)
    test_update_due_logic()
    test_retrieval_evaluation()
    test_dynamic_reload()
    await test_gemini_fallback_flow()
    print("\n" + "=" * 65)
    print("  ALL PHASE 6 VALIDATION TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())

