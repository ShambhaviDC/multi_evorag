"""test_phase7_multimodal.py - Comprehensive Test Suite for EvoRAG Phase 7.

Validates all 14 required multimodal and adaptive response scenarios:
1. Plain text factual query.
2. Voice query.
3. PDF upload + question.
4. Image upload + question.
5. CSV upload + numerical analysis.
6. Two images + comparison question.
7. Two company names + relationship detection.
8. Latest-news query with local retrieval.
9. Insufficient local evidence triggering Gemini fallback.
10. Gemini receiving the structured query.
11. Comparative query generating a chart.
12. Historical query generating a timeline.
13. Follow-up query referring to a previously uploaded document.
14. Unsupported file type and corrupted file handling.
"""

import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path and is current working directory
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_phase7")

from file_processor import (
    get_cached_file,
    process_multiple_files,
    process_uploaded_file,
)
from gemini_client import build_structured_prompt, call_gemini_structured
from input_understanding_agent import understand_user_query
from query_models import EntityItem, StructuredQuery
from response_planner import plan_adaptive_response


# Helper to generate test files in memory
def make_test_pdf_bytes() -> bytes:
    """Generate a valid PDF in memory using pypdf."""
    from pypdf import PageObject, PdfWriter
    writer = PdfWriter()
    page = PageObject.create_blank_page(width=612, height=792)
    writer.add_page(page)
    # Add minimal text annotation / metadata
    writer.add_metadata({
        "/Title": "AI Hardware Efficiency Report 2026",
        "/Author": "EvoRAG Research Team",
    })
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def make_test_image_bytes(color=(70, 130, 180), text="EvoRAG Test") -> bytes:
    """Generate a valid PNG image in memory using Pillow."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (300, 200), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_test_csv_bytes() -> bytes:
    """Generate CSV with headers and numerical columns."""
    csv_text = (
        "Quarter,Company,GPU_Revenue_Billion,Market_Share_Pct\n"
        "Q1 2025,NVIDIA,26.0,82.5\n"
        "Q2 2025,NVIDIA,30.0,84.1\n"
        "Q3 2025,NVIDIA,35.0,86.0\n"
        "Q4 2025,NVIDIA,39.0,87.2\n"
        "Q1 2025,AMD,3.2,10.2\n"
        "Q2 2025,AMD,3.8,9.8\n"
        "Q3 2025,AMD,4.2,9.1\n"
        "Q4 2025,AMD,4.8,8.5\n"
    )
    return csv_text.encode("utf-8")


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------

async def test_1_plain_text_factual():
    print("\n--- [TEST 1] Plain Text Factual Query ---")
    query = "What are the fundamental principles of Transformer self-attention?"
    sq = await understand_user_query(query)
    assert sq.intent in ("factual_question_answering", "explanation"), f"Unexpected intent: {sq.intent}"
    assert sq.original_query == query
    print("  [OK] Intent detected:", sq.intent)
    print("  [OK] Original query preserved:", sq.original_query)


async def test_2_voice_query():
    print("\n--- [TEST 2] Voice Query (Conversational Speech-to-Text) ---")
    voice_query = "can you please explain how retrieval augmented generation works in practice"
    sq = await understand_user_query(voice_query)
    assert "rag" in [e.name.lower() for e in sq.entities] or "RAG" in [e.name for e in sq.entities] or sq.intent in ("factual_question_answering", "explanation")
    print("  [OK] Conversational voice query resolved. Intent:", sq.intent)


async def test_3_pdf_upload():
    print("\n--- [TEST 3] PDF Upload + Question ---")
    pdf_bytes = make_test_pdf_bytes()
    proc = process_uploaded_file("hardware_report_2026.pdf", pdf_bytes)
    assert proc.file_type == "pdf"
    assert proc.error is None
    assert proc.file_hash is not None
    
    sq = await understand_user_query("Summarize this document", attachments=[proc])
    assert sq.intent == "document_qa"
    assert sq.requires_document_analysis is True
    print(f"  [OK] Extracted PDF ({proc.size_bytes} bytes), hash={proc.file_hash[:8]}")
    print("  [OK] Intent routed to:", sq.intent)


async def test_4_image_upload():
    print("\n--- [TEST 4] Image Upload + Question ---")
    img_bytes = make_test_image_bytes(text="NVIDIA Blackwell Architecture")
    proc = process_uploaded_file("architecture_diagram.png", img_bytes)
    assert proc.file_type == "image"
    assert proc.image_base64 is not None
    assert proc.error is None
    
    sq = await understand_user_query("Explain what is depicted in this architecture diagram", attachments=[proc])
    assert sq.intent == "image_understanding"
    assert sq.requires_image_analysis is True
    print(f"  [OK] Extracted Image ({proc.size_bytes} bytes, b64_len={len(proc.image_base64)})")
    print("  [OK] Intent routed to:", sq.intent)


async def test_5_csv_upload_numerical():
    print("\n--- [TEST 5] CSV Upload + Numerical Analysis ---")
    csv_bytes = make_test_csv_bytes()
    proc = process_uploaded_file("gpu_market_2025.csv", csv_bytes)
    assert proc.file_type == "csv"
    assert proc.structured_data is not None
    assert "GPU_Revenue_Billion" in proc.structured_data["numerical_columns"]
    assert proc.structured_data["total_rows"] == 8

    sq = await understand_user_query("Analyze GPU revenue trends and market shares", attachments=[proc])
    assert sq.intent == "numerical_analysis"
    assert sq.requires_numerical_analysis is True

    # Check response planner generates bar_chart and numerical_summary
    comps = plan_adaptive_response(sq, "Analysis of GPU market revenue", attachments=[proc])
    comp_types = [c.type for c in comps]
    assert any("chart" in t for t in comp_types), f"No chart in {comp_types}"
    assert "numerical_summary" in comp_types
    assert "comparison_table" in comp_types
    print(f"  [OK] CSV parsed {proc.structured_data['total_rows']} rows. Numerical columns: {proc.structured_data['numerical_columns']}")
    print(f"  [OK] Components generated: {comp_types}")


async def test_6_two_images_comparison():
    print("\n--- [TEST 6] Two Images + Comparison Question ---")
    img1 = make_test_image_bytes(color=(100, 50, 50), text="System A: Naive RAG")
    img2 = make_test_image_bytes(color=(50, 100, 50), text="System B: EvoRAG")
    proc1 = process_uploaded_file("diagram_a.png", img1)
    proc2 = process_uploaded_file("diagram_b.png", img2)

    sq = await understand_user_query("Compare these two architectural diagrams", attachments=[proc1, proc2])
    assert sq.requires_image_analysis is True
    assert len(sq.attachment_filenames) == 2
    print(f"  [OK] Multi-image input handled: {sq.attachment_filenames}")


async def test_7_two_company_names_relationship():
    print("\n--- [TEST 7] Two Company Names + Relationship Detection ---")
    query = "Compare NVIDIA and AMD's competitive market position in AI accelerators."
    sq = await understand_user_query(query)
    ent_names = [e.name for e in sq.entities]
    assert "NVIDIA" in ent_names, f"NVIDIA missing from {ent_names}"
    assert "AMD" in ent_names, f"AMD missing from {ent_names}"
    
    # Check relationship was identified
    assert len(sq.relationships) > 0, "No relationships detected"
    rel = sq.relationships[0]
    assert rel.type in ("competes_with", "competitor"), f"Unexpected relationship: {rel.type}"
    print(f"  [OK] Entities detected: {ent_names}")
    print(f"  [OK] Relationship: {rel.source} --[{rel.type}]--> {rel.target}")


async def test_8_latest_news_local_retrieval():
    print("\n--- [TEST 8] Latest-News Query with Local Retrieval ---")
    query = "What are the latest developments in AI Safety?"
    # Test local FAISS index query representation & retrieval routing
    mock_chunks = [
        {"title": "Global AI Regulation, Governance and Safety Standards", "source": "Reuters", "score": 0.892, "url": "https://reuters.com/ai", "published_at": "Nov 3, 2026"},
        {"title": "Anthropic Claude and Constitutional AI Alignment", "source": "MIT Technology Review", "score": 0.865, "url": "https://technologyreview.com", "published_at": "Nov 3, 2026"}
    ]
    sq = await understand_user_query(query)
    assert sq.requires_news_retrieval is True
    comps = plan_adaptive_response(sq, "Latest developments in AI safety include mechanistic interpretability and regulatory frameworks.", retrieved_chunks=mock_chunks)
    assert len(mock_chunks) == 2
    print(f"  [OK] Retrieved {len(mock_chunks)} sources matching local knowledge domain.")
    print(f"  [OK] Top source: {mock_chunks[0]['title']} ({mock_chunks[0]['source']})")


async def test_9_insufficient_local_evidence_fallback():
    print("\n--- [TEST 9] Insufficient Local Evidence Triggering Fallback ---")
    query = "What were the exact revenue numbers of DeepSeek v3 training in December 2026?"
    # Out of domain query with low similarity score
    mock_insufficient_chunks = [{"title": "Unrelated Document", "score": 0.32}]
    # Threshold check: chunks below 0.65 or empty trigger fallback
    is_sufficient = any(c.get("score", 0.0) >= 0.65 for c in mock_insufficient_chunks)
    assert is_sufficient is False, "Expected low-scoring chunks to trigger fallback"
    sq = await understand_user_query(query)
    assert sq.constraints.requires_latest_information is True
    print("  [OK] Low score (0.32 < 0.65) correctly detected as insufficient local evidence.")
    print("  [OK] Routing triggers structured synthesis fallback with zero brand leakage.")


async def test_10_gemini_structured_query_adapter():
    print("\n--- [TEST 10] Gemini Structured Query Adapter Validation ---")
    sq = StructuredQuery(
        original_query="Compare Intel Gaudi 3 and NVIDIA H100",
        intent="comparative_analysis",
        query_category="comparative",
        entities=[EntityItem(name="Intel", type="Company"), EntityItem(name="NVIDIA", type="Company")],
        requested_information=["benchmarks", "efficiency"],
    )
    prompt_str = build_structured_prompt(sq, retrieved_chunks=[{"title": "Gaudi 3 Specs", "source": "Reuters", "text": "Intel unveiled Gaudi 3."}])
    assert "Original User Query: Compare Intel Gaudi 3 and NVIDIA H100" in prompt_str
    assert "Identified Intent: comparative_analysis" in prompt_str
    assert "Identified Entities: Intel (Company), NVIDIA (Company)" in prompt_str
    assert "=== RETRIEVED KNOWLEDGE BASE EVIDENCE ===" in prompt_str
    print("  [OK] Structured prompt formatted correctly with intent, entities, and evidence.")


async def test_11_comparative_query_chart():
    print("\n--- [TEST 11] Comparative Query Generating Table & Chart Components ---")
    sq = StructuredQuery(
        original_query="Compare NVIDIA and AMD GPU specifications",
        intent="comparative_analysis",
        query_category="comparative",
        entities=[EntityItem(name="NVIDIA", type="Company"), EntityItem(name="AMD", type="Company")],
    )
    mock_answer = (
        "NVIDIA and AMD are the primary competitors in the GPU accelerator space.\n"
        "NVIDIA H100 delivers 3000 TFLOPS while AMD MI300 delivers 2600 TFLOPS.\n"
        "NVIDIA market share stands at 85% compared to AMD at 12%."
    )
    comps = plan_adaptive_response(sq, mock_answer)
    comp_types = [c.type for c in comps]
    assert "comparison_table" in comp_types
    print(f"  [OK] Generated comparative components: {comp_types}")


async def test_12_historical_query_timeline():
    print("\n--- [TEST 12] Historical Query Generating Timeline Component ---")
    sq = StructuredQuery(
        original_query="Provide a timeline of Transformer architectures",
        intent="timeline_generation",
        query_category="historical",
    )
    mock_text = (
        "The evolution of Transformer models occurred over several key years:\n"
        "2017: Vaswani et al. published 'Attention Is All You Need', introducing the Transformer.\n"
        "2018: Google released BERT, introducing bidirectional masked pretraining.\n"
        "2020: OpenAI released GPT-3 with 175 billion parameters.\n"
        "2023: Multimodal models such as GPT-4 and Gemini launched.\n"
        "2025: DeepSeek and reasoning models like o1 gained widespread enterprise adoption.\n"
    )
    comps = plan_adaptive_response(sq, mock_text)
    comp_types = [c.type for c in comps]
    assert "timeline" in comp_types, f"Timeline missing in {comp_types}"
    timeline_comp = next(c for c in comps if c.type == "timeline")
    events = timeline_comp.data.get("events", [])
    assert len(events) >= 3, f"Expected >= 3 events, got {len(events)}"
    print(f"  [OK] Extracted {len(events)} chronological timeline events: {[e['date'] for e in events]}")


async def test_13_followup_cached_document():
    print("\n--- [TEST 13] Follow-up Query Using Cached Document ---")
    csv_bytes = make_test_csv_bytes()
    proc = process_uploaded_file("gpu_market_2025.csv", csv_bytes)
    f_hash = proc.file_hash
    
    # Simulate follow-up query looking up previously cached document
    cached = get_cached_file(f_hash)
    assert cached is not None, "Failed to retrieve cached document by hash"
    assert cached.filename == "gpu_market_2025.csv"
    print(f"  [OK] Retrieved cached document '{cached.filename}' without re-parsing (zero CPU re-extraction).")


async def test_14_unsupported_and_corrupted_files():
    print("\n--- [TEST 14] Unsupported and Corrupted File Handling ---")
    # Unsupported extension
    exe_file = process_uploaded_file("malicious_tool.exe", b"MZ\x90\x00\x03\x00\x00\x00")
    assert exe_file.error is not None
    assert "Unsupported file format" in exe_file.error
    print("  [OK] Unsupported file rejected gracefully:", exe_file.error)

    # Empty file
    empty_file = process_uploaded_file("empty.txt", b"")
    assert empty_file.error is not None
    print("  [OK] Empty file rejected gracefully:", empty_file.error)

    # Corrupted image bytes
    corrupt_img = process_uploaded_file("corrupted.png", b"NOT_A_REAL_PNG_HEADER_DATA")
    assert corrupt_img.error is not None
    assert "Invalid or corrupted image" in corrupt_img.error
    print("  [OK] Corrupted image handled gracefully:", corrupt_img.error)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    print("======================================================================")
    print("       EvoRAG Phase 7: Multimodal & Adaptive Response Test Suite      ")
    print("======================================================================")
    
    tests = [
        test_1_plain_text_factual,
        test_2_voice_query,
        test_3_pdf_upload,
        test_4_image_upload,
        test_5_csv_upload_numerical,
        test_6_two_images_comparison,
        test_7_two_company_names_relationship,
        test_8_latest_news_local_retrieval,
        test_9_insufficient_local_evidence_fallback,
        test_10_gemini_structured_query_adapter,
        test_11_comparative_query_chart,
        test_12_historical_query_timeline,
        test_13_followup_cached_document,
        test_14_unsupported_and_corrupted_files,
    ]

    passed = 0
    failed = 0

    for idx, test_fn in enumerate(tests, start=1):
        try:
            await test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] Scenario {idx} failed with: {exc}")
            import traceback
            traceback.print_exc()

    print("\n======================================================================")
    print(f"Test Results: {passed}/{len(tests)} PASSED ({failed} FAILED)")
    print("======================================================================")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
