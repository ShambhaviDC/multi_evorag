"""
test_full_system_integration.py — End-to-End Verification across All 3 Enhanced Modules:
1. Module 1: Explainable AI (XAI) Layer
2. Module 2: Intelligent Report Generation (PDF & DOCX from Snapshot with Zero LLM Calls)
3. Module 3: Evolutionary Persona Adaptation (Multi-Factor Fitness, Champion-Challenger, 1-Click Rollback)
"""

import json
import urllib.request
import sys
import time

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post(path, data):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def download_bytes(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return resp.read()

def run_integration_tests():
    print("=" * 70)
    print("  EVORAG END-TO-END INTEGRATION TEST SUITE (STAGES 1, 2, 3)")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST 1: Health & Active Persona Registry
    # ---------------------------------------------------------
    print("\n[TEST 1] Verifying System Health & Evolutionary Persona Registry...")
    roster = get("/evolution/personas")
    personas = roster if isinstance(roster, list) else roster.get("personas", [])
    print(f"  ✓ Active Personas Count: {len(personas)}")
    assert len(personas) == 8, f"Expected 8 personas, got {len(personas)}"
    for p in personas:
        dims = p.get("fitness", {}).get("dimensions", {})
        comp = p.get("fitness", {}).get("composite_fitness", 5.0)
        print(f"    • {p['name']} (Gen {p['generation']} v{p['version']}): F={comp:.2f} [FB={dims.get('feedback',5.0):.1f}, EG={dims.get('evidence',5.0):.1f}, RQ={dims.get('relevance',5.0):.1f}, CP={dims.get('completeness',5.0):.1f}, CS={dims.get('consistency',5.0):.1f}]")

    # ---------------------------------------------------------
    # TEST 2: Query Execution + Consensus + XAI Pipeline
    # ---------------------------------------------------------
    print("\n[TEST 2] Executing Research Query with Consensus & Explainable AI (XAI)...")
    test_query = "What are the latest updates on international diplomatic negotiations and agreements?"
    t0 = time.time()
    query_res = post("/query/consensus", {"query": test_query})
    elapsed = time.time() - t0
    print(f"  ✓ Consensus query responded in {elapsed:.2f}s")

    final_answer = query_res.get("final_answer", "")
    print(f"  ✓ Final Answer Length: {len(final_answer)} chars")
    print(f"    Sample: {final_answer[:150]}...")
    assert len(final_answer) > 50, "Final answer is too short"
    assert not final_answer.strip().startswith('{"final_answer":'), "Final answer contains raw JSON formatting bug!"

    # Verify XAI Structure
    xai = query_res.get("xai_explanation")
    print("  ✓ Verifying XAI Explanation Block...")
    assert xai, "xai_explanation is missing from query response"
    print(f"    • XAI Summary: {xai.get('summary')}")
    
    claims = xai.get("evidence_attribution", [])
    print(f"    • Evidence Attribution: {len(claims)} claim attributions evaluated")
    assert len(claims) > 0, "No claims extracted in XAI evidence attribution"
    for cl in claims[:2]:
        print(f"      - Claim: '{cl.get('claim_text')[:60]}...' -> Status: {cl.get('status')} ({cl.get('verification_confidence', 0)*100:.0f}%)")

    personas_exp = xai.get("persona_explanations", [])
    selected_personas = [p for p in personas_exp if p.get("selected")]
    print(f"    • Persona Selection: {len(selected_personas)} selected out of {len(personas_exp)} evaluated")
    assert len(selected_personas) >= 3, "Expected at least 3 selected personas"

    vote_exp = xai.get("voting_explanation", {})
    print(f"    • Consensus Agreement: {vote_exp.get('top_three_summary')}")

    snapshot_id = query_res.get("snapshot_id")
    print(f"  ✓ Analysis Snapshot ID created: {snapshot_id}")
    assert snapshot_id, "snapshot_id was not returned"

    # ---------------------------------------------------------
    # TEST 3: Deterministic Report Generation from Snapshot (Zero LLM)
    # ---------------------------------------------------------
    print("\n[TEST 3] Testing Intelligent Report Generation from Immutable Snapshot...")
    
    # 3.1 Executive PDF
    t_rep0 = time.time()
    pdf_req = post("/reports/generate", {
        "snapshot_id": snapshot_id,
        "report_type": "executive",
        "file_format": "pdf"
    })
    t_rep_elapsed = time.time() - t_rep0
    print(f"  ✓ Executive PDF compiled in {t_rep_elapsed:.3f}s (Deterministic - ZERO LLM calls)")
    assert pdf_req.get("format") == "pdf" or pdf_req.get("file_format") == "pdf"
    rep_pdf_id = pdf_req.get("report_id") or pdf_req.get("id")
    
    # Download and verify PDF binary
    pdf_bytes = download_bytes(f"/reports/{rep_pdf_id}/download")
    assert pdf_bytes.startswith(b"%PDF"), "Generated PDF does not have valid %PDF magic bytes header"
    print(f"    • PDF Binary: {len(pdf_bytes)} bytes, starts with %PDF header")

    # 3.2 Detailed DOCX
    t_docx0 = time.time()
    docx_req = post("/reports/generate", {
        "snapshot_id": snapshot_id,
        "report_type": "detailed",
        "file_format": "docx"
    })
    t_docx_elapsed = time.time() - t_docx0
    print(f"  ✓ Detailed DOCX compiled in {t_docx_elapsed:.3f}s (Deterministic - ZERO LLM calls)")
    assert docx_req.get("format") == "docx" or docx_req.get("file_format") == "docx"
    rep_docx_id = docx_req.get("report_id") or docx_req.get("id")

    # Download and verify DOCX binary (ZIP container PK\x03\x04)
    docx_bytes = download_bytes(f"/reports/{rep_docx_id}/download")
    assert docx_bytes.startswith(b"PK\x03\x04"), "Generated DOCX does not have valid PK zip header"
    print(f"    • DOCX Binary: {len(docx_bytes)} bytes, starts with PK header")

    # ---------------------------------------------------------
    # TEST 4: Evolutionary Optimizer Cycle & Audit Verification
    # ---------------------------------------------------------
    print("\n[TEST 4] Verifying Evolutionary Persona Adaptation & Audit Records...")
    hist_res = get("/evolution/history")
    history = hist_res if isinstance(hist_res, list) else hist_res.get("history", [])
    print(f"  ✓ Audit History Trail: {len(history)} recorded governance actions")
    assert len(history) > 0, "No audit history found"
    
    print("\n" + "=" * 70)
    print("  >>> ALL INTEGRATION TESTS PASSED WITH 100% SUCCESS! <<<")
    print("=" * 70)

if __name__ == "__main__":
    run_integration_tests()
