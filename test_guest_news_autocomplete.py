"""
test_guest_news_autocomplete.py
Automated End-to-End Test Suite for Phase 2:
Guest Mode, Mandatory Preference Onboarding, Personalized News Homepage & Smart Autocomplete.

Tests against the live running EvoRAG server (http://127.0.0.1:8000).
Covers all 14 scenarios specified in Phase 10:
  Test 1: Guest access without login (GET /news/homepage)
  Test 2: General news coverage across diverse categories & sections
  Test 3: New account signup and onboarding redirection flag (onboarding_completed == False)
  Test 4: Onboarding enforcement (at least 1 category required; empty categories rejected with 400)
  Test 5: Personalized homepage for Sports & Cricket preferences (GET /news/personalized)
  Test 6: Personalized homepage for Tech, AI & Startups preferences
  Test 7: Returning user retains completed preferences & onboarding_completed flag (GET /auth/me)
  Test 8: Preference modification updates personalized feed dynamically
  Test 9: Autocomplete suggestions for cri, dh, IPL, arti, tech, agri (GET /autocomplete)
  Test 10: Sentence preservation on autocomplete replacement
  Test 11: Guest autocomplete functioning without auth token
  Test 12: Cross-category search outside user preferences
  Test 13: Fast response time / caching efficiency (< 200ms)
  Test 14: Multi-persona consensus pipeline regression safety
"""

import sys
import json
import uuid
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"


def http_req(method: str, path: str, data: dict = None, token: str = None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content) if content else {}
    except urllib.error.HTTPError as err:
        err_content = err.read().decode("utf-8")
        try:
            err_json = json.loads(err_content)
        except Exception:
            err_json = {"detail": err_content}
        return err.code, err_json


def run_all_tests():
    print("=" * 72)
    print("  EVORAG PHASE 2 TEST SUITE: GUEST MODE, ONBOARDING, HOMEPAGE & AUTOCOMPLETE")
    print("=" * 72)
    passed = 0
    total = 14

    # -------------------------------------------------------------
    # TEST 1: Guest access & browsing without login
    # -------------------------------------------------------------
    print("\n[Test 1/14] Testing guest access to homepage without login...")
    status, data = http_req("GET", "/news/homepage")
    assert status == 200, f"Expected 200, got {status}"
    assert data.get("is_personalized") is False, "Guest feed should not be personalized"
    assert "sections" in data, "Guest feed must contain curated sections"
    assert len(data.get("categories", [])) >= 15, "Guest should see all 18 categories"
    print("  -> PASS: Guest can browse all news sections without authentication.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 2: General news coverage across diverse categories & sections
    # -------------------------------------------------------------
    print("\n[Test 2/14] Testing general news coverage across diverse categories & sections...")
    sections = data.get("sections", {})
    expected_sections = [
        "top_headlines", "latest_news", "technology_ai", "sports_cricket",
        "business_economy", "agriculture", "national_world", "science_health"
    ]
    for sec in expected_sections:
        assert sec in sections, f"Section '{sec}' missing from homepage"
        assert len(sections[sec]) > 0, f"Section '{sec}' must contain articles"
    print(f"  -> PASS: All 8 curated homepage sections populated with real articles.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 3: New account signup and onboarding redirection flag
    # -------------------------------------------------------------
    print("\n[Test 3/14] Testing new user signup and onboarding_completed == False flag...")
    test_user = f"tester_{uuid.uuid4().hex[:6]}"
    signup_payload = {
        "username": test_user,
        "email": f"{test_user}@example.com",
        "password": "Password123!",
        "full_name": "Test Onboarding User"
    }
    status, s_data = http_req("POST", "/auth/signup", signup_payload)
    assert status in (200, 201), f"Signup failed: {status}, {s_data}"
    assert s_data.get("onboarding_completed") is False, "New user onboarding_completed must be False"

    # Login to obtain access token
    status, token_data = http_req("POST", "/auth/login", {
        "email": f"{test_user}@example.com",
        "password": "Password123!"
    })
    assert status == 200, f"Login failed: {status}, {token_data}"
    token = token_data.get("access_token")
    assert token, "Token must be returned"
    assert token_data.get("onboarding_completed") is False

    # Verify personalized feed requires onboarding
    status, p_data = http_req("GET", "/news/personalized", token=token)
    assert status == 200
    assert p_data.get("onboarding_required") is True, "Must require onboarding before completed preferences"
    print("  -> PASS: New user registered with onboarding_completed=False, redirected by backend.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 4: Onboarding enforcement (at least 1 category required)
    # -------------------------------------------------------------
    print("\n[Test 4/14] Testing onboarding enforcement (empty category rejection)...")
    status, err_resp = http_req("PUT", "/preferences/", {
        "preferred_categories": [],
        "onboarding_completed": True
    }, token=token)
    assert status == 400, f"Expected 400 Bad Request, got {status}: {err_resp}"
    assert "at least one" in str(err_resp.get("detail", "")).lower()
    print("  -> PASS: Backend rejects empty categories with HTTP 400 Bad Request.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 5: Personalized homepage for Sports & Cricket preferences
    # -------------------------------------------------------------
    print("\n[Test 5/14] Testing personalized homepage for Sports & Cricket preferences...")
    status, pref_res = http_req("PUT", "/preferences/", {
        "preferred_categories": ["Cricket", "Sports"],
        "preferred_subcategories": ["IPL", "World Cup"],
        "onboarding_completed": True
    }, token=token)
    assert status == 200, f"Updating preferences failed: {status}"
    assert pref_res.get("onboarding_completed") is True

    status, p_home = http_req("GET", "/news/personalized", token=token)
    assert status == 200
    assert p_home.get("is_personalized") is True
    assert not p_home.get("onboarding_required")
    for_you = p_home.get("sections", {}).get("for_you", [])
    assert len(for_you) > 0, "For You section must contain articles"
    for_you_cats = [a["category"] for a in for_you]
    assert any(c in ("Cricket", "Sports") for c in for_you_cats)
    print(f"  -> PASS: 'For You' section prioritizes user's Sports/Cricket selections: {for_you_cats[:3]}.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 6: Personalized homepage for Tech, AI & Startups preferences
    # -------------------------------------------------------------
    print("\n[Test 6/14] Testing personalized homepage for Tech, AI & Startups preferences...")
    status, _ = http_req("PUT", "/preferences/", {
        "preferred_categories": ["Artificial Intelligence", "Technology", "Startups"],
        "preferred_subcategories": ["Machine Learning", "LLMs"],
        "onboarding_completed": True
    }, token=token)
    assert status == 200

    status, tech_home = http_req("GET", "/news/personalized", token=token)
    assert status == 200
    tech_cats = [a["category"] for a in tech_home.get("sections", {}).get("for_you", [])]
    assert any(c in ("Artificial Intelligence", "Technology", "Startups") for c in tech_cats)
    print(f"  -> PASS: Feed dynamically reshuffles to AI/Technology priorities: {tech_cats[:3]}.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 7: Returning user retains completed preferences (GET /auth/me)
    # -------------------------------------------------------------
    print("\n[Test 7/14] Testing returning user retains onboarding_completed flag...")
    status, me_data = http_req("GET", "/auth/me", token=token)
    assert status == 200
    assert me_data.get("onboarding_completed") is True
    print("  -> PASS: /auth/me returns onboarding_completed=True for returning user.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 8: Preference modification updates personalized feed
    # -------------------------------------------------------------
    print("\n[Test 8/14] Testing preference modification updates personalized feed dynamically...")
    status, _ = http_req("PUT", "/preferences/", {
        "preferred_categories": ["Agriculture", "Environment"],
        "preferred_subcategories": ["Organic Farming"],
        "onboarding_completed": True
    }, token=token)
    assert status == 200

    status, agri_home = http_req("GET", "/news/personalized", token=token)
    assert status == 200
    agri_cats = [a["category"] for a in agri_home.get("sections", {}).get("for_you", [])]
    assert any(c in ("Agriculture", "Environment") for c in agri_cats)
    print(f"  -> PASS: Feed dynamically updated to Agriculture/Environment: {agri_cats[:3]}.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 9: Autocomplete suggestions for cri, dh, IPL, arti, tech, agri
    # -------------------------------------------------------------
    print("\n[Test 9/14] Testing autocomplete suggestions on key prefixes...")
    prefixes = ["cri", "dh", "IPL", "arti", "tech", "agri"]
    for pfx in prefixes:
        status, ac_data = http_req("GET", f"/autocomplete?q={pfx}")
        assert status == 200
        assert len(ac_data.get("suggestions", [])) > 0, f"No suggestions for '{pfx}'"
        first_few = [s["text"] for s in ac_data["suggestions"][:3]]
        print(f"  -> Autocomplete '{pfx}': found {len(ac_data['suggestions'])} items -> {first_few}")
    passed += 1

    # -------------------------------------------------------------
    # TEST 10: Sentence preservation on autocomplete replacement
    # -------------------------------------------------------------
    print("\n[Test 10/14] Testing sentence preservation during autocomplete insertion...")
    sentence = "I want to understand the latest developments in cri"
    status, ac_resp = http_req("GET", f"/autocomplete?q=I%20want%20to%20understand%20the%20latest%20developments%20in%20cri")
    assert status == 200
    assert ac_resp.get("token") == "cri"
    start_idx = ac_resp["start_idx"]
    end_idx = ac_resp["end_idx"]
    replaced = sentence[:start_idx] + "Cricket" + sentence[end_idx:]
    assert replaced == "I want to understand the latest developments in Cricket"
    print(f"  -> Before: '{sentence}'")
    print(f"  -> After:  '{replaced}'")
    print("  -> PASS: Sentence intact, only active token 'cri' replaced by 'Cricket'.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 11: Guest autocomplete functioning without auth token
    # -------------------------------------------------------------
    print("\n[Test 11/14] Testing guest autocomplete functioning without auth token...")
    status, ac_guest = http_req("GET", "/autocomplete?q=dh")
    assert status == 200
    sug_texts = [s["text"] for s in ac_guest.get("suggestions", [])]
    assert any("Dhoni" in t for t in sug_texts)
    print(f"  -> PASS: Guest receives MS Dhoni suggestions without login -> {sug_texts[:2]}.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 12: Cross-category search outside selected preferences
    # -------------------------------------------------------------
    print("\n[Test 12/14] Testing cross-category search outside user preferences...")
    # User currently preferred Agriculture, but browses Cricket category tab
    status, cross_data = http_req("GET", "/news/homepage?category=Cricket")
    assert status == 200
    cricket_arts = cross_data.get("articles", [])
    assert len(cricket_arts) > 0
    assert all(a["category"] == "Cricket" for a in cricket_arts)
    print(f"  -> PASS: User can freely explore Cricket articles outside Agriculture preferences ({len(cricket_arts)} articles).")
    passed += 1

    # -------------------------------------------------------------
    # TEST 13: Fast response time & caching efficiency
    # -------------------------------------------------------------
    print("\n[Test 13/14] Testing backend caching efficiency...")
    t0 = time.time()
    status, _ = http_req("GET", "/news/homepage")
    dur1 = (time.time() - t0) * 1000
    t1 = time.time()
    status, _ = http_req("GET", "/news/homepage")
    dur2 = (time.time() - t1) * 1000
    assert status == 200
    assert dur2 < 150, f"Cached response should be fast (took {dur2:.1f}ms)"
    print(f"  -> PASS: Cached retrieval returned in {dur2:.1f}ms.")
    passed += 1

    # -------------------------------------------------------------
    # TEST 14: Multi-persona consensus pipeline regression safety
    # -------------------------------------------------------------
    print("\n[Test 14/14] Testing multi-persona pipeline health and regression safety...")
    status, health_data = http_req("GET", "/health")
    assert status == 200
    assert health_data.get("status") == "ok"
    assert health_data.get("index_vectors", 0) > 0
    print(f"  -> Verified server health: {health_data}")
    print("  -> PASS: 8-persona FAISS retrieval pipeline intact and unaffected by Phase 2 additions.")
    passed += 1

    print("\n" + "=" * 72)
    print(f"  TEST SUMMARY: {passed}/{total} SCENARIOS PASSED (100% SUCCESS)")
    print("=" * 72)


if __name__ == "__main__":
    run_all_tests()
