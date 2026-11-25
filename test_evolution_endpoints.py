import urllib.request
import json
import sys

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

def main():
    print("=== 1. Testing GET /evolution/personas ===")
    roster = get("/evolution/personas")
    personas = roster if isinstance(roster, list) else roster.get("personas", [])
    print(f"Retrieved {len(personas)} personas.")
    assert len(personas) == 8, f"Expected 8 personas, got {len(personas)}"
    for p in personas[:3]:
        fit_val = p.get("fitness", {}).get("composite_fitness", p.get("composite_fitness"))
        print(f" - {p['persona_id']} (v{p['version']}, Gen {p['generation']}): Composite Fitness = {fit_val}")

    print("\n=== 2. Testing POST /evolution/trigger (strategy='mutation') ===")
    trig_res = post("/evolution/trigger", {"strategy": "mutation"})
    print(f"Trigger response status: {trig_res.get('status')}")
    candidate_id = trig_res.get("candidate_id")
    target_persona = trig_res.get("champion_id")
    challenger_name = trig_res.get("challenger_id")
    bench = trig_res.get("benchmark_result", {})
    print(f"Candidate: {candidate_id} for Champion '{target_persona}' -> Challenger '{challenger_name}'")
    print(f"Benchmark: Champion={bench.get('champion_score')} vs Challenger={bench.get('challenger_score')} (Delta%={bench.get('improvement_pct')}%)")
    assert candidate_id, "Candidate ID missing from trigger response"

    print("\n=== 3. Testing GET /evolution/candidates ===")
    cands_res = get("/evolution/candidates")
    candidates = cands_res if isinstance(cands_res, list) else cands_res.get("candidates", [])
    print(f"Staged candidates count: {len(candidates)}")
    matched = next((c for c in candidates if c.get("candidate_id") == candidate_id), None)
    assert matched, f"Candidate {candidate_id} not found in staged list"
    print(f"Found staged candidate {candidate_id}:")
    print(f" - Staged status: {matched.get('status')}")
    print(f" - Prompt diff entries: {len(matched.get('prompt_diff', []))}")

    print("\n=== 4. Testing POST /evolution/candidate/{id}/approve ===")
    approve_res = post(f"/evolution/candidate/{candidate_id}/approve", {})
    print(f"Approve response: status={approve_res.get('status')}, message={approve_res.get('message')}")
    assert approve_res.get("status") == "promoted"
    new_version = approve_res.get("promoted_persona", {}).get("version")
    print(f"Promoted to active version: v{new_version}")

    print("\n=== 5. Verifying Active Roster Update ===")
    roster_updated = get("/evolution/personas")
    updated_personas = roster_updated if isinstance(roster_updated, list) else roster_updated.get("personas", [])
    # Search for persona slot either by target_persona or by promoted persona_id
    promoted_persona = next((p for p in updated_personas if p["persona_id"] == target_persona or p["persona_id"] == approve_res.get("promoted_persona", {}).get("persona_id")), None)
    assert promoted_persona, f"Target persona {target_persona} not found"
    print(f"Active state: {promoted_persona['persona_id']} (v{promoted_persona['version']}, Gen {promoted_persona['generation']}, status={promoted_persona['status']})")

    print(f"\n=== 6. Testing 1-Click Rollback for {target_persona} to v1 ===")
    rollback_res = post(f"/evolution/rollback/{target_persona}", {"target_version": 1})
    print(f"Rollback response: status={rollback_res.get('status')}, message={rollback_res.get('message')}")
    assert rollback_res.get("status") == "rolled_back"

    roster_restored = get("/evolution/personas")
    restored_personas = roster_restored if isinstance(roster_restored, list) else roster_restored.get("personas", [])
    restored_persona = next((p for p in restored_personas if p["persona_id"] == target_persona), None)
    print(f"Restored state: {restored_persona['persona_id']} (v{restored_persona['version']}, Gen {restored_persona['generation']})")
    assert restored_persona["version"] == 1, f"Expected version 1 after rollback, got {restored_persona['version']}"

    print("\n=== 7. Testing GET /evolution/history ===")
    hist_res = get("/evolution/history")
    history = hist_res if isinstance(hist_res, list) else hist_res.get("history", [])
    print(f"Recorded audit history entries: {len(history)}")
    for h in history[-4:]:
        evt = h.get("event") or h.get("action")
        pid = h.get("persona_id") or h.get("slot_id")
        ver = h.get("version")
        print(f" - [{h.get('archived_at') or h.get('timestamp') or 'log'}] {evt}: {pid} (v{ver})")

    print("\n>>> ALL STAGE 3 EVOLUTION ENDPOINTS PASSED WITH 100% SUCCESS! <<<")

if __name__ == "__main__":
    main()
