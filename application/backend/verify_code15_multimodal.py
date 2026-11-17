"""
Step 2 verification: run from application/backend with `python verify_code15_multimodal.py`.

Uses a temporary companion database (seeded at startup) so backend/companion.db is untouched.
Checks GET /patients, GET /patients/pat_ramesh_001/visits, POST /predict/code15-multimodal
(sample .npy upload), POST /visits/{id}/approve, /summarize and /chat fallbacks.
"""
import os
import shutil
import sys
import tempfile
import json

import torch  # noqa: F401  (Windows: torch before scipy/sklearn)

TMP = tempfile.mkdtemp(prefix="vitalsense_verify_")
os.environ["COMPANION_DB_PATH"] = os.path.join(TMP, "companion_verify.db")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

failures = 0


def check(name, ok, info=""):
    global failures
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {info}" if info else ""))
    if not ok:
        failures += 1


def sse_text(resp):
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            evt = json.loads(line[6:])
            if evt.get("done"):
                return evt["full_text"]
            if evt.get("error"):
                return "ERROR: " + evt["error"]
    return ""


try:
    with TestClient(main.app) as client:
        health = client.get("/health").json()
        check("health: db + engine ready", health["db_ready"] and health["code15_engine_ready"],
              f"weights_loaded={health['weights_loaded']} torch={health['torch_available']} llm={health['llm_provider']}")

        r = client.get("/patients")
        ids = sorted(p["id"] for p in r.json()) if r.status_code == 200 else []
        check("GET /patients returns the 2 seeded patients",
              r.status_code == 200 and ids == ["pat_eleanor_002", "pat_ramesh_001"], str(ids))

        r = client.get("/patients/pat_ramesh_001/visits")
        visits = r.json().get("visits", []) if r.status_code == 200 else []
        check("GET /patients/pat_ramesh_001/visits returns 2 visits, newest first",
              len(visits) == 2 and visits[0]["visit_date"] >= visits[1]["visit_date"],
              str([v["visit_date"] for v in visits]))
        check("seeded visits have deltas and diet plans",
              all(v["delta_summary"] and v["diet_plan"].get("daily_targets") for v in visits))

        samples = [s["filename"] for s in client.get("/ecg/code15-samples").json()["samples"]]
        npy = sorted(n for n in samples if n.endswith(".npy"))[0]  # Req 21.7: first .npy sample
        from pathlib import Path
        sample_path = Path(main.__file__).resolve().parent.parent / "sample_code15_ecgs" / npy
        with open(sample_path, "rb") as fh:
            r = client.post(
                "/predict/code15-multimodal",
                data={"patient_id": "pat_ramesh_001", "visit_date": "2026-12-15",
                      "hospital_name": "City General Hospital", "adherence_notes": "Walked 3 days a week.",
                      "hr": "92", "bp_sys": "150", "bp_dia": "92", "spo2": "95", "rr": "18", "temp": "36.8",
                      "glucose": "160", "cholesterol": "236", "creatinine": "1.6", "hrv": "30",
                      "AF": "true", "LBBB": "true"},
                files={"file": (npy, fh, "application/octet-stream")},
            )
        check("POST /predict/code15-multimodal returns 200", r.status_code == 200, r.text[:300] if r.status_code != 200 else "")
        visit = r.json() if r.status_code == 200 else {}
        pred = visit.get("predictions", {})
        check("ResNet1d ran (heart age from model)",
              pred.get("heart_age_source") == "resnet1d" and isinstance(pred.get("predicted_heart_age"), (int, float)),
              f"heart_age={pred.get('predicted_heart_age')} source={pred.get('heart_age_source')}")
        check("risk score + tier", 0 <= pred.get("overall_risk_score", -1) <= 100,
              f"{pred.get('overall_risk_score')}% {pred.get('severity_tier')}; arrhythmias={pred.get('arrhythmias')}")
        ds = visit.get("delta_summary") or {}
        check("deltas computed against the previous visit",
              ds.get("is_baseline") is False and ds.get("previous_visit_id") == "vis_ramesh_02",
              f"risk_delta={ds.get('risk_delta')} trend={ds.get('overall_trend')}")
        plan = visit.get("diet_plan") or {}
        check("diet plan generated", bool(plan.get("daily_targets")) and bool(plan.get("exercise_plan")),
              f"sodium={plan.get('daily_targets', {}).get('sodium_mg')} max_hr={plan.get('exercise_plan', {}).get('max_hr_ceiling')} by={plan.get('generated_by')}")
        hm = visit.get("heatmap") or []
        check("12x128 heatmap + 12x400 leads preview",
              len(hm) == 12 and all(len(row) == 128 for row in hm)
              and len(visit.get("ecg_metadata", {}).get("leads_preview", {}).get("V6", [])) == 400)
        check("SHAP drivers (<= 7)", 0 < len(visit.get("shap", [])) <= 7,
              ", ".join(d["display_name"] for d in visit.get("shap", [])))
        check("new visit saved unapproved", visit.get("approved_by_staff") is False)
        check("visit list now has 3 visits, new one first",
              client.get("/patients/pat_ramesh_001/visits?include_signals=false").json()["visits"][0]["id"] == visit.get("id"))

        r = client.post(f"/visits/{visit.get('id')}/approve",
                        json={"approved_by_staff": True, "staff_notes": "Reviewed by dietitian."})
        check("POST /visits/{id}/approve", r.status_code == 200 and r.json()["approved_by_staff"] is True
              and r.json()["diet_plan"]["staff_review"]["staff_notes"] == "Reviewed by dietitian.")
        check("approve unknown visit -> 404",
              client.post("/visits/nope/approve", json={"approved_by_staff": True}).status_code == 404)

        r = client.post("/patients", json={"name": "Test Patient", "mrn": "MRN-TEST-1", "age": 40, "gender": "F"})
        check("POST /patients creates a profile", r.status_code == 201 and r.json()["visit_count"] == 0)

        text = sse_text(client.post("/summarize", json={"patient_id": "pat_ramesh_001"}))
        check("/summarize longitudinal fallback", len(text) > 100, text[:160] + "...")
        text = sse_text(client.post("/chat", json={
            "patient_id": "pat_ramesh_001", "is_staff_mode": False,
            "messages": [{"role": "user", "content": "Can I eat paneer pakoda tonight, or why was my walking reduced?"}]}))
        check("/chat longitudinal fallback", len(text) > 50, text[:160] + "...")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
