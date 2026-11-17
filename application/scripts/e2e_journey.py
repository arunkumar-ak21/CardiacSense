"""
End-to-end API walk-through of the 6-step user journey against a RUNNING backend
(default http://localhost:8000). Run it against a throwaway database, e.g.:

    $env:COMPANION_DB_PATH="$env:TEMP\\vitalsense_e2e.db"; python -m uvicorn main:app --port 8000
    python scripts/e2e_journey.py

It creates one extra visit for Ramesh and one new patient, so do not point it at a database you care about.
"""
import json
import os
import sys
import urllib.request
import uuid

API = os.environ.get("VS_API", "http://localhost:8000")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = 0


def call(method, path, body=None, form=None, files=None):
    headers, data = {}, None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    elif form is not None:
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in form.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        for k, (name, blob) in (files or {}).items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                         f'Content-Type: application/octet-stream\r\n\r\n'.encode() + blob + b"\r\n")
        data = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(API + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {info}" if info else ""))


def sse_text(raw):
    for line in raw.splitlines():
        if line.startswith("data: "):
            evt = json.loads(line[6:])
            if "done" in evt:
                return evt["full_text"]
            if "error" in evt:
                return "ERROR " + evt["error"]
    return ""


# 1. Profiles -----------------------------------------------------------------
s, patients = call("GET", "/patients")
names = [p["name"] for p in patients] if s == 200 else []
check("1. GET /patients lists Ramesh & Eleanor", s == 200 and {"Ramesh Kumar", "Eleanor Vance"} <= set(names), str(names))
mrn = f"MRN-E2E-{uuid.uuid4().hex[:6]}"
s, new = call("POST", "/patients", body={"name": "Anita Sharma", "mrn": mrn, "age": 52, "gender": "F", "diet_preference": "vegetarian"})
check("1. POST /patients creates a new profile", s == 201 and new.get("visit_count") == 0, str(new)[:120])
s, v = call("GET", "/patients/pat_ramesh_001/visits?include_signals=false")
check("1. Ramesh's past visits load", s == 200 and v["total"] >= 2, f"{v.get('total')} visits")

# 2-3. Upload + model ----------------------------------------------------------
s, samples = call("GET", "/ecg/code15-samples")
check("2. sample ECG list", s == 200 and len(samples["samples"]) >= 8, f"{len(samples.get('samples', []))} files")
s, prev = call("GET", "/ecg/code15-samples/2_atrial_fib_lbbb_high_risk_4096x12.npy/preview")
check("2. 1-click sample preview (12 leads)", s == 200 and len(prev["leads_preview"]) == 12)
with open(os.path.join(ROOT, "sample_code15_ecgs", "2_atrial_fib_lbbb_high_risk_4096x12.npy"), "rb") as f:
    blob = f.read()
s, up = call("POST", "/ecg/code15-preview", form={}, files={"file": ("my_ecg.npy", blob)})
check("2. uploaded-file preview", s == 200 and len(up.get("leads_preview", {})) == 12)
form = {"patient_id": "pat_ramesh_001", "visit_date": "2026-11-14", "hospital_name": "City General Hospital",
        "adherence_notes": "Partially followed the previous diet & exercise plan. Skipped evening walks.",
        "diet_preference": "vegetarian", "hr": "92", "bp_sys": "158", "bp_dia": "96", "spo2": "95",
        "glucose": "170", "cholesterol": "245", "creatinine": "1.8", "hrv": "26", "AF": "true", "LBBB": "true"}
s, visit = call("POST", "/predict/code15-multimodal", form=form, files={"file": ("my_ecg.npy", blob)})
ok = s == 200 and isinstance(visit, dict)
check("3. Analyze & Save Visit (upload) -> 200", ok, "" if ok else str(visit)[:300])
if ok:
    p = visit["predictions"]
    check("3. severity % + tier", 0 <= p["overall_risk_score"] <= 100 and p["severity_tier"] in ("Low", "Moderate", "High", "Critical"),
          f"{p['overall_risk_score']}% {p['severity_tier']}")
    check("3. ResNet1d heart age", p.get("heart_age_source") == "resnet1d", f"{p['predicted_heart_age']} yrs")
    check("3. 12-lead graph + heatmap + SHAP", len(visit["ecg_metadata"]["leads_preview"]) == 12 and len(visit["heatmap"]) == 12 and len(visit["shap"]) > 0)
    d = visit["delta_summary"]
    check("4. Before vs Now computed vs previous visit", d and not d["is_baseline"],
          f"{d['previous_risk_score']}% -> {d['current_risk_score']}% ({d['risk_delta']:+})")
    check("4. what went wrong text", bool(d["what_went_wrong_summary"]), d["what_went_wrong_summary"][:140] + "...")
    plan = visit["diet_plan"]
    check("5. updated diet & exercise plan", all(k in plan["meals"] for k in ("breakfast", "lunch", "snack", "dinner")) and plan["foods_to_avoid"],
          f"sodium {plan['daily_targets']['sodium_mg']} mg, max HR {plan['exercise_plan']['max_hr_ceiling']}")
    check("5. why plan changed text", bool(plan["why_plan_changed_summary"]), plan["why_plan_changed_summary"][:140] + "...")

    # 5. Explanation + chat (same payload shape the frontend sends) ----------
    s, raw = call("POST", "/summarize", body={"patient_id": "pat_ramesh_001", "is_staff_mode": False})
    txt = sse_text(raw) if s == 200 else ""
    check("5. report explained in simple words (/summarize)", len(txt) > 100 and "ERROR" not in txt, txt[:140] + "...")
    for q in ("Why did my severity go up?", "What can I eat tonight?"):
        s, raw = call("POST", "/chat", body={"patient_id": "pat_ramesh_001", "is_staff_mode": False,
                                             "messages": [{"role": "user", "content": q}]})
        txt = sse_text(raw) if s == 200 else ""
        check(f"5. chat: {q}", len(txt) > 40 and "ERROR" not in txt, txt[:140] + "...")

    # 6. Staff approval -----------------------------------------------------
    plan["daily_targets"]["sodium_mg"] = 1300
    s, appr = call("POST", f"/visits/{visit['id']}/approve", body={"approved_by_staff": True,
                   "staff_notes": "Verified against renal-cardiac protocol.", "diet_plan": plan})
    check("6. staff edits + approves plan", s == 200 and appr["approved_by_staff"] and appr["diet_plan"]["daily_targets"]["sodium_mg"] == 1300)
    s, patients = call("GET", "/patients")
    r = next(p for p in patients if p["id"] == "pat_ramesh_001")
    check("6. profile shows Hospital Verified", r.get("latest_plan_approved") is True, f"latest {r['latest_risk_score']}% approved={r.get('latest_plan_approved')}")

# New patient baseline visit via 1-click sample -------------------------------
s, base = call("POST", "/predict/code15-multimodal", form={"patient_id": new.get("id", ""), "sample_filename": "1_normal_sinus_4096x12.npy",
               "hr": "72", "bp_sys": "122", "bp_dia": "78", "glucose": "95", "cholesterol": "176", "creatinine": "0.9"})
check("New patient baseline visit (sample ECG)", s == 200 and base["delta_summary"]["is_baseline"],
      f"{base['predictions']['overall_risk_score']}% {base['predictions']['severity_tier']}" if s == 200 else str(base)[:200])

print("\nALL JOURNEY CHECKS PASSED" if not fails else f"\n{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
