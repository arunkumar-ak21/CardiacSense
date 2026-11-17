"""
Live proof that predictions, Before-vs-Now deltas, diet/exercise plans, /summarize and /chat
are computed from the submitted report numbers (nothing static).

    python verify_dynamic_reports.py            # uses GEMINI_API_KEY from backend/.env when set
    python verify_dynamic_reports.py --offline  # forces the local clinical generator (no LLM)

Runs the real FastAPI app in-process against a temporary database (companion.db is untouched).
"""
import os
import shutil
import sys
import tempfile
import textwrap

import torch  # noqa: F401  (Windows: torch before scipy/sklearn)

OFFLINE = "--offline" in sys.argv
if OFFLINE:
    os.environ["GEMINI_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
TMP = tempfile.mkdtemp(prefix="vitalsense_dyn_")
os.environ["COMPANION_DB_PATH"] = os.path.join(TMP, "companion_dyn.db")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from genai.narrative import build_longitudinal_chat_answer  # noqa: E402

SAMPLES = Path(main.__file__).resolve().parent.parent / "sample_code15_ecgs"
W = 110


def show(title, text):
    print(f"\n  {title}:")
    for line in textwrap.wrap(str(text), W - 6):
        print("      " + line)


def sse(resp):
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            evt = json.loads(line[6:])
            if "done" in evt:
                return evt["full_text"].strip()
            if "error" in evt:
                return "ERROR: " + evt["error"]
    return ""


def submit(client, pid, date, sample, values, flags, notes):
    with open(SAMPLES / sample, "rb") as fh:
        data = {"patient_id": pid, "visit_date": date, "hospital_name": "City General Hospital",
                "adherence_notes": notes, **{k: str(v) for k, v in values.items()},
                **{c: "true" for c in flags}}
        r = client.post("/predict/code15-multimodal", data=data, files={"file": (sample, fh, "application/octet-stream")})
    assert r.status_code == 200, r.text[:400]
    return r.json()


def print_visit(label, v):
    p, plan = v["predictions"], v["diet_plan"]
    ex = plan["exercise_plan"]
    print("\n" + "─" * W + f"\n{label}  ({v['visit_date']})\n" + "─" * W)
    print(f"  model_execution_mode : {p['model_execution_mode']}")
    me = p.get("model_execution", {})
    print(f"  ResNet1d raw output  : {me.get('resnet1d_raw_output_years')} yrs  (input {me.get('input_tensor_shape')}, "
          f"features {me.get('feature_map_shape')}, forward {me.get('forward_pass_ms')} ms)")
    print(f"  Risk / tier          : {p['overall_risk_score']}% {p['severity_tier']}   heart age {p['predicted_heart_age']} yrs")
    present = [c for c, x in p["arrhythmias"].items() if x >= 0.5]
    print(f"  ECG arrhythmias ≥50% : {present or 'none'}   DSP HR {v['ecg_metadata']['dsp_measurements'].get('heart_rate')} bpm")
    print(f"  Plan generated_by    : {plan.get('generated_by')}   title: {plan.get('plan_title')}")
    print(f"  daily_targets        : {plan['daily_targets']}")
    print(f"  exercise             : {ex['allowed_activity']} · {ex['duration_mins']} min · max HR {ex['max_hr_ceiling']} bpm "
          f"({ex.get('hr_calculation', {}).get('method')})")
    print(f"  dinner               : {plan['meals']['dinner']['items']}")
    show("what_went_wrong_summary", v["delta_summary"]["what_went_wrong_summary"])
    show("why_plan_changed_summary", plan["why_plan_changed_summary"])


try:
    with TestClient(main.app) as client:
        h = client.get("/health").json()
        print("=" * W)
        print(f"/health  model_execution_mode = {h['model_execution_mode']}")
        print(f"         weights_path         = {h['code15_weights_path']}")
        print(f"         torch_available      = {h['torch_available']}   llm_provider = {h['llm_provider']}"
              f"{'   (forced offline)' if OFFLINE else ''}")
        print("=" * W)

        pat = client.post("/patients", json={"name": "Suresh Patil", "mrn": "MRN-DYN-001", "age": 56,
                                             "gender": "M", "diet_preference": "vegetarian"}).json()
        print(f"Created patient {pat['name']} ({pat['id']}), age {pat['age']}, {pat['diet_preference']}")

        a = submit(client, pat["id"], "2026-08-01", "1_normal_sinus_4096x12.npy",
                   {"hr": 72, "bp_sys": 128, "bp_dia": 82, "spo2": 98, "glucose": 110, "cholesterol": 185, "creatinine": 1.0, "hrv": 48},
                   [], "Baseline visit.")
        print_visit("VISIT A — mild numbers, normal sinus ECG", a)

        b = submit(client, pat["id"], "2026-11-14", "2_atrial_fib_lbbb_high_risk_4096x12.npy",
                   {"hr": 96, "bp_sys": 162, "bp_dia": 98, "spo2": 95, "glucose": 190, "cholesterol": 250, "creatinine": 2.0, "hrv": 26},
                   ["AF"], "Partially followed the previous plan. Ate salty restaurant food and skipped walks.")
        print_visit("VISIT B — creatinine 2.0, glucose 190, BP 162/98, AF on ECG", b)

        print("\n" + "─" * W + "\nTARGET CHANGES A → B\n" + "─" * W)
        ta, tb = a["diet_plan"]["daily_targets"], b["diet_plan"]["daily_targets"]
        for k in tb:
            mark = "  ←" if ta.get(k) != tb.get(k) else ""
            print(f"  {k:16} {str(ta.get(k)):>6} → {str(tb.get(k)):<6}{mark}")
        print(f"  {'max_hr_ceiling':16} {a['diet_plan']['exercise_plan']['max_hr_ceiling']:>6} → {b['diet_plan']['exercise_plan']['max_hr_ceiling']}")

        print("\n" + "─" * W + "\n/summarize and /chat (longitudinal context loaded from companion.db)\n" + "─" * W)
        show("/summarize (patient view)", sse(client.post("/summarize", json={"patient_id": pat["id"], "is_staff_mode": False})))
        ctx = main._load_longitudinal(pat["id"])
        for q in ("Why did my severity go up?", "Can I eat paneer pakoda tonight?", "What happened to my creatinine?"):
            msgs = [{"role": "user", "content": q}]
            ans = sse(client.post("/chat", json={"patient_id": pat["id"], "is_staff_mode": False, "messages": msgs}))
            local = build_longitudinal_chat_answer(msgs, {**ctx, "is_staff_mode": False}).strip()
            src = "local clinical generator" if ans == local else ("ERROR" if ans.startswith("ERROR") else "Gemini/LLM")
            show(f"/chat “{q}”  [answered by: {src}]", ans)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
