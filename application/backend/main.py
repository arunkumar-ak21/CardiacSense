# ── Windows rule: import torch BEFORE sklearn / scipy / joblib (OpenMP DLL conflicts) ──
import os
import logging

logger = logging.getLogger("vitalsense")
try:
    import torch  # noqa: F401
    _TORCH_OK, _TORCH_ERROR = True, None
except Exception as _exc:  # server must still start without torch
    _TORCH_OK, _TORCH_ERROR = False, f"{type(_exc).__name__}: {_exc}"
    logger.error("torch import failed: %s", _TORCH_ERROR)

import sys
import time
import uuid
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone

# Startup messages use ✓/✗; never crash on a cp1252 Windows console or redirected output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except Exception:
        pass
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from db import SessionLocal, Patient, Visit, init_db, session_scope, json_dumps
from companion_requests import (
    RequestValidationError, PatientCreate, merge_form_and_payload,
    parse_prediction_fields, validate_approval_body,
)
from genai.llm_provider import configured_provider

# backend/.env (GEMINI_API_KEY / OPENAI_API_KEY). Never overrides variables already set in the shell.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Model state ──────────────────────────────────────────────────────────────
_state: dict[str, Any] = {
    "models_loaded": False,
    "ensemble_ready": False,
    "ecg_model_ready": False,
    "gemini_connected": False,
    "medgemma_loaded": False,
    "demos_loaded": False,
    "db_ready": False,
    "code15_engine_ready": False,
    "weights_loaded": False,
    "torch_available": _TORCH_OK,
    "code15_weights_path": None,
}


def _run_startup_step(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        print(f"✗ startup step '{name}' failed: {type(e).__name__}: {e}")


def _startup_companion() -> None:
    """init_db -> sample ECG files -> seed demo patients -> Code15Engine."""

    def _db():
        init_db()
        _state["db_ready"] = True
        print("✓ companion.db ready")

    def _samples():
        from db.seed_and_samples import SAMPLES_DIR, generate_sample_code15_files
        if not (SAMPLES_DIR / "manifest.json").is_file():
            generate_sample_code15_files()
            print(f"✓ CODE-15% sample ECGs written to {SAMPLES_DIR}")

    def _seed():
        from db.seed_and_samples import seed_initial_database
        db = SessionLocal()
        try:
            print(f"✓ Demo seed: {seed_initial_database(db)}")
        finally:
            db.close()

    def _engine():
        from ml.code15_engine import get_code15_engine
        eng = get_code15_engine()
        _state["code15_engine_ready"] = eng.ready
        _state["weights_loaded"] = eng.weights_loaded
        _state["code15_weights_path"] = eng.weights_path
        print(f"✓ Code15Engine ready={eng.ready} weights_loaded={eng.weights_loaded}")

    _run_startup_step("init_db", _db)
    _run_startup_step("sample_files", _samples)
    if _state["db_ready"]:
        _run_startup_step("seed_demo_data", _seed)
    _run_startup_step("code15_engine", _engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Load ensemble ─────────────────────────────────────────────────────
    try:
        from ml.predict import load_models
        load_models()
        _state["models_loaded"] = True
        _state["ensemble_ready"] = True
        print("✓ Ensemble loaded")
    except Exception as e:
        print(f"✗ Ensemble not loaded: {e}")

    # ── Load ECG model (optional) ─────────────────────────────────────────
    ecg_path = os.path.join(os.path.dirname(__file__), "models", "ecg_resnet.pt")
    if os.path.exists(ecg_path):
        try:
            from ml.ecg_predict import load_ecg_model
            load_ecg_model(ecg_path)
            _state["ecg_model_ready"] = True
            print("✓ ECG model loaded")
        except Exception as e:
            print(f"✗ ECG model failed: {e}")
    else:
        print("⚠ ECG model not found — /predict/ecg will return 503")

    # ── Load ECG demo records ─────────────────────────────────────────────
    try:
        from ml.ecg_demos import _load_demos, DEMO_RECORDS
        ptbxl_dir = os.path.join(os.path.dirname(__file__), "data", "ptbxl")
        if os.path.exists(ptbxl_dir):
            _load_demos(ptbxl_dir)
            _state["demos_loaded"] = len(DEMO_RECORDS) > 0
            print(f"✓ ECG demos loaded: {list(DEMO_RECORDS.keys())}")
        else:
            print("⚠ PTB-XL data not found — demo endpoints will return 503")
            _state["demos_loaded"] = False
    except Exception as e:
        print(f"✗ ECG demos failed: {e}")
        _state["demos_loaded"] = False

    # ── Connect OpenAI ────────────────────────────────────────────────────
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import AsyncOpenAI  # noqa: F401
            _state["gemini_connected"] = True
            print("✓ OpenAI connected")
        except Exception as e:
            print(f"✗ OpenAI connection failed: {e}")
    else:
        print("⚠ OPENAI_API_KEY not set — GenAI endpoints will use fallback")

    # ── MedGemma (OpenAI-backed clinical assistant) ───────────────────────
    try:
        from genai.medgemma import is_available
        if is_available():
            _state["medgemma_loaded"] = True
            print("✓ MedGemma (OpenAI) ready")
        else:
            print("⚠ MedGemma unavailable — OpenAI key not set")
    except Exception as e:
        print(f"⚠ MedGemma skipped: {e}")

    # ── Longitudinal companion: DB, samples, seed, CODE-15% engine ────────
    await asyncio.to_thread(_startup_companion)

    yield


app = FastAPI(title="CardiacSense API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://localhost:5178",
        "http://localhost:5179",
        "http://localhost:5180",
        "http://localhost:5181",
        "http://localhost:3000",
        "https://vitalsense.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error helpers ─────────────────────────────────────────────────────────────
def api_error(status: int, error: str, detail: str):
    raise HTTPException(status_code=status, detail={"error": error, "detail": detail})


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    hr: float
    bp_systolic: float
    bp_diastolic: float
    spo2: float
    rr: float
    temp: float
    age: float | None = None
    sex: str | None = None
    chest_pain: str | None = None
    cholesterol: float | None = None
    fasting_bs: bool | None = None
    rest_ecg: str | None = None
    max_hr: float | None = None
    exercise_angina: bool | None = None
    st_depression: float | None = None
    st_slope: str | None = None
    num_vessels: int | None = None
    thal: str | None = None
    patient_id: str | None = "anonymous"
    session_id: str | None = None


class LongitudinalFields(BaseModel):
    patient_id: str | None = None
    patient_profile: dict | None = None
    current_visit: dict | None = None
    previous_visit: dict | None = None
    deltas: dict | None = None
    diet_plan: dict | None = None
    is_staff_mode: bool | None = None


class SummarizeRequest(LongitudinalFields):
    vitals: dict = {}
    conditions: list = []
    shap: list = []
    trend: list = []
    patient_context: dict = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(LongitudinalFields):
    messages: list[ChatMessage]
    vitals_context: dict = {}
    predictions_context: dict = {}


class HistoryStoreRequest(BaseModel):
    patient_id: str
    prediction_result: dict
    timestamp: str


class RecommendRequest(BaseModel):
    vitals: dict
    conditions: list
    shap: list
    narrative: str = ''


# ── In-memory patient history store ──────────────────────────────────────────
_patient_history: dict[str, list] = {}


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": _state["models_loaded"],
        "ensemble_ready": _state["ensemble_ready"],
        "ecg_model_ready": _state["ecg_model_ready"],
        "gemini_connected": _state["gemini_connected"],
        "medgemma_loaded": _state["medgemma_loaded"],
        "db_ready": _state["db_ready"],
        "code15_engine_ready": _state["code15_engine_ready"],
        "weights_loaded": _state["weights_loaded"],
        "code15_weights_path": _state["code15_weights_path"],
        "torch_available": _state["torch_available"],
        "model_execution_mode": (
            "PyTorch_ResNet1d_Pretrained_Model1"
            if _state["code15_engine_ready"] and _state["weights_loaded"] else "Fallback_Heuristic"
        ),
        "llm_provider": configured_provider(),
    }


# ── /predict ──────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(req: PredictRequest):
    if not _state["ensemble_ready"]:
        api_error(503, "Models not loaded", "Ensemble is still loading. Retry in a moment.")

    from ml.predict import run_prediction
    try:
        result = run_prediction(req.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"valid": False, "error": str(e), "detail": str(e)})
    except Exception as e:
        api_error(500, "Prediction failed", str(e))


# ── /predict/ecg ──────────────────────────────────────────────────────────────
@app.post("/predict/ecg")
async def predict_ecg(file: UploadFile = File(...)):
    if not _state["ecg_model_ready"]:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ECG model unavailable",
                "detail": (
                    "PTB-XL model not trained. Download the dataset from "
                    "https://physionet.org/content/ptb-xl/1.0.3/ into backend/data/ptbxl/ "
                    "and run models/train_ecg.ipynb."
                ),
            },
        )
    try:
        import numpy as np
        from ml.ecg_predict import run_ecg_prediction
        contents = await file.read()
        import io
        data = np.genfromtxt(io.StringIO(contents.decode()), delimiter=",")
        if data.ndim != 2 or data.shape[1] != 12 or data.shape[0] != 1000:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid ECG format",
                    "detail": f"Expected 12 columns × 1000 rows. Got {data.shape[1] if data.ndim==2 else '?'} columns × {data.shape[0] if data.ndim>=1 else '?'} rows.",
                },
            )
        return run_ecg_prediction(data)
    except HTTPException:
        raise
    except Exception as e:
        api_error(500, "ECG inference failed", str(e))


# ── /summarize (SSE) ──────────────────────────────────────────────────────────
def _load_longitudinal(patient_id: str) -> dict | None:
    """Latest visit, its previous visit, deltas and plan for a patient (None if no visits)."""
    with session_scope() as db:
        patient = db.get(Patient, patient_id)
        if patient is None:
            raise KeyError(patient_id)
        visits = (
            db.query(Visit).filter(Visit.patient_id == patient_id)
            .order_by(Visit.visit_date.desc(), Visit.created_at.desc()).limit(2).all()
        )
        if not visits:
            return None
        current = visits[0].to_dict(include_signals=False)
        return {
            "patient_profile": patient.to_dict(),
            "current_visit": current,
            "previous_visit": visits[1].to_dict(include_signals=False) if len(visits) > 1 else None,
            "deltas": current.get("delta_summary"),
            "diet_plan": current.get("diet_plan"),
        }


async def _resolve_longitudinal(req: LongitudinalFields) -> dict | None:
    if req.current_visit:
        return {
            "patient_profile": req.patient_profile or {},
            "current_visit": req.current_visit,
            "previous_visit": req.previous_visit,
            "deltas": req.deltas,
            "diet_plan": req.diet_plan,
            "is_staff_mode": bool(req.is_staff_mode),
        }
    if not req.patient_id:
        return None
    try:
        ctx = await asyncio.to_thread(_load_longitudinal, req.patient_id)
    except KeyError:
        api_error(404, "Patient not found", f"No patient with id '{req.patient_id}'")
    if ctx is not None:
        ctx["is_staff_mode"] = bool(req.is_staff_mode)
    return ctx


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    from genai.narrative import generate_narrative_stream

    longitudinal = await _resolve_longitudinal(req)

    async def event_stream():
        full = ""
        try:
            async for token in generate_narrative_stream(
                req.vitals, req.conditions, req.shap, req.trend, req.patient_context,
                longitudinal=longitudinal,
            ):
                full += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full_text': full})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── /chat (SSE) ───────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    from genai.narrative import generate_chat_stream

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    longitudinal = await _resolve_longitudinal(req)

    async def event_stream():
        full = ""
        try:
            async for token in generate_chat_stream(
                messages, req.vitals_context, req.predictions_context,
                longitudinal=longitudinal,
            ):
                full += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full_text': full})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── /parse-pdf ────────────────────────────────────────────────────────────────
@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            api_error(422, "File too large", "PDF must be under 10MB.")
        from parsers.pdf_parser import parse_pdf_vitals
        result = await parse_pdf_vitals(contents)
        return result
    except HTTPException:
        raise
    except Exception as e:
        api_error(500, "PDF parsing failed", str(e))


# ── /voice-to-vitals ──────────────────────────────────────────────────────────
@app.post("/voice-to-vitals")
async def voice_to_vitals(audio: UploadFile = File(...)):
    try:
        contents = await audio.read()
        if len(contents) > 5 * 1024 * 1024:
            api_error(422, "File too large", "Audio must be under 5MB.")
        mime_type = audio.content_type or "audio/webm"
        from genai.multimodal import voice_to_vitals as _voice
        result = await _voice(contents, mime_type)
        return result
    except HTTPException:
        raise
    except Exception as e:
        api_error(500, "Voice parsing failed", str(e))


# ── /recommend ────────────────────────────────────────────────────────────────
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    try:
        from genai.narrative import generate_recommendations
        return await generate_recommendations(req.vitals, req.conditions, req.shap, req.narrative)
    except Exception as e:
        api_error(500, "Recommendation failed", str(e))


# ── /patient/history (store) ──────────────────────────────────────────────────
@app.post("/patient/history")
async def store_history(req: HistoryStoreRequest):
    record_id = f"rec_{uuid.uuid4().hex[:8]}"
    entry = {
        "record_id": record_id,
        "timestamp": req.timestamp,
        "prediction_result": req.prediction_result,
        "overall_risk_score": req.prediction_result.get("overall_risk_score"),
        "overall_risk_label": req.prediction_result.get("overall_risk_label"),
        "primary_condition": (
            req.prediction_result.get("primary_condition", {}).get("name")
            if isinstance(req.prediction_result.get("primary_condition"), dict)
            else req.prediction_result.get("primary_condition")
        ),
        "vitals": {
            k: req.prediction_result.get(k)
            for k in ("hr", "bp_systolic", "bp_diastolic", "spo2", "rr", "temp")
        },
    }
    if req.patient_id not in _patient_history:
        _patient_history[req.patient_id] = []
    _patient_history[req.patient_id].append(entry)
    return {"stored": True, "record_id": record_id}


# ── /patient/{id}/history (fetch) ─────────────────────────────────────────────
@app.get("/patient/{patient_id}/history")
async def get_history(patient_id: str, limit: int = 20, offset: int = 0):
    records = _patient_history.get(patient_id, [])
    sliced = records[offset: offset + limit]
    return {"patient_id": patient_id, "records": sliced, "total": len(records)}


# ── /ecg/demos ────────────────────────────────────────────────────────────────
@app.get("/ecg/demos")
async def ecg_demos():
    if not _state.get("demos_loaded"):
        raise HTTPException(status_code=503, detail={"error": "Demo records not available", "detail": "PTB-XL dataset not found."})
    from ml.ecg_demos import get_demo_list
    return {"demos": get_demo_list()}


# ── /ecg/demo/{ecg_id}/signal ─────────────────────────────────────────────────
@app.get("/ecg/demo/{ecg_id}/signal")
async def ecg_demo_signal(ecg_id: int):
    if not _state.get("demos_loaded"):
        raise HTTPException(status_code=503, detail={"error": "Demo records not available"})
    try:
        from ml.ecg_demos import get_demo_signal
        return get_demo_signal(ecg_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "ECG record not found", "detail": f"No demo record with id {ecg_id}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Signal loading failed", "detail": str(e)})


# ── /ecg/demo/{ecg_id}/predict ────────────────────────────────────────────────
@app.get("/ecg/demo/{ecg_id}/predict")
async def ecg_demo_predict(ecg_id: int):
    if not _state.get("ecg_model_ready"):
        raise HTTPException(status_code=503, detail={"error": "ECG model not loaded"})
    if not _state.get("demos_loaded"):
        raise HTTPException(status_code=503, detail={"error": "Demo records not available"})
    try:
        from ml.ecg_demos import get_demo_prediction
        return get_demo_prediction(ecg_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "ECG record not found"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Prediction failed", "detail": str(e)})


# ═════════════════════════════════════════════════════════════════════════════
# Longitudinal companion endpoints (companion.db + CODE-15% engine)
# ═════════════════════════════════════════════════════════════════════════════
def _require_db() -> None:
    if not _state["db_ready"]:
        api_error(503, "Database unavailable", "The patient database (companion.db) is not ready.")


def _validation_422(exc: RequestValidationError):
    raise HTTPException(status_code=422, detail={
        "error": "Invalid request", "detail": str(exc), "fields": exc.fields,
    })


# ── GET /patients ─────────────────────────────────────────────────────────────
@app.get("/patients")
async def list_patients():
    _require_db()

    def work():
        with session_scope() as db:
            patients = db.query(Patient).order_by(Patient.created_at.desc(), Patient.id.asc()).all()
            out = []
            for p in patients:
                visits = sorted(p.visits, key=lambda v: (v.visit_date, v.created_at), reverse=True)
                latest = visits[0].predictions if visits else {}
                out.append({
                    **p.to_dict(),
                    "visit_count": len(visits),
                    "latest_visit_date": visits[0].visit_date if visits else None,
                    "latest_risk_score": latest.get("overall_risk_score"),
                    "latest_severity_tier": latest.get("severity_tier"),
                    "latest_plan_approved": bool(visits[0].approved_by_staff) if visits else False,
                })
            return out

    return await asyncio.to_thread(work)


# ── POST /patients ────────────────────────────────────────────────────────────
@app.post("/patients", status_code=201)
async def create_patient(request: Request):
    _require_db()
    try:
        body = PatientCreate.model_validate(await request.json())
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={
            "error": "Invalid patient", "detail": "Patient body failed validation.",
            "fields": [{"field": ".".join(str(p) for p in err["loc"]), "message": err["msg"]} for err in e.errors()],
        })
    except ValueError:
        api_error(422, "Invalid patient", "Body must be a JSON object.")

    def work():
        with session_scope() as db:
            if db.query(Patient).filter(Patient.mrn == body.mrn).first():
                return None
            p = Patient(
                id=f"pat_{uuid.uuid4().hex[:12]}", name=body.name, mrn=body.mrn, age=body.age,
                gender=body.gender, diet_preference=body.diet_preference,
                known_conditions=json_dumps(body.known_conditions),
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            db.add(p)
            db.flush()
            return {**p.to_dict(), "visit_count": 0, "latest_visit_date": None,
                    "latest_risk_score": None, "latest_severity_tier": None}

    created = await asyncio.to_thread(work)
    if created is None:
        api_error(409, "Duplicate MRN", f"A patient with MRN '{body.mrn}' already exists.")
    return created


# ── GET /patients/{patient_id}/visits ─────────────────────────────────────────
@app.get("/patients/{patient_id}/visits")
async def patient_visits(patient_id: str, include_signals: bool = True):
    _require_db()

    def work():
        with session_scope() as db:
            p = db.get(Patient, patient_id)
            if p is None:
                return None
            visits = (
                db.query(Visit).filter(Visit.patient_id == patient_id)
                .order_by(Visit.visit_date.desc(), Visit.created_at.desc()).all()
            )
            data = [v.to_dict(include_signals=include_signals) for v in visits]
            return {"patient": p.to_dict(), "visits": data, "total": len(data)}

    result = await asyncio.to_thread(work)
    if result is None:
        api_error(404, "Patient not found", f"No patient with id '{patient_id}'")
    return result


# ── GET /ecg/code15-samples ───────────────────────────────────────────────────
@app.get("/ecg/code15-samples")
async def code15_samples():
    from ml.code15_engine import SAMPLES_DIR, list_sample_files

    names = list_sample_files(SAMPLES_DIR)
    if not names:
        return {"samples": [], "message": "No sample files have been generated."}
    manifest: dict[str, dict] = {}
    try:
        for entry in json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8")):
            for fname in entry.get("files", []):
                manifest[fname] = entry
    except Exception:
        manifest = {}
    samples = []
    for name in names:
        m = manifest.get(name, {})
        samples.append({
            "filename": name,
            "format": name.rsplit(".", 1)[-1].lower(),
            "size_bytes": (SAMPLES_DIR / name).stat().st_size,
            "label": m.get("label"),
            "description": m.get("description"),
            "expected_code15_flags": m.get("expected_code15_flags"),
        })
    return {"samples": samples, "total": len(samples)}


# ── 12-lead preview (no DB write) for the visit form ──────────────────────────
def _preview_payload(canonical) -> dict:
    from ml.code15_engine import PREVIEW_SAMPLE_RATE, build_leads_preview, get_code15_engine

    measured = get_code15_engine().measurer.measure(canonical)
    return {
        "leads_preview": build_leads_preview(canonical),
        "preview_sample_rate": PREVIEW_SAMPLE_RATE,
        "duration_s": 10.24,
        "dsp_measurements": dict(measured.measurements),
    }


@app.get("/ecg/code15-samples/{filename}/preview")
async def code15_sample_preview(filename: str):
    from ml.code15_engine import ECGInputError, SAMPLES_DIR, get_code15_engine, resolve_sample_path

    path = resolve_sample_path(filename, SAMPLES_DIR)
    if path is None:
        api_error(404, "Sample ECG not found", f"'{filename}' is not an available sample file.")

    def work():
        canonical, _ = get_code15_engine().canonical_from_sample(path)
        return _preview_payload(canonical)

    try:
        return await asyncio.to_thread(work)
    except ECGInputError as e:
        raise HTTPException(status_code=e.status_code, detail=e.to_detail())


@app.post("/ecg/code15-preview")
async def code15_upload_preview(file: UploadFile = File(...), sample_rate: float = 400.0):
    from ml.code15_engine import ECGInputError, get_code15_engine

    data = await file.read()

    def work():
        canonical, _ = get_code15_engine().canonical_from_upload(file.filename or "", data, sample_rate)
        return _preview_payload(canonical)

    try:
        return await asyncio.to_thread(work)
    except ECGInputError as e:
        raise HTTPException(status_code=e.status_code, detail=e.to_detail())


# ── POST /predict/code15-multimodal ───────────────────────────────────────────
DIET_PREFERENCE_VALUES = ("vegetarian", "non_vegetarian", "vegan", "diabetic", "dash")


def _run_prediction_transaction(fields, source: dict, diet_preference: str | None = None) -> dict:
    """Engine -> previous visit -> deltas -> diet plan -> save Visit, all in one transaction."""
    from ml.code15_engine import ECGInputError, EngineInput, get_code15_engine, synthesize_from_flags
    from genai.diet_planner import compute_visit_deltas, generate_diet_exercise_plan

    stage = "load_patient"
    db = SessionLocal()
    try:
        patient = db.get(Patient, fields.patient_id)
        if patient is None:
            return {"_status": 404, "error": "Patient not found", "detail": f"No patient with id '{fields.patient_id}'"}
        if diet_preference and diet_preference != patient.diet_preference:
            patient.diet_preference = diet_preference  # saved with the visit in the same commit
        patient_dict = patient.to_dict()

        stage = "preprocess"
        engine = get_code15_engine()
        try:
            if source["kind"] == "upload":
                canonical, fmt = engine.canonical_from_upload(source["filename"], source["data"], fields.sample_rate)
            elif source["kind"] == "sample":
                canonical, fmt = engine.canonical_from_sample(source["path"])
            else:
                canonical, fmt = synthesize_from_flags(fields.hr, fields.flags), "synthetic"
        except ECGInputError as e:
            return {"_status": e.status_code, **e.to_detail()}

        stage = "inference"
        values = {**fields.vitals, **fields.blood}
        result = engine.predict(EngineInput(
            patient=patient_dict, values=values, flags=fields.flags, canonical=canonical,
            source=source["kind"], filename=source.get("filename"), fmt=fmt,
        ))

        stage = "previous_visit"
        prev = (
            db.query(Visit).filter(Visit.patient_id == fields.patient_id, Visit.visit_date <= fields.visit_date)
            .order_by(Visit.visit_date.desc(), Visit.created_at.desc()).first()
        )
        previous_dict = prev.to_dict(include_signals=False) if prev else None

        stage = "deltas"
        current_dict = {
            "id": None, "visit_date": fields.visit_date, "adherence_notes": fields.adherence_notes,
            "vitals": fields.vitals, "blood": fields.blood, "code15_flags": result.code15_flags,
            "ecg_metadata": {k: v for k, v in result.ecg_metadata.items() if k != "leads_preview"},
            "predictions": result.predictions,
        }
        deltas = compute_visit_deltas(current_dict, previous_dict)

        stage = "diet_plan"
        plan = generate_diet_exercise_plan(patient_dict, current_dict, previous_dict, deltas)

        stage = "save"
        visit = Visit(
            id=f"vis_{uuid.uuid4().hex[:12]}",
            patient_id=fields.patient_id,
            visit_date=fields.visit_date,
            hospital_name=fields.hospital_name or "General Hospital",
            adherence_notes=fields.adherence_notes,
            vitals_json=json_dumps(fields.vitals),
            blood_json=json_dumps(fields.blood),
            code15_flags_json=json_dumps(result.code15_flags),
            ecg_metadata_json=json_dumps(result.ecg_metadata),
            predictions_json=json_dumps(result.predictions),
            shap_json=json_dumps(result.shap),
            heatmap_json=json_dumps(result.heatmap),
            delta_summary_json=json_dumps(deltas),
            diet_plan_json=json_dumps(plan),
            approved_by_staff=False,
            staff_notes=None,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.add(visit)
        db.commit()
        return visit.to_dict(include_signals=True)
    except Exception as e:
        db.rollback()
        logger.exception("code15 prediction failed at stage %s", stage)
        return {"_status": 500, "error": "Prediction failed", "detail": f"{stage} failed: {type(e).__name__}: {e}", "stage": stage}
    finally:
        db.close()


@app.post("/predict/code15-multimodal")
async def predict_code15_multimodal(request: Request):
    _require_db()
    from ml.code15_engine import SAMPLES_DIR, list_sample_files, resolve_sample_path

    upload = None
    ctype = (request.headers.get("content-type") or "").lower()
    try:
        if ctype.startswith("application/json"):
            try:
                body = await request.json()
            except ValueError:
                raise RequestValidationError([{"field": "payload", "message": "body is not valid JSON"}])
            if not isinstance(body, dict):
                raise RequestValidationError([{"field": "payload", "message": "body must be a JSON object"}])
            raw = body
        elif ctype.startswith("multipart/form-data") or ctype.startswith("application/x-www-form-urlencoded"):
            form = await request.form()
            f = form.get("file")
            if f is not None and not isinstance(f, str) and getattr(f, "filename", ""):
                upload = f
            payload = form.get("payload")
            raw = merge_form_and_payload(
                [(k, v) for k, v in form.multi_items() if isinstance(v, str)],
                payload if isinstance(payload, str) else None,
            )
        else:
            api_error(415, "Unsupported content type", "Send multipart/form-data or application/json.")
        fields = parse_prediction_fields(raw, date.today())
        diet_pref = str(raw.get("diet_preference") or "").strip().lower() or None
        if diet_pref is not None and diet_pref not in DIET_PREFERENCE_VALUES:
            raise RequestValidationError([{
                "field": "diet_preference", "message": f"must be one of {', '.join(DIET_PREFERENCE_VALUES)}",
            }])
    except RequestValidationError as e:
        _validation_422(e)

    if upload is not None:
        data = await upload.read()
        source = {"kind": "upload", "filename": os.path.basename(upload.filename), "data": data}
    elif fields.sample_filename:
        path = resolve_sample_path(fields.sample_filename, SAMPLES_DIR)
        if path is None:
            raise HTTPException(status_code=404, detail={
                "error": "Sample ECG not found",
                "detail": f"'{fields.sample_filename}' is not an available sample file.",
                "available_samples": list_sample_files(SAMPLES_DIR),
            })
        source = {"kind": "sample", "filename": path.name, "path": path}
    else:
        source = {"kind": "synthetic"}

    result = await asyncio.to_thread(_run_prediction_transaction, fields, source, diet_pref)
    status = result.pop("_status", 200)
    if status != 200:
        raise HTTPException(status_code=status, detail=result)
    return result


# ── POST /visits/{visit_id}/approve ───────────────────────────────────────────
@app.post("/visits/{visit_id}/approve")
async def approve_visit(visit_id: str, request: Request):
    _require_db()
    try:
        try:
            body = await request.json()
        except ValueError:
            raise RequestValidationError([{"field": "body", "message": "body is not valid JSON"}])
        approval = validate_approval_body(body)
    except RequestValidationError as e:
        _validation_422(e)

    def work():
        with session_scope() as db:
            visit = db.get(Visit, visit_id)
            if visit is None:
                return None
            plan = dict(approval.diet_plan) if approval.diet_plan is not None else visit.diet_plan
            if approval.diet_plan is not None:
                plan["generated_by"] = "staff_modified"
            plan["staff_review"] = {
                "approved_by_staff": approval.approved_by_staff,
                "staff_notes": approval.staff_notes,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
            visit.diet_plan_json = json_dumps(plan)
            visit.approved_by_staff = approval.approved_by_staff
            if approval.staff_notes_provided:
                visit.staff_notes = approval.staff_notes
            db.flush()
            return visit.to_dict()

    result = await asyncio.to_thread(work)
    if result is None:
        api_error(404, "Visit not found", f"No visit with id '{visit_id}'")
    return result
