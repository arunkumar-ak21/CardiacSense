"""Request parsing and validation for the companion endpoints.

Pure Python: the standard library plus pydantic v2 (for ``PatientCreate``).
No NumPy, SciPy or torch, so property tests run fast and ``main.py`` can
import this module at the top level without breaking the torch-first import
order.

Prediction parsing covers Req 5.3 and Req 18.1-18.4, 18.12 and 18.13.
Patient validation covers Req 16.3-16.6; approval validation covers
Req 19.5 and 19.7.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Constants ───────────────────────────────────────────────────────────────

# Inclusive ranges for numeric prediction inputs (Req 18.4, the single source).
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "hr": (20, 250),
    "bp_sys": (60, 260),
    "bp_dia": (30, 160),
    "spo2": (50, 100),
    "rr": (4, 60),
    "temp": (30, 43),
    "glucose": (20, 600),
    "cholesterol": (50, 500),
    "creatinine": (0.1, 15),
    "hrv": (0, 300),
    "sample_rate": (100, 2000),
}

# CODE15_Flag values parsed as true (Req 18.3), compared after strip + lower.
TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})

CODE15_FLAG_KEYS: tuple[str, ...] = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")
VITAL_KEYS: tuple[str, ...] = ("hr", "bp_sys", "bp_dia", "spo2", "rr", "temp")
BLOOD_KEYS: tuple[str, ...] = ("glucose", "cholesterol", "creatinine", "hrv")

ACCEPTED_SAMPLE_EXTENSIONS: tuple[str, ...] = (".npy", ".csv", ".hdf5", ".h5")
DEFAULT_SAMPLE_RATE = 400.0
TEXT_LIMITS: dict[str, int] = {"hospital_name": 200, "adherence_notes": 2000}

# Form keys that never become prediction fields (Req 18.1).
_DROPPED_FORM_KEYS = frozenset({"file", "payload"})
_VISIT_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


# ── Errors ──────────────────────────────────────────────────────────────────


class RequestValidationError(Exception):
    """A 422 validation failure listing every malformed field.

    ``fields`` is a list of ``{"field": name, "message": text}`` entries,
    at most one entry per field name.
    """

    status_code = 422

    def __init__(self, fields: list[dict[str, str]]):
        self.fields = list(fields)
        names = ", ".join(entry["field"] for entry in self.fields)
        super().__init__(f"Invalid request fields: {names}")


# ── Small helpers ───────────────────────────────────────────────────────────


def parse_flag(value: Any) -> bool:
    """Return True iff ``str(value).strip().lower()`` is in TRUE_STRINGS (Req 18.3)."""
    return str(value).strip().lower() in TRUE_STRINGS


def _fmt_number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _range_text(name: str) -> str:
    lo, hi = NUMERIC_RANGES[name]
    return f"{_fmt_number(lo)}-{_fmt_number(hi)}"


def merge_form_and_payload(
    form_items: Mapping[str, Any] | Iterable[tuple[str, Any]],
    payload_text: str | bytes | None,
) -> dict[str, Any]:
    """Merge a JSON ``payload`` object with individual form fields.

    Form fields win over payload keys (Req 18.1). The ``file`` and ``payload``
    keys are dropped from the result. ``form_items`` may be a mapping or an
    iterable of ``(key, value)`` pairs such as ``form.multi_items()``; for a
    repeated key the last value wins. ``payload_text=None`` means no payload.
    Any other text that is not a JSON object raises RequestValidationError
    naming ``payload`` (Req 18.12).
    """
    payload: dict[str, Any] = {}
    if payload_text is not None:
        if not isinstance(payload_text, (str, bytes, bytearray)):
            raise RequestValidationError(
                [{"field": "payload", "message": "payload must be a JSON object"}]
            )
        try:
            parsed = json.loads(payload_text)
        except (ValueError, UnicodeDecodeError):
            raise RequestValidationError(
                [{"field": "payload", "message": "payload is not valid JSON"}]
            ) from None
        if not isinstance(parsed, dict):
            raise RequestValidationError(
                [{"field": "payload", "message": "payload must be a JSON object"}]
            )
        payload = parsed

    items = form_items.items() if isinstance(form_items, Mapping) else form_items
    form = {k: v for k, v in items if k not in _DROPPED_FORM_KEYS}
    merged = {k: v for k, v in payload.items() if k not in _DROPPED_FORM_KEYS}
    merged.update(form)
    return merged


def validate_sample_filename(name: str) -> bool:
    """Return True iff ``name`` is a safe base name with an accepted extension (Req 5.3).

    Rejects path separators (``/`` and ``\\``), ``..`` segments, drive or
    stream markers (``:``), NUL bytes, an empty stem and any extension other
    than ``.npy``, ``.csv``, ``.hdf5`` or ``.h5`` (case-insensitive).
    Existence in the Sample_Directory is checked separately by the caller.
    """
    if not isinstance(name, str) or not name:
        return False
    if any(ch in name for ch in ("/", "\\", ":", "\x00")):
        return False
    if name in (".", ".."):
        return False
    lower = name.lower()
    for ext in ACCEPTED_SAMPLE_EXTENSIONS:
        if lower.endswith(ext):
            return len(name) > len(ext)
    return False


# ── Prediction fields ───────────────────────────────────────────────────────


def _no_flags() -> dict[str, bool]:
    return {code: False for code in CODE15_FLAG_KEYS}


@dataclass(frozen=True)
class PredictionFields:
    """Validated fields of one ``POST /predict/code15-multimodal`` request (Req 18.2)."""

    patient_id: str
    visit_date: str  # ISO YYYY-MM-DD
    sample_filename: str | None = None  # trimmed; None when absent or blank
    sample_rate: float = DEFAULT_SAMPLE_RATE
    hospital_name: str | None = None
    adherence_notes: str | None = None
    hr: float | None = None
    bp_sys: float | None = None
    bp_dia: float | None = None
    spo2: float | None = None
    rr: float | None = None
    temp: float | None = None
    glucose: float | None = None
    cholesterol: float | None = None
    creatinine: float | None = None
    hrv: float | None = None
    flags: dict[str, bool] = field(default_factory=_no_flags)

    @property
    def vitals(self) -> dict[str, float | None]:
        """Vitals keyed by name; missing values are None (stored as null, Req 18.13)."""
        return {k: getattr(self, k) for k in VITAL_KEYS}

    @property
    def blood(self) -> dict[str, float | None]:
        """Blood values keyed by name; missing values are None (Req 18.13)."""
        return {k: getattr(self, k) for k in BLOOD_KEYS}


class _Missing:
    pass


_MISSING = _Missing()


def _parse_number(value: Any) -> float | _Missing | str:
    """Parse one numeric input.

    Returns a finite float, ``_MISSING`` for None / empty / whitespace, or an
    error message string for booleans, NaN, infinities and non-numeric values.
    """
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        return "must be a number, not a boolean"
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return "must be a finite number"
    elif isinstance(value, (str, bytes, bytearray)):
        text = value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else value
        text = text.strip()
        if not text:
            return _MISSING
        try:
            number = float(text)
        except (ValueError, OverflowError):
            return "must be a number"
    else:
        return "must be a number"
    if not math.isfinite(number):
        return "must be a finite number"
    return number


def _parse_text(
    raw: Mapping[str, Any], name: str, errors: dict[str, str]
) -> str | None:
    """Return the stripped text value, or None when absent or blank."""
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        errors[name] = f"{name} must be a string"
        return None
    text = value.strip()
    if not text:
        return None
    limit = TEXT_LIMITS.get(name)
    if limit is not None and len(text) > limit:
        errors[name] = f"{name} must be at most {limit} characters (got {len(text)})"
        return None
    return text


def parse_prediction_fields(raw: Mapping[str, Any], today: date | None = None) -> PredictionFields:
    """Validate a merged prediction request and return PredictionFields.

    Collects every problem before raising one RequestValidationError
    (Req 18.4, 18.12): blank ``patient_id``, a ``visit_date`` that is not a
    real ``YYYY-MM-DD`` date, over-long text, non-numeric or out-of-range
    numbers, and ``bp_dia >= bp_sys`` when both are present. Empty or absent
    numbers are treated as missing (Req 18.13). ``visit_date`` defaults to
    ``today`` (the server's local date when None).
    """
    if not isinstance(raw, Mapping):
        raise RequestValidationError(
            [{"field": "payload", "message": "request body must be a JSON object"}]
        )

    errors: dict[str, str] = {}

    # patient_id (required, non-blank)
    patient_id = ""
    pid_value = raw.get("patient_id")
    if pid_value is None:
        errors["patient_id"] = "patient_id is required"
    elif not isinstance(pid_value, str):
        errors["patient_id"] = "patient_id must be a non-empty string"
    elif not pid_value.strip():
        errors["patient_id"] = "patient_id must not be empty"
    else:
        patient_id = pid_value.strip()

    # visit_date (YYYY-MM-DD calendar date, default today)
    visit_date = ""
    vd_value = raw.get("visit_date")
    if vd_value is None or (isinstance(vd_value, str) and not vd_value.strip()):
        visit_date = (today or date.today()).isoformat()
    elif not isinstance(vd_value, str):
        errors["visit_date"] = "visit_date must be a date string in YYYY-MM-DD format"
    else:
        text = vd_value.strip()
        parsed: date | None = None
        if _VISIT_DATE_RE.fullmatch(text):
            try:
                parsed = date.fromisoformat(text)
            except ValueError:
                parsed = None
        if parsed is None:
            errors["visit_date"] = (
                f"visit_date must be a valid calendar date in YYYY-MM-DD format (got {text!r})"
            )
        else:
            visit_date = parsed.isoformat()

    # Text fields
    sample_filename = _parse_text(raw, "sample_filename", errors)
    hospital_name = _parse_text(raw, "hospital_name", errors)
    adherence_notes = _parse_text(raw, "adherence_notes", errors)

    # Numbers: parse, then inclusive range check
    numbers: dict[str, float | None] = {}
    finite: dict[str, float] = {}
    for name, (lo, hi) in NUMERIC_RANGES.items():
        result = _parse_number(raw.get(name))
        if isinstance(result, _Missing):
            numbers[name] = None
            continue
        if isinstance(result, str):
            errors[name] = f"{name} {result} (allowed range {_range_text(name)}, inclusive)"
            numbers[name] = None
            continue
        finite[name] = result
        numbers[name] = result
        if not lo <= result <= hi:
            errors[name] = (
                f"{name} must be between {_fmt_number(lo)} and {_fmt_number(hi)} inclusive "
                f"(got {_fmt_number(result)})"
            )

    # Cross-field rule: bp_dia < bp_sys when both are present
    if "bp_sys" in finite and "bp_dia" in finite and finite["bp_dia"] >= finite["bp_sys"]:
        rule = (
            f"bp_dia must be less than bp_sys (got bp_dia {_fmt_number(finite['bp_dia'])}, "
            f"bp_sys {_fmt_number(finite['bp_sys'])})"
        )
        errors["bp_dia"] = f"{errors['bp_dia']}; {rule}" if "bp_dia" in errors else rule

    if errors:
        raise RequestValidationError(
            [{"field": name, "message": message} for name, message in errors.items()]
        )

    sample_rate = numbers.pop("sample_rate")
    flags = {code: parse_flag(raw[code]) if code in raw else False for code in CODE15_FLAG_KEYS}

    return PredictionFields(
        patient_id=patient_id,
        visit_date=visit_date,
        sample_filename=sample_filename,
        sample_rate=DEFAULT_SAMPLE_RATE if sample_rate is None else sample_rate,
        hospital_name=hospital_name,
        adherence_notes=adherence_notes,
        flags=flags,
        **numbers,
    )


# ── Patient creation (Req 16.3-16.6) ───────────────────────────────────────

DIET_PREFERENCES: tuple[str, ...] = ("vegetarian", "non_vegetarian", "vegan", "diabetic", "dash")
DEFAULT_DIET_PREFERENCE = "non_vegetarian"
MAX_KNOWN_CONDITIONS = 20

DietPreference = Literal["vegetarian", "non_vegetarian", "vegan", "diabetic", "dash"]
ConditionText = Annotated[str, Field(min_length=1, max_length=100)]


class PatientCreate(BaseModel):
    """Body of ``POST /patients``.

    Unknown keys (including a client ``id`` or ``created_at``) are ignored
    (Req 16.3). ``name``, ``mrn`` and each ``known_conditions`` item are
    stripped before their length checks (Req 16.4, 16.5). ``age`` is a strict
    integer, so booleans, floats and numeric strings are rejected.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=120)
    mrn: str = Field(min_length=1, max_length=50)
    age: Annotated[int, Field(strict=True, ge=0, le=120)]
    gender: Literal["M", "F", "Other"]
    diet_preference: DietPreference = DEFAULT_DIET_PREFERENCE
    known_conditions: list[ConditionText] = Field(
        default_factory=list, max_length=MAX_KNOWN_CONDITIONS
    )

    @field_validator("name", "mrn", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("known_conditions", mode="before")
    @classmethod
    def _strip_conditions(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value


# ── Staff approval (Req 19.5, 19.7) ────────────────────────────────────────

STAFF_NOTES_MAX = 2000
DIET_PLAN_REQUIRED_KEYS: tuple[str, ...] = ("daily_targets", "meals", "exercise_plan")
EXERCISE_POSITIVE_KEYS: tuple[str, ...] = ("max_hr_ceiling", "duration_mins")


@dataclass(frozen=True)
class ApprovalRequest:
    """Validated body of ``POST /visits/{visit_id}/approve``.

    ``staff_notes`` is None when omitted (``staff_notes_provided`` is then
    False). ``diet_plan`` is a deep copy of the submitted object, or None when
    omitted (Req 19.9).
    """

    approved_by_staff: bool
    staff_notes: str | None = None
    staff_notes_provided: bool = False
    diet_plan: dict[str, Any] | None = None


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_finite(value: Any) -> bool:
    if not _is_json_number(value):
        return False
    if isinstance(value, int):
        return value > 0  # Python ints are always finite
    return math.isfinite(value) and value > 0


def _check_numbers_positive(value: Any, path: str, errors: dict[str, str]) -> None:
    """Record an error for every JSON number under ``value`` that is not positive and finite."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            _check_numbers_positive(child, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_numbers_positive(child, f"{path}[{index}]", errors)
    elif _is_json_number(value) and not _is_positive_finite(value):
        errors[path] = f"{path} must be a positive finite number (got {value!r})"


def _validate_diet_plan(plan: Any, errors: dict[str, str]) -> None:
    if not isinstance(plan, Mapping):
        errors["diet_plan"] = "diet_plan must be a JSON object"
        return
    missing = [key for key in DIET_PLAN_REQUIRED_KEYS if key not in plan]
    if missing:
        errors["diet_plan"] = "diet_plan is missing required keys: " + ", ".join(missing)

    if "daily_targets" in plan:
        targets = plan["daily_targets"]
        if not isinstance(targets, Mapping):
            errors["diet_plan.daily_targets"] = "diet_plan.daily_targets must be a JSON object"
        else:
            _check_numbers_positive(targets, "diet_plan.daily_targets", errors)

    if "exercise_plan" in plan:
        exercise = plan["exercise_plan"]
        if not isinstance(exercise, Mapping):
            errors["diet_plan.exercise_plan"] = "diet_plan.exercise_plan must be a JSON object"
        else:
            for key in EXERCISE_POSITIVE_KEYS:
                if key in exercise and not _is_positive_finite(exercise[key]):
                    name = f"diet_plan.exercise_plan.{key}"
                    errors[name] = f"{name} must be a positive finite number (got {exercise[key]!r})"


def validate_approval_body(body: Any) -> ApprovalRequest:
    """Validate a parsed approval JSON body and return an ApprovalRequest.

    Collects every problem before raising one RequestValidationError:

    - ``approved_by_staff`` is required and must be a JSON boolean (Req 19.7).
    - ``staff_notes``, when present, must be a string of 0-2000 characters;
      an explicit null counts as present and is rejected (Req 19.7).
    - ``diet_plan``, when present, must be an object containing
      ``daily_targets``, ``meals`` and ``exercise_plan``. Every JSON number in
      ``daily_targets`` (searched recursively) and
      ``exercise_plan.max_hr_ceiling`` / ``exercise_plan.duration_mins`` (when
      present) must be positive and finite (Req 19.5).

    Error entries name the failing key, e.g. ``diet_plan.daily_targets.sodium_mg``.
    """
    if not isinstance(body, Mapping):
        raise RequestValidationError(
            [{"field": "body", "message": "request body must be a JSON object"}]
        )

    errors: dict[str, str] = {}

    approved = body.get("approved_by_staff")
    if "approved_by_staff" not in body:
        errors["approved_by_staff"] = "approved_by_staff is required"
    elif not isinstance(approved, bool):
        errors["approved_by_staff"] = "approved_by_staff must be a JSON boolean (true or false)"

    notes_provided = "staff_notes" in body
    notes = body.get("staff_notes")
    if notes_provided:
        if not isinstance(notes, str):
            errors["staff_notes"] = "staff_notes must be a string"
        elif len(notes) > STAFF_NOTES_MAX:
            errors["staff_notes"] = (
                f"staff_notes must be at most {STAFF_NOTES_MAX} characters (got {len(notes)})"
            )

    plan = body.get("diet_plan")
    if "diet_plan" in body:
        _validate_diet_plan(plan, errors)

    if errors:
        raise RequestValidationError(
            [{"field": name, "message": message} for name, message in errors.items()]
        )

    return ApprovalRequest(
        approved_by_staff=approved,
        staff_notes=notes if notes_provided else None,
        staff_notes_provided=notes_provided,
        diet_plan=copy.deepcopy(dict(plan)) if "diet_plan" in body else None,
    )
