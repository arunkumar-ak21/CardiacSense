"""
ORM models for the longitudinal companion database.

JSON payloads are stored in ``Text`` columns (portable, human-inspectable in any
SQLite browser). ``to_dict()`` parses them back into Python objects and tolerates
NULL or malformed values by returning a sensible empty default.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

LEAD_NAMES: list[str] = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def json_dumps(value: Any) -> str | None:
    """Serialize to compact JSON; ``None`` stays ``None`` (SQL NULL)."""
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def json_loads(text: str | None, default: Any) -> Any:
    """Parse JSON text, returning ``default`` for NULL, empty or invalid input."""
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    mrn: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)  # "M" | "F" | "Other"
    diet_preference: Mapped[str] = mapped_column(String(30), nullable=False, default="non_vegetarian")
    known_conditions: Mapped[str | None] = mapped_column(Text, nullable=True, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    visits: Mapped[list["Visit"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Visit.visit_date",
    )

    def to_dict(self, include_visits: bool = False, include_signals: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "mrn": self.mrn,
            "age": self.age,
            "gender": self.gender,
            "diet_preference": self.diet_preference,
            "known_conditions": json_loads(self.known_conditions, []),
            "created_at": _iso(self.created_at),
        }
        if include_visits:
            data["visits"] = [v.to_dict(include_signals=include_signals) for v in self.visits]
        return data

    def __repr__(self) -> str:
        return f"<Patient id={self.id!r} name={self.name!r} mrn={self.mrn!r}>"


class Visit(Base):
    __tablename__ = "visits"
    __table_args__ = (Index("ix_visits_patient_date", "patient_id", "visit_date"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    visit_date: Mapped[str] = mapped_column(String(10), nullable=False)  # ISO "YYYY-MM-DD"
    hospital_name: Mapped[str] = mapped_column(String(160), nullable=False, default="General Hospital")
    adherence_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    vitals_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    blood_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    code15_flags_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ecg_metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    predictions_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    shap_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    heatmap_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    delta_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    diet_plan_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    approved_by_staff: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    staff_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    patient: Mapped[Patient] = relationship(back_populates="visits")

    # ── Parsed accessors ────────────────────────────────────────────────────
    @property
    def vitals(self) -> dict[str, Any]:
        return json_loads(self.vitals_json, {})

    @property
    def blood(self) -> dict[str, Any]:
        return json_loads(self.blood_json, {})

    @property
    def code15_flags(self) -> dict[str, Any]:
        return json_loads(self.code15_flags_json, {})

    @property
    def ecg_metadata(self) -> dict[str, Any]:
        return json_loads(self.ecg_metadata_json, {})

    @property
    def predictions(self) -> dict[str, Any]:
        return json_loads(self.predictions_json, {})

    @property
    def shap(self) -> list[dict[str, Any]]:
        return json_loads(self.shap_json, [])

    @property
    def heatmap(self) -> list[list[float]] | None:
        return json_loads(self.heatmap_json, None)

    @property
    def delta_summary(self) -> dict[str, Any] | None:
        return json_loads(self.delta_summary_json, None)

    @property
    def diet_plan(self) -> dict[str, Any]:
        return json_loads(self.diet_plan_json, {})

    def to_dict(self, include_signals: bool = True) -> dict[str, Any]:
        """
        Serialize the visit with all JSON columns parsed.

        ``include_signals=False`` drops the heavy fields (12-lead waveform preview and
        saliency heatmap) for compact list/timeline responses.
        """
        ecg_metadata = self.ecg_metadata
        if not include_signals and ecg_metadata:
            ecg_metadata = {k: v for k, v in ecg_metadata.items() if k != "leads_preview"}

        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "visit_date": self.visit_date,
            "hospital_name": self.hospital_name,
            "adherence_notes": self.adherence_notes,
            "vitals": self.vitals,
            "blood": self.blood,
            "code15_flags": self.code15_flags,
            "ecg_metadata": ecg_metadata,
            "predictions": self.predictions,
            "shap": self.shap,
            "heatmap": self.heatmap if include_signals else None,
            "delta_summary": self.delta_summary,
            "diet_plan": self.diet_plan,
            "approved_by_staff": bool(self.approved_by_staff),
            "staff_notes": self.staff_notes,
            "created_at": _iso(self.created_at),
        }

    def __repr__(self) -> str:
        return f"<Visit id={self.id!r} patient_id={self.patient_id!r} date={self.visit_date!r}>"
