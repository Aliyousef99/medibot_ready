from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict
import hashlib

import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.models.symptom_event import SymptomEvent

logger = logging.getLogger("medibot")


def save_symptom_event(db: Session, user_id: str, text: str, result: Dict) -> SymptomEvent:
    """Insert a SymptomEvent with idempotency window of 10 seconds.

    If an event for the same user with the same raw_text exists within the last
    10 seconds, reuse it instead of inserting a duplicate row.
    """
    clean_text = (text or "").strip()
    cutoff = datetime.utcnow() - timedelta(seconds=10)
    hash_key = hashlib.sha256(f"{user_id}|{clean_text}".encode("utf-8")).hexdigest()[:16]

    # Reuse recent identical event if present
    existing = (
        db.query(SymptomEvent)
        .filter(
            SymptomEvent.user_id == user_id,
            SymptomEvent.raw_text == clean_text,
            SymptomEvent.created_at >= cutoff,
        )
        .order_by(desc(SymptomEvent.created_at))
        .first()
    )
    if existing:
        try:
            logger.info({
                "function": "save_symptom_event",
                "status": "reused",
                "event_id": str(existing.id),
                "key": hash_key,
            })
        except Exception:
            pass
        return existing

    ev = SymptomEvent(user_id=user_id, raw_text=clean_text, result_json=result)
    db.add(ev)
    db.commit()
    db.refresh(ev)
    try:
        logger.info({
            "function": "save_symptom_event",
            "status": "inserted",
            "event_id": str(ev.id),
            "key": hash_key,
        })
    except Exception:
        pass
    return ev


__all__ = ["save_symptom_event"]
