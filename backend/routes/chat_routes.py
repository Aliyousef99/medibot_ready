"""Legacy chat endpoints (start/send/history).

These are kept for reference but are no longer mounted by default; the
primary chat entrypoint lives in backend/app.py:/api/chat.
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Body, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel

from backend.db.session import SessionLocal
from backend.auth.deps import get_current_user
from backend.models.user import User
from backend.models.conversation import Conversation
from backend.models.message import Message, MessageRole
from backend.models.lab_report import LabReport
from backend.services import gemini
from backend.services import summarizer


router = APIRouter(prefix="/api/chat", tags=["chat"])


class StartChatIn(BaseModel):
    title: Optional[str] = None
    active_lab_id: Optional[str] = None


@router.post("/start")
def start_chat(
    payload: StartChatIn = Body(...),
    current_user: User = Depends(get_current_user),
):
    title = (payload.title or "").strip() or None
    active_lab_id = (payload.active_lab_id or "").strip() or None

    with SessionLocal() as db:
        conv = Conversation(user_id=str(current_user.id), title=title, active_lab_id=active_lab_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return {"conversation_id": conv.id}


class SendIn(BaseModel):
    conversation_id: str
    message: str


def _build_system_prompt(db: Session, conv: Conversation) -> str:
    # Prefer active lab context; otherwise fall back to latest lab for this user
    lab_payload: Optional[Dict[str, Any]] = None
    if conv.active_lab_id:
        lab = db.query(LabReport).filter(LabReport.id == conv.active_lab_id).first()
        if lab and lab.structured_json:
            lab_payload = lab.structured_json
    if lab_payload is None:
        latest = (
            db.query(LabReport)
            .filter(LabReport.user_id == conv.user_id)
            .order_by(desc(LabReport.created_at))
            .first()
        )
        if latest and latest.structured_json:
            lab_payload = latest.structured_json
    sys_parts = ["You are a concise health assistant."]
    if lab_payload:
        try:
            sys_parts.append("ActiveLab=" + json.dumps(lab_payload))
        except Exception:
            sys_parts.append("ActiveLab=unavailable")
    return "\n".join(sys_parts)


@router.post("/send")
async def send_message(
    payload: SendIn = Body(...),
    current_user: User = Depends(get_current_user),
):
    conv_id = (payload.conversation_id or "").strip()
    text = (payload.message or "").strip()
    if not conv_id or not text:
        raise HTTPException(status_code=422, detail="conversation_id and message are required")

    db = SessionLocal()
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == str(current_user.id)).first()
    if not conv:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Persist user message
    umsg = Message(conversation_id=conv.id, user_id=str(current_user.id), role=MessageRole.USER, content=text)
    db.add(umsg)
    db.commit()
    db.refresh(umsg)

    # Prepare context and call LLM
    system_prompt = _build_system_prompt(db, conv)
    history: List[Dict[str, str]] = [
        {"role": "user", "content": text}
    ]
    try:
        reply = await gemini.generate_chat(system_prompt, history)
    except Exception:
        reply = ""

    # Persist assistant message
    amsg = Message(conversation_id=conv.id, user_id=str(current_user.id), role=MessageRole.ASSISTANT, content=str(reply or ""))
    db.add(amsg)
    db.commit()
    db.refresh(amsg)

    # Rolling summary and retention
    RETAIN = 50
    total = db.query(Message).filter(Message.conversation_id == conv.id).count()
    if total > RETAIN:
        # Summarize and trim oldest to keep size under RETAIN
        try:
            all_msgs = (
                db.query(Message)
                .filter(Message.conversation_id == conv.id)
                .order_by(Message.created_at.asc())
                .all()
            )
            minimal_history = [{"role": m.role.value, "content": m.content} for m in all_msgs]
            conv.rolling_summary = await summarizer.summarize_messages(minimal_history, previous_summary=conv.rolling_summary)
            db.add(conv)

            # Trim messages to retain last RETAIN messages
            to_delete = all_msgs[:-RETAIN] if len(all_msgs) > RETAIN else []
            for m in to_delete:
                db.delete(m)
            db.commit()
        except Exception:
            # best-effort; do not fail the request
            pass

    result = {
        "conversation_id": conv.id,
        "message": {
            "id": amsg.id,
            "role": amsg.role.value,
            "content": amsg.content,
        },
    }
    db.close()
    return result


@router.get("/{conversation_id}/history")
def get_history(
    conversation_id: str,
    limit: int = Query(20, ge=1, le=100),
    before_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == str(current_user.id)).first()
    if not conv:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Fetch all ordered messages (conversations are small), then slice for simple cursor pagination
    ordered = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .all()
    )
    items: list[Message]
    if before_id:
        ids = [m.id for m in ordered]
        try:
            idx = ids.index(before_id)
            items = ordered[idx + 1 : idx + 1 + limit]
        except ValueError:
            items = ordered[:limit]
    else:
        items = ordered[:limit]
    # Return oldest-first within the page for stable pagination semantics
    items = list(reversed(items))

    def to_dict(m: Message) -> Dict[str, Any]:
        return {
            "id": m.id,
            "role": m.role.value,
            "content": m.content,
        }
    out = {"messages": [to_dict(m) for m in items]}
    db.close()
    return out
