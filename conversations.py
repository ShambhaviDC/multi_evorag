# conversations.py - API routes for handling chat conversations and messages

import re
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from auth import get_current_user
from models import Conversation, Message, User
from database import get_session

router = APIRouter(prefix="/conversations", tags=["conversations"])


def clean_title(raw_text: Optional[str]) -> str:
    """Generate a concise, readable title from user query."""
    if not raw_text:
        return "New Research Chat"
    t = raw_text.strip()
    starters = [
        r"^(?:what\s+(?:is|are|were|was)\s+(?:the\s+)?)",
        r"^(?:how\s+(?:do|does|did|can|could|would)\s+(?:the\s+)?)",
        r"^(?:why\s+(?:is|are|do|does|did)\s+)",
        r"^(?:can\s+you\s+(?:please\s+)?(?:explain|tell\s+me\s+about|describe)\s+)",
        r"^(?:could\s+you\s+(?:please\s+)?(?:explain|tell\s+me\s+about|describe)\s+)",
        r"^(?:explain\s+(?:to\s+me\s+)?(?:the\s+)?)",
        r"^(?:tell\s+me\s+about\s+(?:the\s+)?)",
        r"^(?:give\s+(?:me\s+)?(?:an\s+overview|a\s+summary|details)\s+of\s+)",
    ]
    cleaned = t
    for pat in starters:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[?!.,;:]+$", "", cleaned).strip()
    if not cleaned:
        cleaned = t
    if len(cleaned) > 42:
        words = cleaned[:42].rsplit(" ", 1)
        cleaned = words[0] if len(words) > 1 and len(words[0]) > 10 else cleaned[:40]
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or "New Research Chat"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def verify_ownership(user: User, convo: Conversation):
    if convo.user_id != user.id:
        raise HTTPException(status_code=403, detail="Operation not permitted.")

# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=List[Conversation])
def list_conversations(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    statement = select(Conversation).where(Conversation.user_id == current_user.id).order_by(Conversation.updated_at.desc())
    return session.exec(statement).all()


@router.post("/", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: Request,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    title = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            title = body.get("title")
        except Exception:
            pass
    if not title:
        title = request.query_params.get("title")

    formatted_title = clean_title(title)
    convo = Conversation(user_id=current_user.id, title=formatted_title)
    session.add(convo)
    session.commit()
    session.refresh(convo)
    return convo


@router.get("/{conv_id}")
def get_conversation(conv_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    convo = session.get(Conversation, conv_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(current_user, convo)
    stmt = select(Message).where(Message.conversation_id == conv_id).order_by(Message.timestamp)
    messages = session.exec(stmt).all()
    convo_dict = convo.model_dump() if hasattr(convo, "model_dump") else convo.dict()
    convo_dict["messages"] = [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in messages]
    return convo_dict


@router.get("/{conv_id}/messages", response_model=List[Message])
def get_messages(conv_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    convo = session.get(Conversation, conv_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(current_user, convo)
    stmt = select(Message).where(Message.conversation_id == conv_id).order_by(Message.timestamp)
    return session.exec(stmt).all()


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
def clear_all_conversations(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Delete all conversations and messages for the current user."""
    stmt = select(Conversation).where(Conversation.user_id == current_user.id)
    convos = session.exec(stmt).all()
    for convo in convos:
        msg_stmt = select(Message).where(Message.conversation_id == convo.id)
        for msg in session.exec(msg_stmt):
            session.delete(msg)
        session.delete(convo)
    session.commit()
    return


@router.delete("/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conv_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    convo = session.get(Conversation, conv_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(current_user, convo)
    msg_stmt = select(Message).where(Message.conversation_id == conv_id)
    for msg in session.exec(msg_stmt):
        session.delete(msg)
    session.delete(convo)
    session.commit()
    return


@router.patch("/{conv_id}", response_model=Conversation)
async def update_conversation(
    conv_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    convo = session.get(Conversation, conv_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(current_user, convo)
    title = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            title = body.get("title")
        except Exception:
            pass
    if not title:
        title = request.query_params.get("title")
    if title:
        convo.title = clean_title(title)
        convo.updated_at = datetime.now(timezone.utc)
        session.add(convo)
        session.commit()
        session.refresh(convo)
    return convo


# ---------------------------------------------------------------------------
# Message endpoint
# ---------------------------------------------------------------------------

class MessageCreate(BaseModel):
    role: str  # "user" or "assistant"
    content: str


@router.post("/{conv_id}/messages", response_model=Message, status_code=status.HTTP_201_CREATED)
def add_message(conv_id: int, msg: MessageCreate, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    convo = session.get(Conversation, conv_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(current_user, convo)
    if msg.role not in {"user", "assistant"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    message = Message(conversation_id=conv_id, role=msg.role, content=msg.content)
    session.add(message)
    convo.updated_at = datetime.now(timezone.utc)
    session.add(convo)
    session.commit()
    session.refresh(message)
    return message
