# conversations.py – API routes for handling chat conversations and messages

"""FastAPI router exposing CRUD endpoints for user conversations.

Endpoints (all require authentication via the ``/auth/me`` dependency):

* ``GET /conversations`` – List a user's conversations (brief info).
* ``POST /conversations`` – Create a new conversation (title optional).
* ``GET /conversations/{conv_id}`` – Retrieve a conversation with its messages.
* ``DELETE /conversations/{conv_id}`` – Delete a conversation and all its messages.
* ``PATCH /conversations/{conv_id}`` – Update conversation title.
* ``POST /conversations/{conv_id}/messages`` – Append a user or assistant message.
"""

from typing import List
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from auth import get_current_user
from models import Conversation, Message, User
from database import get_session

router = APIRouter(prefix="/conversations", tags=["conversations"])

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
    statement = select(Conversation).where(Conversation.user_id == current_user.id)
    return session.exec(statement).all()

@router.post("/", response_model=Conversation, status_code=status.HTTP_201_CREATED)
def create_conversation(title: str = None, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    convo = Conversation(user_id=current_user.id, title=title)
    session.add(convo)
    session.commit()
    session.refresh(convo)
    return convo

@router.get("/{conv_id}/messages", response_model=List[Message])
def get_messages(conv_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    convo = session.get(Conversation, conv_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(current_user, convo)
    stmt = select(Message).where(Message.conversation_id == conv_id).order_by(Message.timestamp)
    return session.exec(stmt).all()


@router.delete("/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conv_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    convo = session.get(Conversation, conv_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(current_user, convo)
    # Delete associated messages first
    msg_stmt = select(Message).where(Message.conversation_id == conv_id)
    for msg in session.exec(msg_stmt):
        session.delete(msg)
    session.delete(convo)
    session.commit()
    return

@router.patch("/{conv_id}", response_model=Conversation)
def update_conversation(conv_id: int, title: str, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    convo = session.get(Conversation, conv_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(current_user, convo)
    convo.title = title
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
    session.commit()
    session.refresh(message)
    return message
