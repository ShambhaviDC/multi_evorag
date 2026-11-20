"""
feedback_api.py — REST Endpoints for Feedback Intelligence & Analytics.

Provides:
  - POST   /api/feedback                    → Submit or update like/dislike feedback
  - DELETE /api/feedback/{message_id}       → Remove feedback for a message
  - GET    /api/feedback/message/{msg_id}   → Fetch active feedback for a message
  - GET    /api/feedback/user               → Get authenticated user's feedback history
  - GET    /api/feedback/analytics          → Aggregate feedback analytics & Chart.js data
  - GET    /api/feedback/recommendations    → Automated system diagnostic recommendations
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, status
from sqlmodel import Session

from auth import get_current_user, get_optional_current_user
from database import get_session
from feedback_intelligence import (
    delete_feedback,
    generate_system_improvement_recommendations,
    get_feedback_analytics,
    get_feedback_for_message,
    get_user_feedback_history,
    learn_user_preferences_from_feedback,
    save_or_update_feedback,
)
from models import FeedbackCreate, FeedbackRead, User

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def _format_feedback_response(fb) -> Dict[str, Any]:
    import json
    meta = {}
    if getattr(fb, "execution_metadata_json", None):
        try:
            meta = json.loads(fb.execution_metadata_json)
        except Exception:
            meta = {}

    return {
        "feedback_id": fb.feedback_id,
        "user_id": fb.user_id,
        "session_id": fb.session_id,
        "conversation_id": fb.conversation_id,
        "message_id": fb.message_id,
        "query_text": fb.query_text,
        "feedback_type": fb.feedback_type,
        "reason": fb.reason,
        "comment": fb.comment,
        "execution_metadata": meta,
        "created_at": fb.created_at.isoformat() if fb.created_at else "",
        "updated_at": fb.updated_at.isoformat() if fb.updated_at else "",
    }


@router.post("", summary="Submit or update like/dislike feedback on assistant response")
def submit_feedback(
    payload: FeedbackCreate,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """
    Record like/dislike feedback with optional issue reason and comment.
    Supports both authenticated users and guest sessions.
    Automatically triggers continuous preference learning in background for authenticated users.
    """
    clean_session = payload.session_id or x_session_id

    try:
        fb_record, action = save_or_update_feedback(
            session=session,
            message_id=payload.message_id,
            feedback_type=payload.feedback_type,
            user=current_user,
            session_id=clean_session,
            conversation_id=payload.conversation_id,
            query_text=payload.query_text,
            reason=payload.reason,
            comment=payload.comment,
            execution_metadata=payload.execution_metadata,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        log.exception("Error saving feedback")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save feedback.")

    # Trigger background preference learning if user is authenticated
    if current_user and getattr(current_user, "id", None):
        background_tasks.add_task(learn_user_preferences_from_feedback, user_id=current_user.id, session=session)

    return {
        "status": "success",
        "action": action,
        "feedback": _format_feedback_response(fb_record),
    }


@router.delete("/{message_id}", summary="Remove feedback for a message")
def remove_feedback(
    message_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """Remove active feedback (toggled off by user clicking active icon again)."""
    deleted = delete_feedback(
        session=session,
        message_id=message_id,
        user=current_user,
        session_id=x_session_id,
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found for this message.")

    return {"status": "success", "message": "Feedback removed successfully.", "message_id": message_id}


@router.get("/message/{message_id}", summary="Get active feedback for a specific message")
def get_message_feedback(
    message_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """Fetch feedback submitted by the current user or guest session for a specific message."""
    fb = get_feedback_for_message(
        session=session,
        message_id=message_id,
        user=current_user,
        session_id=x_session_id,
    )
    if not fb:
        return {"has_feedback": False, "feedback": None}

    return {"has_feedback": True, "feedback": _format_feedback_response(fb)}


@router.get("/user", summary="Get feedback history for authenticated user")
def list_user_feedback(
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """List recent feedback submitted by the authenticated user."""
    records = get_user_feedback_history(session=session, user_id=current_user.id, limit=limit)
    return {
        "user_id": current_user.id,
        "count": len(records),
        "history": [_format_feedback_response(r) for r in records],
    }


@router.get("/analytics", summary="Aggregated feedback analytics and visualizations")
def get_analytics(session: Session = Depends(get_session)):
    """Fetch live system analytics, satisfaction rates, and Chart.js datasets."""
    return get_feedback_analytics(session=session)


@router.get("/recommendations", summary="Continuous system diagnostic recommendations")
def get_recommendations(session: Session = Depends(get_session)):
    """Get automated improvement recommendations for retrieval, synthesis, and personas."""
    recs = generate_system_improvement_recommendations(session=session)
    return {"recommendations": recs}
