# recommendations.py – simple recommendation endpoint

"""Generate lightweight, personalized prompt suggestions for a user.

The logic is deliberately simple: it reads the user's preferences and recent
conversation titles (if any) then returns a small list of suggested prompts.
No LLM calls are made – the goal is to satisfy the “personalized
recommendation engine” requirement without heavy processing.
"""

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from auth import get_current_user
from models import Conversation, UserPreference, User
from database import get_session

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Helper to fetch recent conv titles (last 5)
def recent_titles(session: Session, user_id: int):
    stmt = (
        select(Conversation.title)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(5)
    )
    return [t for (t,) in session.exec(stmt).all() if t]

@router.get("/", summary="Get personalized prompt suggestions")
def get_recommendations(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    # Load preferences (create default if missing)
    pref_stmt = select(UserPreference).where(UserPreference.user_id == current_user.id)
    pref = session.exec(pref_stmt).first()
    if not pref:
        pref = UserPreference(user_id=current_user.id)
        session.add(pref)
        session.commit()
        session.refresh(pref)
    # Base suggestions
    suggestions = []
    if pref.interests:
        interests = [i.strip() for i in pref.interests.split(',') if i]
        for interest in interests[:3]:
            suggestions.append(f"Explain the latest developments in {interest}.")
    # Fallback generic
    if not suggestions:
        suggestions.extend([
            "Explain a scientific concept.",
            "Summarize recent AI research.",
            "Give a quick overview of a technology trend.",
        ])
    # Add recent conversation‑based suggestions
    titles = recent_titles(session, current_user.id)
    for t in titles:
        suggestions.append(f"Continue the discussion on '{t}'.")
    # Limit to 6 total
    return {"suggestions": suggestions[:6]}

