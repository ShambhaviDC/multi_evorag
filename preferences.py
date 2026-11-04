# preferences.py – API routes for user preferences (onboarding & profile)

"""Endpoints to read and update a user's preferences.

All routes require authentication via the ``/auth/me`` dependency.

* ``GET /preferences`` – Return the current ``UserPreference`` record (or create a default one).
* ``PUT /preferences`` – Update fields such as age_range, role, interests, response_style, onboarding_completed.
"""

from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from auth import get_current_user
from models import UserPreference, User
from database import get_session

router = APIRouter(prefix="/preferences", tags=["preferences"])


@router.get("/", response_model=UserPreference)
def get_preferences(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    stmt = select(UserPreference).where(UserPreference.user_id == current_user.id)
    pref = session.exec(stmt).first()
    if not pref:
        pref = UserPreference(user_id=current_user.id)
        session.add(pref)
        session.commit()
        session.refresh(pref)
    return pref


@router.put("/", response_model=UserPreference)
async def update_preferences(
    request: Request,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    data = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
        except Exception:
            data = {}
    else:
        data = dict(request.query_params)

    stmt = select(UserPreference).where(UserPreference.user_id == current_user.id)
    pref = session.exec(stmt).first()
    if not pref:
        pref = UserPreference(user_id=current_user.id)
        session.add(pref)
        session.commit()
        session.refresh(pref)

    if "age_range" in data and data["age_range"] is not None:
        pref.age_range = str(data["age_range"])
    if "role" in data and data["role"] is not None:
        pref.role = str(data["role"])
    if "interests" in data and data["interests"] is not None:
        pref.interests = str(data["interests"])
    if "response_style" in data and data["response_style"] is not None:
        pref.response_style = str(data["response_style"])
    if "onboarding_completed" in data and data["onboarding_completed"] is not None:
        pref.onboarding_completed = bool(data["onboarding_completed"])

    session.add(pref)
    session.commit()
    session.refresh(pref)
    return pref
