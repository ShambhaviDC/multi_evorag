# preferences.py – API routes for user preferences (onboarding & profile)

"""Endpoints to read and update a user's preferences.

All routes require authentication via the ``/auth/me`` dependency.

* ``GET /preferences`` – Return the current ``UserPreference`` record (or create a default one).
* ``PUT /preferences`` – Update fields such as age_range, role, interests, response_style, onboarding_completed.
"""

from fastapi import APIRouter, Depends, HTTPException, status
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
        # create a default preference record for the user
        pref = UserPreference(user_id=current_user.id)
        session.add(pref)
        session.commit()
        session.refresh(pref)
    return pref

@router.put("/", response_model=UserPreference)
def update_preferences(
    age_range: str | None = None,
    role: str | None = None,
    interests: str | None = None,
    response_style: str | None = None,
    onboarding_completed: bool | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    stmt = select(UserPreference).where(UserPreference.user_id == current_user.id)
    pref = session.exec(stmt).first()
    if not pref:
        pref = UserPreference(user_id=current_user.id)
        session.add(pref)
        session.commit()
        session.refresh(pref)
    if age_range is not None:
        pref.age_range = age_range
    if role is not None:
        pref.role = role
    if interests is not None:
        pref.interests = interests
    if response_style is not None:
        pref.response_style = response_style
    if onboarding_completed is not None:
        pref.onboarding_completed = onboarding_completed
    session.add(pref)
    session.commit()
    session.refresh(pref)
    return pref

