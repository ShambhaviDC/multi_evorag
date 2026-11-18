# preferences.py – API routes for user preferences (onboarding & profile)

"""Endpoints to read and update a user's preferences.

All routes require authentication via the ``/auth/me`` dependency.

* ``GET /preferences`` – Return the current ``UserPreference`` record (or create a default one).
* ``PUT /preferences`` – Update fields such as age_range, role, interests, response_style, onboarding_completed.
"""

import json
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from auth import get_current_user
from models import UserPreference, User, PersonalizationProfile, utc_now
from database import get_session
from news_service import ALL_NEWS_CATEGORIES

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

    # Categories handling
    categories = data.get("categories") or data.get("preferred_categories")
    if categories is not None:
        if isinstance(categories, list):
            pref.preferred_categories_json = json.dumps(categories)
        elif isinstance(categories, str):
            try:
                pref.preferred_categories_json = json.dumps(json.loads(categories))
            except Exception:
                pref.preferred_categories_json = json.dumps([c.strip() for c in categories.split(",") if c.strip()])

    # Subcategories handling
    subcategories = data.get("subcategories") or data.get("preferred_subcategories")
    if subcategories is not None:
        if isinstance(subcategories, list):
            pref.preferred_subcategories_json = json.dumps(subcategories)
        elif isinstance(subcategories, str):
            try:
                pref.preferred_subcategories_json = json.dumps(json.loads(subcategories))
            except Exception:
                pref.preferred_subcategories_json = json.dumps([s.strip() for s in subcategories.split(",") if s.strip()])

    # Enforce mandatory onboarding requirement: at least 1 category required!
    onboard_flag = data.get("onboarding_completed")
    if onboard_flag:
        current_cats = json.loads(pref.preferred_categories_json or "[]")
        if not current_cats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one major category must be selected to complete onboarding."
            )
        pref.onboarding_completed = True
    elif onboard_flag is not None:
        pref.onboarding_completed = bool(onboard_flag)

    if "age_range" in data and data["age_range"] is not None:
        pref.age_range = str(data["age_range"])
    if "role" in data and data["role"] is not None:
        pref.role = str(data["role"])
    if "interests" in data and data["interests"] is not None:
        pref.interests = str(data["interests"])
    if "response_style" in data and data["response_style"] is not None:
        pref.response_style = str(data["response_style"])

    pref.updated_at = utc_now()
    session.add(pref)

    # Sync with PersonalizationProfile
    profile = session.exec(select(PersonalizationProfile).where(PersonalizationProfile.user_id == current_user.id)).first()
    if not profile:
        profile = PersonalizationProfile(user_id=current_user.id)

    if pref.preferred_categories_json:
        profile.preferred_categories_json = pref.preferred_categories_json
    if pref.preferred_subcategories_json:
        profile.preferred_subcategories_json = pref.preferred_subcategories_json
    if pref.onboarding_completed:
        profile.onboarding_completed = True
        profile.enabled = True
    profile.last_profile_update = utc_now()
    session.add(profile)

    session.commit()
    session.refresh(pref)
    return pref
