"""
personalization_api.py — REST API endpoints for User Personalization Profile management.

Provides:
  - GET  /personalization/profile
  - PUT  /personalization/profile
  - POST /personalization/interests/manual
  - DELETE /personalization/interests/{category}
  - POST /personalization/interests/restore/{category}
  - POST /personalization/toggle
  - POST /personalization/reset
  - GET  /personalization/categories
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session

from auth import get_current_user
from database import get_session
from models import (
    PersonalizationProfile,
    PersonalizationProfileRead,
    PersonalizationProfileUpdate,
    User,
    UserPreference,
    utc_now,
)
from personalization_agent import (
    PREDEFINED_CATEGORIES,
    generate_personalization_summary,
    get_or_create_profile,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/personalization", tags=["Personalization"])


class ManualInterestRequest(BaseModel):
    category: str = Field(..., min_length=2, max_length=60)


class ToggleRequest(BaseModel):
    enabled: Optional[bool] = None


def _format_profile_response(profile: PersonalizationProfile) -> PersonalizationProfileRead:
    """Format a PersonalizationProfile SQLModel into a structured PersonalizationProfileRead schema."""
    try:
        inferred = json.loads(profile.inferred_interests_json or "[]")
    except Exception:
        inferred = []

    try:
        excluded = json.loads(profile.excluded_interests_json or "[]")
    except Exception:
        excluded = []

    try:
        preferred_cats = json.loads(profile.preferred_categories_json or "[]")
    except Exception:
        preferred_cats = []

    try:
        preferred_persp = json.loads(profile.preferred_perspectives_json or "[]")
    except Exception:
        preferred_persp = []

    summary = generate_personalization_summary(profile)

    return PersonalizationProfileRead(
        user_id=profile.user_id,
        enabled=profile.enabled,
        explicit_interests=preferred_cats,
        inferred_interests=inferred,
        excluded_interests=excluded,
        preferred_categories=preferred_cats,
        preferred_perspectives=preferred_persp,
        preferred_language=profile.preferred_language,
        preferred_length=profile.preferred_length,
        last_profile_update=profile.last_profile_update.isoformat() if profile.last_profile_update else "",
        summary_text=summary,
    )


@router.get("/categories", summary="List predefined interest categories")
def list_categories() -> List[str]:
    """Return all 14 predefined supported interest categories."""
    return PREDEFINED_CATEGORIES


@router.get("/profile", response_model=PersonalizationProfileRead, summary="Get user personalization profile")
def get_user_personalization_profile(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Retrieve the current user's complete dynamic personalization profile."""
    profile = get_or_create_profile(current_user.id, session)
    return _format_profile_response(profile)


@router.put("/profile", response_model=PersonalizationProfileRead, summary="Update explicit preferences")
def update_user_personalization_profile(
    update_data: PersonalizationProfileUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Update explicit preferences (preferred categories, perspectives, language, length, enabled)."""
    profile = get_or_create_profile(current_user.id, session)

    if update_data.enabled is not None:
        profile.enabled = update_data.enabled

    if update_data.preferred_categories is not None:
        valid_cats = [c for c in update_data.preferred_categories if c in PREDEFINED_CATEGORIES]
        profile.preferred_categories_json = json.dumps(valid_cats)

    if update_data.preferred_perspectives is not None:
        profile.preferred_perspectives_json = json.dumps(update_data.preferred_perspectives)

    if update_data.preferred_language is not None:
        profile.preferred_language = update_data.preferred_language

    if update_data.preferred_length is not None:
        profile.preferred_length = update_data.preferred_length

    if update_data.excluded_interests is not None:
        profile.excluded_interests_json = json.dumps(list(set(update_data.excluded_interests)))

    profile.last_profile_update = utc_now()
    session.add(profile)

    # Sync back to legacy UserPreference table for backwards compatibility
    legacy_pref = session.query(UserPreference).filter(UserPreference.user_id == current_user.id).first()
    if legacy_pref:
        if update_data.preferred_categories is not None:
            legacy_pref.preferred_categories_json = profile.preferred_categories_json
        if update_data.preferred_perspectives is not None:
            legacy_pref.preferred_perspectives_json = profile.preferred_perspectives_json
        if update_data.preferred_language is not None:
            legacy_pref.preferred_language = profile.preferred_language
        if update_data.preferred_length is not None:
            legacy_pref.preferred_length = profile.preferred_length
        session.add(legacy_pref)

    session.commit()
    session.refresh(profile)
    return _format_profile_response(profile)


@router.post("/interests/manual", response_model=PersonalizationProfileRead, summary="Manually add an interest")
def add_manual_interest(
    req: ManualInterestRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Manually add or activate an interest topic in the user's profile."""
    # Find matching predefined category (case-insensitive)
    matched_cat = next((c for c in PREDEFINED_CATEGORIES if c.lower() == req.category.strip().lower()), None)
    if not matched_cat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Category '{req.category}' is not supported. Must be one of: {PREDEFINED_CATEGORIES}",
        )

    profile = get_or_create_profile(current_user.id, session)

    # Remove from excluded list if previously excluded
    try:
        excluded = set(json.loads(profile.excluded_interests_json or "[]"))
    except Exception:
        excluded = set()
    if matched_cat in excluded:
        excluded.remove(matched_cat)
        profile.excluded_interests_json = json.dumps(list(excluded))

    # Add to inferred interests as high-confidence manual interest
    try:
        inferred = json.loads(profile.inferred_interests_json or "[]")
    except Exception:
        inferred = []

    now_iso = utc_now().isoformat()
    existing = next((item for item in inferred if item.get("category") == matched_cat), None)
    if existing:
        existing["score"] = 0.90
        existing["confidence"] = 0.90
        existing["status"] = "recurring"
        existing["last_interaction"] = now_iso
        existing["source"] = "manual"
    else:
        inferred.append({
            "category": matched_cat,
            "score": 0.90,
            "confidence": 0.90,
            "supporting_interactions": 3,
            "first_detected": now_iso,
            "last_interaction": now_iso,
            "status": "recurring",
            "source": "manual",
        })

    profile.inferred_interests_json = json.dumps(inferred)
    profile.last_profile_update = utc_now()
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return _format_profile_response(profile)


@router.delete("/interests/{category}", response_model=PersonalizationProfileRead, summary="Remove/exclude an interest")
def remove_or_exclude_interest(
    category: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Remove an inferred interest and permanently mark it as excluded so it is never re-inferred."""
    profile = get_or_create_profile(current_user.id, session)

    target_cat = category.strip()
    matched_cat = next((c for c in PREDEFINED_CATEGORIES if c.lower() == target_cat.lower()), target_cat)

    # 1. Remove from inferred
    try:
        inferred = json.loads(profile.inferred_interests_json or "[]")
        inferred = [i for i in inferred if i.get("category", "").lower() != matched_cat.lower()]
        profile.inferred_interests_json = json.dumps(inferred)
    except Exception:
        pass

    # 2. Remove from explicit preferred categories if present
    try:
        preferred = json.loads(profile.preferred_categories_json or "[]")
        preferred = [c for c in preferred if c.lower() != matched_cat.lower()]
        profile.preferred_categories_json = json.dumps(preferred)
    except Exception:
        pass

    # 3. Add to excluded list
    try:
        excluded = set(json.loads(profile.excluded_interests_json or "[]"))
    except Exception:
        excluded = set()
    excluded.add(matched_cat)
    profile.excluded_interests_json = json.dumps(list(excluded))

    profile.last_profile_update = utc_now()
    session.add(profile)
    session.commit()
    session.refresh(profile)
    log.info("User id=%d excluded interest '%s'", current_user.id, matched_cat)
    return _format_profile_response(profile)


@router.post("/interests/restore/{category}", response_model=PersonalizationProfileRead, summary="Restore an excluded interest")
def restore_excluded_interest(
    category: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Restore a previously excluded category so it can be learned or used again."""
    profile = get_or_create_profile(current_user.id, session)

    target_cat = category.strip()
    try:
        excluded = set(json.loads(profile.excluded_interests_json or "[]"))
        to_remove = {c for c in excluded if c.lower() == target_cat.lower()}
        excluded.difference_update(to_remove)
        profile.excluded_interests_json = json.dumps(list(excluded))
    except Exception:
        pass

    profile.last_profile_update = utc_now()
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return _format_profile_response(profile)


@router.post("/toggle", response_model=PersonalizationProfileRead, summary="Toggle personalization on/off")
def toggle_personalization(
    req: Optional[ToggleRequest] = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Enable or disable personalization for the current user."""
    profile = get_or_create_profile(current_user.id, session)
    if req and req.enabled is not None:
        profile.enabled = req.enabled
    else:
        profile.enabled = not profile.enabled

    profile.last_profile_update = utc_now()
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return _format_profile_response(profile)


@router.post("/reset", response_model=PersonalizationProfileRead, summary="Reset learned interests")
def reset_inferred_interests(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Reset all learned inferred interests and clear exclusions, keeping explicit profile settings."""
    profile = get_or_create_profile(current_user.id, session)
    profile.inferred_interests_json = "[]"
    profile.excluded_interests_json = "[]"
    profile.last_profile_update = utc_now()
    session.add(profile)
    session.commit()
    session.refresh(profile)
    log.info("Reset learned personalization interests for user_id=%d", current_user.id)
    return _format_profile_response(profile)
