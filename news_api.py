"""
news_api.py — REST API router for News Discovery, Personalized Feeds, and Smart Autocomplete.

Endpoints:
  - GET  /news/categories
  - GET  /news/homepage
  - GET  /news/personalized
  - GET  /news/recent
  - POST /news/refresh
  - GET  /autocomplete
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from auth import get_current_user, get_optional_current_user
from autocomplete_service import get_autocomplete_suggestions
from database import get_session
from models import NewsArticle, NewsArticleRead, User, UserPreference
from news_service import (
    ALL_NEWS_CATEGORIES,
    NEWS_TAXONOMY,
    get_homepage_news,
    get_personalized_homepage_news,
    seed_news_if_needed,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["News & Autocomplete"])


@router.get("/news/categories", summary="List all 18 news categories and taxonomy")
def get_categories():
    """Return all 18 major news categories and their hierarchical subcategory taxonomy."""
    return {
        "categories": ALL_NEWS_CATEGORIES,
        "taxonomy": NEWS_TAXONOMY,
    }


@router.get("/news/homepage", summary="Get general news homepage for visitors/guests")
def get_general_homepage(
    category: Optional[str] = Query(None, description="Optional category filter (e.g. 'Cricket', 'Agriculture')"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    """Retrieve balanced, real-time news across all 18 categories for guest/general exploration."""
    return get_homepage_news(session, active_category=category, limit=limit, offset=offset)


@router.get("/news/personalized", summary="Get personalized news homepage for authenticated user")
def get_user_personalized_news(
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
):
    """
    Retrieve curated news tailored to user's onboarding preferences & learned interests.
    If unauthenticated, gracefully returns the general diverse news homepage.
    If authenticated user has not completed onboarding, indicates onboarding is required.
    """
    if not current_user:
        return get_homepage_news(session)

    # Check onboarding completion
    pref = session.exec(select(UserPreference).where(UserPreference.user_id == current_user.id)).first()
    if not pref or not pref.onboarding_completed:
        return {
            "onboarding_required": True,
            "is_personalized": False,
            "message": "Mandatory preference onboarding required before accessing personalized feed.",
            "categories": ALL_NEWS_CATEGORIES,
            "taxonomy": NEWS_TAXONOMY,
        }

    return get_personalized_homepage_news(current_user.id, session)


@router.get("/news/recent", summary="Get recent paginated news articles")
def get_recent_news(
    category: Optional[str] = Query(None),
    subcategory: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    """Query stored news articles with optional category and subcategory filtering."""
    stmt = select(NewsArticle)
    if category and category.lower() != "all":
        stmt = stmt.where(NewsArticle.category.ilike(category))
    if subcategory:
        stmt = stmt.where(NewsArticle.subcategory.ilike(subcategory))

    stmt = stmt.order_by(NewsArticle.created_at.desc()).offset(offset).limit(limit)
    articles = session.exec(stmt).all()

    return [
        {
            "id": a.id,
            "title": a.title,
            "url": a.url,
            "source": a.source,
            "category": a.category,
            "subcategory": a.subcategory or a.category,
            "published_at": a.published_at,
            "snippet": a.snippet,
            "image_url": a.image_url,
        }
        for a in articles
    ]


@router.post("/news/refresh", summary="Trigger news feed refresh sweep")
def refresh_news_feeds(background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    """Fetch fresh news articles in the background from all configured RSS sources."""
    background_tasks.add_task(seed_news_if_needed, force_refresh=True)
    return {"status": "ok", "message": "News ingestion refresh scheduled in background."}


@router.get("/autocomplete", summary="Smart context-aware query autocomplete")
def autocomplete_endpoint(
    q: str = Query("", min_length=1, max_length=200, description="Partial query string being typed"),
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
):
    """
    Generate instant autocomplete suggestions for query terms.
    Returns matched topics, categories, entities, headlines and replacement ranges.
    """
    user_id = current_user.id if current_user else None
    return get_autocomplete_suggestions(query_text=q, user_id=user_id, session=session)
