"""
event_api.py — REST Endpoints for Temporal News Event Tracking and Evolution Analysis

Provides:
  - GET  /events                     → List tracked evolving events with category & status filters
  - GET  /events/categories          → List categories and active event counts
  - GET  /events/{event_id}          → Full event overview and synthesized evolution summary
  - GET  /events/{event_id}/timeline → Chronological development timeline with verified citations
  - GET  /events/{event_id}/relationships → Evidence-supported related and follow-up events
  - POST /events/cluster/refresh     → Trigger incremental clustering sweep
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from database import get_session
from models import NewsEvent, EventArticleMapping, EventRelationship, NewsArticle
from event_evolution_service import (
    get_event_timeline,
    generate_event_evolution_summary,
    cluster_all_unassigned_articles,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", summary="List tracked evolving news events")
def list_events(
    category: Optional[str] = Query(None, description="Filter by event category"),
    status: Optional[str] = Query(None, description="Filter by event status (Emerging, Ongoing, Developing, Resolved)"),
    search: Optional[str] = Query(None, description="Keyword search in event titles"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    stmt = select(NewsEvent)
    if category and category.lower() != "all":
        stmt = stmt.where(NewsEvent.event_category == category)
    if status and status.lower() != "all":
        stmt = stmt.where(NewsEvent.current_status == status)
    if search:
        search_term = f"%{search.strip()}%"
        stmt = stmt.where(NewsEvent.event_title.ilike(search_term))

    stmt = stmt.order_by(NewsEvent.last_updated.desc()).offset(offset).limit(limit)
    events = session.exec(stmt).all()

    # Get total count
    total_count = len(session.exec(select(NewsEvent.id)).all())

    results = []
    for evt in events:
        entities = json.loads(evt.involved_entities or "[]")
        topics = json.loads(evt.associated_topics or "[]")
        results.append({
            "event_id": evt.event_id,
            "title": evt.event_title,
            "description": evt.event_description,
            "category": evt.event_category,
            "status": evt.current_status,
            "status_confidence": round(evt.status_confidence, 2),
            "start_date": evt.event_start_date,
            "last_updated": evt.last_updated,
            "article_count": evt.article_count,
            "primary_location": evt.primary_location,
            "entities": entities[:6],
            "topics": topics[:6],
        })

    return {
        "total_events": total_count,
        "returned": len(results),
        "events": results,
    }


@router.get("/categories", summary="List categories and event counts")
def get_event_categories(session: Session = Depends(get_session)):
    events = session.exec(select(NewsEvent)).all()
    from collections import Counter
    counts = Counter(e.event_category for e in events)
    return {
        "categories": sorted(list(counts.keys())),
        "counts": counts,
        "total_events": len(events),
    }


@router.get("/{event_id}", summary="Get event details and evolved summary")
def get_event_detail(event_id: str, session: Session = Depends(get_session)):
    evt = session.exec(select(NewsEvent).where(NewsEvent.event_id == event_id)).first()
    if not evt:
        raise HTTPException(status_code=404, detail="News event not found")

    summary_data = generate_event_evolution_summary(event_id, session)
    return summary_data


@router.get("/{event_id}/timeline", summary="Get sorted chronological timeline for an event")
def get_timeline(event_id: str, session: Session = Depends(get_session)):
    evt = session.exec(select(NewsEvent).where(NewsEvent.event_id == event_id)).first()
    if not evt:
        raise HTTPException(status_code=404, detail="News event not found")

    timeline = get_event_timeline(event_id, session)
    return {
        "event_id": evt.event_id,
        "event_title": evt.event_title,
        "category": evt.event_category,
        "status": evt.current_status,
        "total_developments": len(timeline),
        "timeline": timeline,
    }


@router.get("/{event_id}/relationships", summary="Get related and follow-up events")
def get_event_relationships(event_id: str, session: Session = Depends(get_session)):
    stmt = select(EventRelationship).where(
        (EventRelationship.source_event_id == event_id) |
        (EventRelationship.target_event_id == event_id)
    )
    rels = session.exec(stmt).all()
    results = []
    for r in rels:
        other_id = r.target_event_id if r.source_event_id == event_id else r.source_event_id
        other_evt = session.exec(select(NewsEvent).where(NewsEvent.event_id == other_id)).first()
        results.append({
            "target_event_id": other_id,
            "target_event_title": other_evt.event_title if other_evt else "Related Event",
            "relationship_type": r.relationship_type,
            "confidence": round(r.confidence, 2),
            "description": r.description,
        })
    return {"event_id": event_id, "relationships": results}


@router.post("/cluster/refresh", summary="Trigger incremental clustering sweep")
def refresh_event_clustering(session: Session = Depends(get_session)):
    res = cluster_all_unassigned_articles(session)
    return {
        "status": "success",
        "details": res,
    }
