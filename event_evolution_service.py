"""
event_evolution_service.py — Event Evolution, Timeline, and Conflict/Correction Service

Handles:
  1. Incremental Ingestion & Idempotent Event Assignment
  2. Development Classification (Initial announcement, Policy modification, Correction, Conflict, etc.)
  3. Evidence-Grounded Event Lifecycle Management (Emerging, Developing, Ongoing, Resolved, Dormant)
  4. Non-Destructive Correction and Conflict Detection (Preserves historical record)
  5. Chronological Timeline Generation with explicit date basis
  6. Evolving Event Synthesis Summaries
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from models import NewsArticle, NewsEvent, EventArticleMapping, EventRelationship
from event_clustering_service import (
    extract_entities,
    parse_date_str,
    match_article_to_event,
    ClusteringConfig,
    DEFAULT_CONFIG,
    compute_text_embedding,
)

log = logging.getLogger(__name__)

# ── Development Type Classification Patterns ──────────────────────────────────

DEVELOPMENT_PATTERNS = [
    # Corrections and Retractions
    ("Correction", [
        r"\bcorrect(?:s|ed|ion)\b", r"\bretract(?:s|ed|ion)\b", r"\bamend(?:s|ed|ment)\b",
        r"\bpreviously reported\b", r"\berroneously\b", r"\bin contrast to earlier\b"
    ]),
    # Clarifications
    ("Clarification", [
        r"\bclarif(?:y|ies|ied|ication)\b", r"\bstatement clarify\b", r"\bclear up\b"
    ]),
    # Legal Action
    ("Legal action", [
        r"\blawsuit\b", r"\bsue(?:s|d)?\b", r"\bcourt\b", r"\bjudge\b", r"\bruling\b",
        r"\bverdict\b", r"\bindict(?:ment|ed)?\b", r"\blegal battle\b", r"\bplea\b"
    ]),
    # Policy Modification
    ("Policy modification", [
        r"\bpolicy modif(?:y|ication)\b", r"\bnew rule\b", r"\brevis(?:e|ed|ion)\b",
        r"\bpostpone(?:s|d)?\b", r"\bdelay(?:s|ed)?\b", r"\bdeadline extended\b",
        r"\bimplementation date\b", r"\bamended policy\b"
    ]),
    # Conflict or Controversy
    ("Conflict or controversy", [
        r"\bclash(?:es|ed)?\b", r"\bcontrovers(?:y|ial)\b", r"\bdispute(?:s|d)?\b",
        r"\bdenie(?:s|d)\b", r"\bdisagree(?:s|d)?\b", r"\bprotest(?:s|ed|ing)?\b",
        r"\boppos(?:e|ed|ition)\b", r"\bcriticiz(?:e|ed)\b"
    ]),
    # Official Response
    ("Official response", [
        r"\bofficial(?:ly)? respond(?:s|ed)?\b", r"\bgovernment said\b", r"\bspokesperson\b",
        r"\bstatement released\b", r"\bwhite house said\b", r"\bministry issued\b"
    ]),
    # Public Reaction
    ("Public reaction", [
        r"\bpublic react(?:s|ion)\b", r"\bcitizens\b", r"\bvoter(?:s)?\b",
        r"\bbacklash\b", r"\boutcry\b", r"\bdemonstrat(?:ion|ors)\b"
    ]),
    # Investigation
    ("Investigation", [
        r"\binvestigat(?:e|ion|ors|ing)\b", r"\bprobe\b", r"\binquiry\b",
        r"\baudit\b", r"\bscandal\b", r"\buncover(?:ed)?\b"
    ]),
    # Resolution
    ("Resolution", [
        r"\bsettle(?:s|d|ment)\b", r"\bresolution\b", r"\bagreement signed\b",
        r"\bdeal finalized\b", r"\bverdict reached\b", r"\bconclude(?:s|d)?\b"
    ]),
    # Initial Announcement
    ("Initial announcement", [
        r"\bannounc(?:e|es|ed|ing)\b", r"\bunveil(?:s|ed|ing)\b", r"\blaunch(?:es|ed|ing)\b",
        r"\bintroduc(?:e|es|ed|ing)\b", r"\bfirst reported\b", r"\bbreakout\b"
    ]),
]


def classify_development_type(title: str, snippet: str, sequence_num: int) -> str:
    """Classify the nature of the development from headline and snippet."""
    combined = f"{title}. {snippet}".lower()
    for dev_type, patterns in DEVELOPMENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, combined):
                return dev_type
    
    # Sequence based fallback
    if sequence_num == 1:
        return "Initial announcement"
    return "New development"


def determine_event_status(
    articles: List[Dict[str, Any]],
    current_status: str = "Emerging"
) -> Tuple[str, float]:
    """
    Evidence-grounded lifecycle status evaluation.
    Never marks an event as resolved solely because no recent articles exist.
    Requires official resolution evidence.
    """
    if not articles:
        return "Unknown", 0.3

    # Check for resolution keywords in any recent article
    recent_texts = [f"{a.get('title', '')} {a.get('snippet', '')}".lower() for a in articles[-3:]]
    for text in recent_texts:
        if re.search(r"\b(settled|agreement reached|final verdict|investigation concluded|deal finalized)\b", text):
            return "Resolved", 0.90

    # Count of development stages
    count = len(articles)
    if count == 1:
        return "Emerging", 0.85
    elif count < 4:
        return "Developing", 0.88
    else:
        return "Ongoing", 0.92


# ── Incremental Ingestion & Processing ────────────────────────────────────────

def cluster_article_incrementally(
    article: NewsArticle,
    session: Session,
    config: ClusteringConfig = DEFAULT_CONFIG,
) -> Tuple[NewsEvent, EventArticleMapping]:
    """
    Idempotently cluster a single news article into the temporal event database.
    Assigns to an existing event or seeds a new event cluster.
    """
    # 1. Idempotency check: see if article already mapped to an event
    existing_map = session.exec(
        select(EventArticleMapping).where(EventArticleMapping.article_id == article.id)
    ).first()
    if existing_map:
        evt = session.exec(select(NewsEvent).where(NewsEvent.event_id == existing_map.event_id)).first()
        if evt:
            return evt, existing_map

    # 2. Retrieve all active existing events in the database
    existing_events = session.exec(select(NewsEvent)).all()

    # 3. Match article to candidate events using Two-Stage Clustering
    match_result = match_article_to_event(article, existing_events, config=config)

    now = datetime.now(timezone.utc)
    art_date = article.published_at or now.strftime("%Y-%m-%d")

    # 4. Handle matching outcomes
    if match_result.match_type == "same_event" and match_result.matched_event:
        # ── Link to Existing Event ──
        evt = match_result.matched_event
        evt.article_count += 1
        evt.updated_at = now

        # Update last_updated if article is newer
        evt_dt = parse_date_str(evt.last_updated)
        art_dt = parse_date_str(art_date)
        if art_dt and (not evt_dt or art_dt > evt_dt):
            evt.last_updated = art_date

        # Update entities & topics
        art_ent = extract_entities(f"{article.title}. {article.snippet}")
        merged_entities = list(dict.fromkeys(json.loads(evt.involved_entities or "[]") + art_ent["all_entities"]))
        evt.involved_entities = json.dumps(merged_entities[:20])

        merged_topics = list(dict.fromkeys(json.loads(evt.associated_topics or "[]") + art_ent["keywords"]))
        evt.associated_topics = json.dumps(merged_topics[:15])

        # Sequence number
        seq_num = evt.article_count
        dev_type = classify_development_type(article.title, article.snippet, seq_num)

        # Update event status
        mappings = session.exec(
            select(EventArticleMapping).where(EventArticleMapping.event_id == evt.event_id)
        ).all()
        status_name, status_conf = determine_event_status(
            [{"title": article.title, "snippet": article.snippet}] + [{"title": "", "snippet": ""} for _ in mappings]
        )
        evt.current_status = status_name
        evt.status_confidence = status_conf

        # Create mapping record
        mapping = EventArticleMapping(
            event_id=evt.event_id,
            article_id=article.id,
            publication_date=art_date,
            occurrence_date=art_date,
            event_sequence=seq_num,
            similarity_score=match_result.semantic_score,
            association_confidence=match_result.confidence_score,
            development_type=dev_type,
            created_at=now,
        )
        session.add(evt)
        session.add(mapping)
        session.commit()
        session.refresh(evt)
        session.refresh(mapping)

        log.info(
            "Appended Article '%s' -> Existing Event '%s' [Seq #%d, Type: %s]",
            article.title[:45], evt.event_title[:45], seq_num, dev_type
        )
        return evt, mapping

    else:
        # ── Seed a New Event Cluster ──
        event_id = f"evt_{uuid.uuid4().hex[:10]}"
        art_ent = extract_entities(f"{article.title}. {article.snippet}")
        primary_loc = art_ent["locations"][0] if art_ent["locations"] else None

        dev_type = "Initial announcement"
        evt = NewsEvent(
            event_id=event_id,
            event_title=article.title,
            event_description=article.snippet[:300],
            event_category=article.category,
            associated_topics=json.dumps(art_ent["keywords"][:15]),
            involved_entities=json.dumps(art_ent["all_entities"][:20]),
            primary_location=primary_loc,
            event_start_date=art_date,
            last_updated=art_date,
            current_status="Emerging",
            status_confidence=0.85,
            article_count=1,
            created_at=now,
            updated_at=now,
        )
        session.add(evt)
        session.commit()
        session.refresh(evt)

        mapping = EventArticleMapping(
            event_id=event_id,
            article_id=article.id,
            publication_date=art_date,
            occurrence_date=art_date,
            event_sequence=1,
            similarity_score=1.0,
            association_confidence=1.0,
            development_type=dev_type,
            created_at=now,
        )
        session.add(mapping)

        # If related event detected (but not same event), record explicit EventRelationship
        if match_result.match_type in ("related_event", "same_topic_diff_event") and match_result.matched_event:
            rel = EventRelationship(
                source_event_id=event_id,
                target_event_id=match_result.matched_event.event_id,
                relationship_type="Related event" if match_result.match_type == "related_event" else "Parallel jurisdiction",
                confidence=match_result.confidence_score,
                description=match_result.reason,
                created_at=now,
            )
            session.add(rel)

        session.commit()
        session.refresh(evt)
        session.refresh(mapping)

        log.info(
            "Created New Event Cluster '%s' [%s] from Article '%s'",
            evt.event_title[:45], evt.event_category, article.title[:45]
        )
        return evt, mapping


def cluster_all_unassigned_articles(session: Session) -> Dict[str, int]:
    """
    Sweep across all stored news articles in database and cluster any
    that have not yet been associated with an event. Idempotent.
    """
    all_articles = session.exec(select(NewsArticle).order_by(NewsArticle.published_at.asc())).all()
    mapped_article_ids = set(session.exec(select(EventArticleMapping.article_id)).all())

    new_assigned = 0
    new_events = 0
    existing_events = session.exec(select(NewsEvent)).all()

    for art in all_articles:
        if art.id not in mapped_article_ids:
            evt, mapping = cluster_article_incrementally(art, session)
            mapped_article_ids.add(art.id)
            if mapping.event_sequence == 1:
                new_events += 1
            else:
                new_assigned += 1

    return {
        "total_articles": len(all_articles),
        "newly_clustered_articles": new_assigned,
        "newly_created_events": new_events,
        "total_events": len(session.exec(select(NewsEvent)).all()),
    }


# ── Chronological Timeline & Evolution Summaries ─────────────────────────────

def get_event_timeline(event_id: str, session: Session) -> List[Dict[str, Any]]:
    """
    Generate sorted chronological timeline for a tracked event.
    Preserves exact source citations and distinguishing development types.
    """
    stmt = (
        select(EventArticleMapping, NewsArticle)
        .where(EventArticleMapping.event_id == event_id)
        .where(EventArticleMapping.article_id == NewsArticle.id)
    )
    records = session.exec(stmt).all()

    timeline = []
    for mapping, article in records:
        timeline.append({
            "mapping_id": mapping.id,
            "article_id": article.id,
            "headline": article.title,
            "publication_date": mapping.publication_date,
            "occurrence_date": mapping.occurrence_date or mapping.publication_date,
            "date_basis": "occurrence_date" if mapping.occurrence_date else "publication_date",
            "source": article.source,
            "url": article.url,
            "snippet": article.snippet,
            "event_sequence": mapping.event_sequence,
            "development_type": mapping.development_type,
            "development_description": f"[{mapping.development_type}] {article.title}",
            "association_confidence": round(mapping.association_confidence, 3),
        })

    # Chronological sort: first by occurrence/pub date, then sequence
    timeline.sort(key=lambda x: (x["occurrence_date"] or x["publication_date"], x["event_sequence"]))
    return timeline


def generate_event_evolution_summary(event_id: str, session: Session) -> Dict[str, Any]:
    """
    Construct an evidence-grounded evolution summary of how the event evolved over time.
    Separates genesis, milestones, policy shifts, corrections/conflicts, and current status.
    """
    evt = session.exec(select(NewsEvent).where(NewsEvent.event_id == event_id)).first()
    if not evt:
        return {}

    timeline = get_event_timeline(event_id, session)
    if not timeline:
        return {
            "event_id": evt.event_id,
            "title": evt.event_title,
            "status": evt.current_status,
            "summary": evt.event_description,
        }

    # Genesis
    first_item = timeline[0]
    genesis = {
        "headline": first_item["headline"],
        "date": first_item["occurrence_date"],
        "source": first_item["source"],
        "snippet": first_item["snippet"],
    }

    # Latest Development
    latest_item = timeline[-1]
    latest = {
        "headline": latest_item["headline"],
        "date": latest_item["occurrence_date"],
        "source": latest_item["source"],
        "snippet": latest_item["snippet"],
        "development_type": latest_item["development_type"],
    }

    # Milestones & Turning Points
    milestones = [
        item for item in timeline
        if item["development_type"] in ("Policy modification", "Official response", "Investigation", "Legal action", "Resolution")
    ]

    # Corrections & Clarifications
    corrections = [
        item for item in timeline
        if item["development_type"] in ("Correction", "Clarification")
    ]

    # Conflicts & Controversies
    conflicts = [
        item for item in timeline
        if item["development_type"] == "Conflict or controversy"
    ]

    # Construct narrative synthesis
    g_date = genesis.get("date") or genesis.get("publication_date") or "Initial report"
    l_date = latest.get("date") or latest.get("publication_date") or "Latest report"

    narrative_parts = [
        f"**Origin ({g_date})**: {genesis['headline']} (Source: {genesis['source']}).",
    ]
    if milestones:
        m_date = milestones[0].get("occurrence_date") or milestones[0].get("publication_date") or "Reported"
        narrative_parts.append(
            f"**Key Milestone ({m_date})**: {milestones[0]['development_type']} reported — {milestones[0]['headline']}."
        )
    if corrections:
        c_date = corrections[0].get("occurrence_date") or corrections[0].get("publication_date") or "Reported"
        narrative_parts.append(
            f"**Subsequent Clarification/Correction ({c_date})**: {corrections[0]['headline']}."
        )
    if conflicts:
        narrative_parts.append(
            f"**Disputed Claims/Controversy**: Conflicting reporting noted in {conflicts[0]['headline']}."
        )
    if len(timeline) > 1:
        narrative_parts.append(
            f"**Latest Reported Development ({l_date})**: {latest['headline']} ({latest['development_type']} via {latest['source']})."
        )
    narrative_parts.append(f"**Current Assessed Status**: {evt.current_status} (Confidence: {int(evt.status_confidence*100)}%).")

    synthesized_summary = " ".join(narrative_parts)

    # Persist updated summary to event record
    evt.event_summary = synthesized_summary
    session.add(evt)
    session.commit()

    return {
        "event_id": evt.event_id,
        "title": evt.event_title,
        "category": evt.event_category,
        "current_status": evt.current_status,
        "status_confidence": evt.status_confidence,
        "start_date": evt.event_start_date,
        "last_updated": evt.last_updated,
        "article_count": len(timeline),
        "genesis": genesis,
        "latest_development": latest,
        "milestones_count": len(milestones),
        "corrections_count": len(corrections),
        "conflicts_count": len(conflicts),
        "synthesized_narrative": synthesized_summary,
        "timeline": timeline,
    }
