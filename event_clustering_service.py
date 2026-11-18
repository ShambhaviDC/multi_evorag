"""
event_clustering_service.py — Two-Stage Event Identification & Clustering Engine for EvoRAG

Provides:
  1. Configurable Clustering Thresholds (Semantic, Entity Overlap, Temporal Window)
  2. Lightweight Entity & Keyword Extractor (CPU-friendly, no heavy dependencies)
  3. Stage 1: Candidate Event Retrieval using MiniLM semantic embeddings
  4. Stage 2: Multi-Factor Verification (Semantic, Named Entity Overlap, Location, Temporal)
  5. Distinction between Same Event, Related Event, Same Topic Different Event, and Unrelated Event
"""

import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure PyTorch is used and avoid accidental TensorFlow / Keras sandbox access
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sqlmodel import Session, select

from models import NewsArticle, NewsEvent, EventArticleMapping, EventRelationship
from embedder import _get_model

log = logging.getLogger(__name__)

# ── Configurable Thresholds ──────────────────────────────────────────────────

@dataclass
class ClusteringConfig:
    """Configurable hyperparameters for two-stage event clustering."""
    # Stage 1: Minimum semantic cosine similarity to qualify as candidate event
    candidate_similarity_threshold: float = 0.40
    candidate_limit: int = 5

    # Stage 2: Verification thresholds
    same_event_threshold: float = 0.48         # Composite score >= 0.48 -> same evolving event
    related_event_threshold: float = 0.38      # 0.38 <= score < 0.48 -> related event relationship
    min_entity_overlap_for_same_event: float = 0.08  # Must share at least key entities/actors/keywords

    # Temporal window: events active within max_active_days (default 60 days)
    max_event_active_days: int = 60

    # Weights for composite verification score
    weight_semantic: float = 0.50
    weight_entities: float = 0.30
    weight_temporal: float = 0.10
    weight_category: float = 0.10


DEFAULT_CONFIG = ClusteringConfig()


# ── Lightweight Entity, Location & Temporal Extractor ─────────────────────────

# Common global organizations and institutions
KNOWN_ORGS = {
    "government", "white house", "parliament", "congress", "supreme court",
    "european union", "eu", "united nations", "un", "who", "nato", "fbi", "cia",
    "nasa", "isro", "bcci", "icc", "fifa", "uefa", "rbi", "federal reserve", "sec",
    "google", "microsoft", "openai", "apple", "amazon", "meta", "nvidia", "tesla",
    "bbc", "reuters", "cnbc", "espn", "techcrunch", "bloomberg"
}

# Major geographical locations
KNOWN_LOCATIONS = {
    "india", "united states", "usa", "us", "china", "united kingdom", "uk",
    "russia", "ukraine", "germany", "france", "japan", "australia", "canada",
    "washington", "delhi", "new delhi", "london", "beijing", "tokyo", "paris",
    "berlin", "moscow", "kyiv", "california", "new york", "texas", "mumbai"
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract persons, organizations, locations, and key domain terms using
    lightweight CPU regex and pattern matching.
    """
    lower = text.lower()
    
    # 1. Match known organizations
    found_orgs = set()
    for org in KNOWN_ORGS:
        if re.search(r"\b" + re.escape(org) + r"\b", lower):
            found_orgs.add(org.title())

    # 2. Match known locations
    found_locs = set()
    for loc in KNOWN_LOCATIONS:
        if re.search(r"\b" + re.escape(loc) + r"\b", lower):
            found_locs.add(loc.title())

    # 3. Match proper capitalized named entities (People / Titles)
    # Looks for Capitalized Word Pairs e.g., "Sam Altman", "Narendra Modi", "Joe Biden"
    capitalized_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)\b", text)
    found_persons = set()
    stopwords = {"The", "This", "That", "When", "What", "After", "Before", "While", "Read More", "Latest News"}
    for ent in capitalized_entities:
        words = ent.split()
        if words[0] not in stopwords and ent.lower() not in KNOWN_ORGS and ent.lower() not in KNOWN_LOCATIONS:
            if len(words) <= 3:
                found_persons.add(ent)

    # 4. Extract key topical keywords (filtered)
    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", lower)
    stop_tokens = {
        "with", "from", "that", "this", "have", "were", "been", "their", "there",
        "which", "about", "would", "could", "should", "after", "before", "more",
        "also", "than", "first", "other", "into", "over", "some", "them", "these"
    }
    keywords = list(dict.fromkeys([t for t in tokens if t not in stop_tokens]))[:12]

    return {
        "organizations": sorted(list(found_orgs)),
        "locations": sorted(list(found_locs)),
        "persons": sorted(list(found_persons)),
        "keywords": keywords,
        "all_entities": sorted(list(found_orgs | found_locs | found_persons)),
    }


def parse_date_str(date_str: Optional[str]) -> Optional[datetime]:
    """Parse various ISO and RSS date formats into timezone-aware datetime."""
    if not date_str:
        return None
    # Try ISO format
    try:
        clean = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception:
        pass
    # Try RFC 2822 (common in RSS)
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    # Try common strftime formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute standard Jaccard intersection over union."""
    if not set1 and not set2:
        return 0.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


# ── MiniLM Semantic Vector Helper ─────────────────────────────────────────────

def compute_text_embedding(text: str) -> np.ndarray:
    """Compute normalized 384-d vector for a text passage using cached MiniLM model."""
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    dot = float(np.dot(v1, v2))
    return max(0.0, min(1.0, dot))


# ── Two-Stage Event Matching Implementation ──────────────────────────────────

class EventMatchingResult:
    """Result of evaluating an article against candidate events."""
    def __init__(
        self,
        match_type: str,  # "same_event", "related_event", "same_topic_diff_event", "unrelated"
        matched_event: Optional[NewsEvent] = None,
        confidence_score: float = 0.0,
        semantic_score: float = 0.0,
        entity_score: float = 0.0,
        temporal_score: float = 1.0,
        matched_entities: List[str] = None,
        reason: str = "",
    ):
        self.match_type = match_type
        self.matched_event = matched_event
        self.confidence_score = confidence_score
        self.semantic_score = semantic_score
        self.entity_score = entity_score
        self.temporal_score = temporal_score
        self.matched_entities = matched_entities or []
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_type": self.match_type,
            "event_id": self.matched_event.event_id if self.matched_event else None,
            "event_title": self.matched_event.event_title if self.matched_event else None,
            "confidence_score": round(self.confidence_score, 3),
            "semantic_score": round(self.semantic_score, 3),
            "entity_score": round(self.entity_score, 3),
            "temporal_score": round(self.temporal_score, 3),
            "matched_entities": self.matched_entities,
            "reason": self.reason,
        }


def match_article_to_event(
    article: NewsArticle,
    existing_events: List[NewsEvent],
    config: ClusteringConfig = DEFAULT_CONFIG,
    article_vec: Optional[np.ndarray] = None,
    event_vectors: Optional[Dict[str, np.ndarray]] = None,
) -> EventMatchingResult:
    """
    Two-Stage Event Matching:
      Stage 1: Filter candidate events within same or compatible category using semantic similarity.
      Stage 2: Strict verification using entity overlap, location compatibility, and temporal windowing.
               Ensures distinct event identity within broader topics (e.g. US AI Policy != EU AI Act).
    """
    if not existing_events:
        return EventMatchingResult(
            match_type="unrelated",
            reason="No existing events in database. Create first event cluster."
        )

    # Prepare article features
    art_text = f"{article.title}. {article.snippet}"
    art_features = extract_entities(art_text)
    art_entities_set = {e.lower() for e in art_features["all_entities"]}
    art_keywords_set = set(art_features["keywords"])
    art_date = parse_date_str(article.published_at)

    if article_vec is None:
        article_vec = compute_text_embedding(art_text)

    # ── Stage 1: Candidate Retrieval ──────────────────────────────────────────
    candidates: List[Tuple[NewsEvent, float]] = []
    for evt in existing_events:
        # Category filtering: match only if same category or cross-topic compatibility
        if evt.event_category.lower() != article.category.lower():
            continue

        evt_vec = (event_vectors or {}).get(evt.event_id)
        if evt_vec is None:
            evt_text = f"{evt.event_title}. {evt.event_description}"
            evt_vec = compute_text_embedding(evt_text)
            if event_vectors is not None:
                event_vectors[evt.event_id] = evt_vec

        sim = cosine_similarity(article_vec, evt_vec)
        if sim >= config.candidate_similarity_threshold:
            candidates.append((evt, sim))

    # Sort candidates by semantic similarity descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:config.candidate_limit]

    if not top_candidates:
        return EventMatchingResult(
            match_type="unrelated",
            confidence_score=0.0,
            reason=f"No candidate events met initial similarity threshold ({config.candidate_similarity_threshold:.2f})."
        )

    # ── Stage 2: Event Verification ───────────────────────────────────────────
    best_result: Optional[EventMatchingResult] = None
    highest_composite = 0.0

    for candidate_evt, semantic_score in top_candidates:
        evt_entities = set(json.loads(candidate_evt.involved_entities or "[]"))
        evt_entities_lower = {e.lower() for e in evt_entities}
        
        # 1. Entity overlap (Jaccard + actor presence + token overlap + keyword overlap)
        shared_entities = art_entities_set.intersection(evt_entities_lower)
        entity_jaccard = calculate_jaccard_similarity(art_entities_set, evt_entities_lower)
        keyword_jaccard = calculate_jaccard_similarity(art_keywords_set, set(json.loads(candidate_evt.associated_topics or "[]")))

        # Token-level overlap across extracted entities
        art_tokens = {tok for e in art_entities_set for tok in e.split() if len(tok) > 2}
        evt_tokens = {tok for e in evt_entities_lower for tok in e.split() if len(tok) > 2}
        token_jaccard = calculate_jaccard_similarity(art_tokens, evt_tokens)

        # Effective entity score
        entity_score = max(
            entity_jaccard,
            token_jaccard,
            keyword_jaccard,
            (len(shared_entities) / max(len(art_entities_set), 1)) if art_entities_set else 0.0
        )

        # 2. Location compatibility check
        location_compatible = True
        art_locs = {l.lower() for l in art_features["locations"]}
        evt_loc = (candidate_evt.primary_location or "").lower()
        if evt_loc and art_locs and evt_loc not in art_locs:
            # Different explicit locations mentioned (e.g. US vs Europe) -> separate events!
            location_compatible = False
            entity_score *= 0.4  # Penalize non-matching locations

        # 3. Temporal compatibility check
        temporal_score = 1.0
        evt_updated = parse_date_str(candidate_evt.last_updated or candidate_evt.event_start_date)
        if art_date and evt_updated:
            diff_days = abs((art_date - evt_updated).total_seconds()) / (24 * 3600)
            if diff_days > config.max_event_active_days:
                # Decays if beyond temporal active window
                temporal_score = max(0.2, math.exp(-(diff_days - config.max_event_active_days) / 30.0))
            else:
                temporal_score = 1.0

        # Category match score
        category_score = 1.0 if article.category.lower() == candidate_evt.event_category.lower() else 0.5

        # 4. Composite Verification Score
        composite = (
            config.weight_semantic * semantic_score +
            config.weight_entities * entity_score +
            config.weight_temporal * temporal_score +
            config.weight_category * category_score
        )

        if composite > highest_composite:
            highest_composite = composite
            
            # Critical Distinction: Topic Identity vs. Event Identity
            # 1. Incompatible explicit locations (e.g. US vs EU/Brussels) -> separate events within topic!
            if not location_compatible and semantic_score >= 0.45:
                match_type = "same_topic_diff_event"
                reason = f"Shares topic '{candidate_evt.event_category}', but distinct locations/jurisdictions."
            # 2. Confirmed same event: verified composite match OR strong semantic match within same location
            elif composite >= config.same_event_threshold or (semantic_score >= 0.65 and location_compatible):
                match_type = "same_event"
                reason = f"Confirmed same event: composite {composite:.2f}, semantic {semantic_score:.2f}."
            # 3. Related event
            elif composite >= config.related_event_threshold:
                match_type = "related_event"
                reason = f"Related event: composite {composite:.2f} >= {config.related_event_threshold:.2f}."
            # 4. Same broad topic
            elif semantic_score >= 0.45:
                match_type = "same_topic_diff_event"
                reason = f"Shares broad topic '{candidate_evt.event_category}', but distinct event actors."
            else:
                match_type = "unrelated"
                reason = f"Insufficient match: composite {composite:.2f} < {config.related_event_threshold:.2f}."

            best_result = EventMatchingResult(
                match_type=match_type,
                matched_event=candidate_evt,
                confidence_score=composite,
                semantic_score=semantic_score,
                entity_score=entity_score,
                temporal_score=temporal_score,
                matched_entities=list(shared_entities),
                reason=reason,
            )

    return best_result or EventMatchingResult(
        match_type="unrelated",
        reason="Evaluation concluded without confident event match."
    )
