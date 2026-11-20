'''SQLModel definitions for EvoRAG application.

This module defines the database tables used for user authentication,
user preferences, conversations, and messages. It uses SQLModel, which
combines Pydantic models with SQLAlchemy core for easy ORM operations.
'''

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlmodel import Field, SQLModel

# Re-export Phase 7 query and adaptive models for unified access
from query_models import (
    EntityItem,
    RelationshipItem,
    QueryConstraints,
    OutputPlan,
    StructuredQuery,
    UIComponent,
    AdaptiveConsensusResponse,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    """User account."""
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, nullable=False, unique=True)
    full_name: Optional[str] = Field(default=None)
    hashed_password: Optional[str] = Field(default=None, nullable=True)
    google_sub: Optional[str] = Field(default=None, nullable=True, index=True)
    reset_token: Optional[str] = Field(default=None, nullable=True, index=True)
    created_at: datetime = Field(default_factory=utc_now)


class UserPreference(SQLModel, table=True):
    """User onboarding and personalization preferences."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    age_range: Optional[str] = Field(default=None)
    role: Optional[str] = Field(default=None)
    interests: Optional[str] = Field(default=None)  # comma-separated list
    response_style: Optional[str] = Field(default=None)
    preferred_categories_json: Optional[str] = Field(default="[]")
    preferred_subcategories_json: Optional[str] = Field(default="[]")
    onboarding_completed: bool = Field(default=False)
    updated_at: datetime = Field(default_factory=utc_now, sa_column_kwargs={"onupdate": utc_now})


class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    title: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now, sa_column_kwargs={"onupdate": utc_now})


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    role: str = Field(index=True)  # "user" or "assistant"
    content: str
    sources_json: Optional[str] = Field(default=None, nullable=True)
    components_json: Optional[str] = Field(default=None, nullable=True)
    timestamp: datetime = Field(default_factory=utc_now)


class PersonalizationProfile(SQLModel, table=True):
    """Dynamic User Personalization Profile combining explicit and AI-inferred interests."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", unique=True, index=True)
    enabled: bool = Field(default=True)
    onboarding_completed: bool = Field(default=False)
    inferred_interests_json: str = Field(default="[]")     # JSON: [{category, score, confidence, count, ...}]
    excluded_interests_json: str = Field(default="[]")     # JSON: ["Sports", ...] topics user removed/avoided
    preferred_categories_json: str = Field(default="[]")   # JSON: ["Technology", "Economy", ...]
    preferred_subcategories_json: str = Field(default="[]") # JSON: ["AI", "Startups", ...]
    preferred_perspectives_json: str = Field(default="[]") # JSON: ["Technical", "Economic", ...]
    preferred_language: str = Field(default="English")
    preferred_length: str = Field(default="detailed")
    learned_preferences_json: str = Field(default="{}")    # JSON: learned style preferences from repeated feedback
    last_profile_update: datetime = Field(default_factory=utc_now)


class NewsArticle(SQLModel, table=True):
    """Real news article record for multi-category discovery, autocomplete, and personalization."""
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    url: str = Field(unique=True, index=True)
    url_hash: str = Field(index=True)
    source: str = Field(index=True)
    category: str = Field(index=True)
    subcategory: Optional[str] = Field(default=None, index=True)
    published_at: str = Field(index=True)
    snippet: str
    image_url: Optional[str] = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utc_now)


# ── Temporal News Event Tracking & Evolution Models ──────────────────────────

class NewsEvent(SQLModel, table=True):
    """Tracked real-world news event evolving over time across multiple articles."""
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(unique=True, index=True)
    event_title: str = Field(index=True)
    event_description: str
    event_category: str = Field(index=True)
    associated_topics: str = Field(default="[]")      # JSON list of sub-topics
    involved_entities: str = Field(default="[]")      # JSON list of named entities (people, orgs, places)
    primary_location: Optional[str] = Field(default=None, nullable=True)
    event_start_date: Optional[str] = Field(default=None, index=True)
    last_updated: Optional[str] = Field(default=None, index=True)
    current_status: str = Field(default="Ongoing", index=True)  # Emerging, Ongoing, Developing, Resolved, Dormant, Unknown
    status_confidence: float = Field(default=0.8)
    event_summary: Optional[str] = Field(default=None, nullable=True)
    article_count: int = Field(default=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now, sa_column_kwargs={"onupdate": utc_now})


class EventArticleMapping(SQLModel, table=True):
    """Mapping between an evolving event and a specific news article with chronological sequence."""
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(index=True)
    article_id: int = Field(foreign_key="newsarticle.id", index=True)
    publication_date: str = Field(index=True)
    occurrence_date: Optional[str] = Field(default=None, nullable=True)
    event_sequence: int = Field(default=1)
    similarity_score: float = Field(default=1.0)
    association_confidence: float = Field(default=1.0)
    development_type: str = Field(default="New development")  # Initial announcement, Policy modification, Correction, Conflict, etc.
    created_at: datetime = Field(default_factory=utc_now)


class EventRelationship(SQLModel, table=True):
    """Explicit evidence-supported relationship between distinct events."""
    id: Optional[int] = Field(default=None, primary_key=True)
    source_event_id: str = Field(index=True)
    target_event_id: str = Field(index=True)
    relationship_type: str = Field(index=True)  # Related event, Follow-up, Consequence of, Part of broader event, Supersedes
    confidence: float = Field(default=0.8)
    description: Optional[str] = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utc_now)


class EventTimelineItem(BaseModel):
    mapping_id: Optional[int] = None
    article_id: int
    headline: str
    publication_date: str
    occurrence_date: Optional[str] = None
    source: str
    url: str
    development_type: str
    development_description: str
    association_confidence: float = 1.0


class NewsEventRead(BaseModel):
    id: Optional[int] = None
    event_id: str
    event_title: str
    event_description: str
    event_category: str
    associated_topics: List[str] = []
    involved_entities: List[str] = []
    primary_location: Optional[str] = None
    event_start_date: Optional[str] = None
    last_updated: Optional[str] = None
    current_status: str
    status_confidence: float
    event_summary: Optional[str] = None
    article_count: int
    created_at: datetime
    updated_at: datetime
    timeline: List[EventTimelineItem] = []
    relationships: List[Dict[str, Any]] = []


class NewsArticleRead(BaseModel):
    id: Optional[int] = None
    title: str
    url: str
    source: str
    category: str
    subcategory: Optional[str] = None
    published_at: str
    snippet: str
    image_url: Optional[str] = None


class AutocompleteSuggestion(BaseModel):
    text: str
    category: str
    type: str  # "category", "subcategory", "entity", "headline", "trending"
    preview: Optional[str] = None


class InferredInterestItem(BaseModel):
    category: str
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    supporting_interactions: int = Field(default=1)
    first_detected: str
    last_interaction: str
    status: str = "emerging"  # emerging, recurring, strong, declining
    source: str = "chat_history"


class PersonalizationProfileRead(BaseModel):
    user_id: int
    enabled: bool
    onboarding_completed: bool = False
    explicit_interests: List[str]
    inferred_interests: List[Dict[str, Any]]
    excluded_interests: List[str]
    preferred_categories: List[str]
    preferred_subcategories: List[str] = []
    preferred_perspectives: List[str]
    preferred_language: str
    preferred_length: str
    last_profile_update: str
    summary_text: str = ""


class PersonalizationProfileUpdate(BaseModel):
    enabled: Optional[bool] = None
    onboarding_completed: Optional[bool] = None
    preferred_categories: Optional[List[str]] = None
    preferred_subcategories: Optional[List[str]] = None
    preferred_perspectives: Optional[List[str]] = None
    preferred_language: Optional[str] = None
    preferred_length: Optional[str] = None
    excluded_interests: Optional[List[str]] = None


# ── Response Feedback & Intelligence Models ───────────────────────────────────

class ResponseFeedback(SQLModel, table=True):
    """User satisfaction signals, issue classifications, and feedback intelligence for assistant responses."""
    __tablename__ = "response_feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    feedback_id: str = Field(unique=True, index=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True, nullable=True)
    session_id: Optional[str] = Field(default=None, index=True, nullable=True)
    conversation_id: Optional[int] = Field(default=None, foreign_key="conversation.id", index=True, nullable=True)
    message_id: str = Field(index=True)  # unique assistant message identifier or query_id
    query_text: Optional[str] = Field(default=None, nullable=True)
    feedback_type: str = Field(index=True)  # "like" or "dislike"
    reason: Optional[str] = Field(default=None, index=True, nullable=True)
    comment: Optional[str] = Field(default=None, nullable=True)
    execution_metadata_json: Optional[str] = Field(default="{}", nullable=True)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now, sa_column_kwargs={"onupdate": utc_now})


class FeedbackCreate(BaseModel):
    message_id: str
    feedback_type: str  # "like" or "dislike"
    conversation_id: Optional[int] = None
    query_text: Optional[str] = None
    reason: Optional[str] = None
    comment: Optional[str] = None
    session_id: Optional[str] = None
    execution_metadata: Optional[Dict[str, Any]] = None


class FeedbackRead(BaseModel):
    feedback_id: str
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    conversation_id: Optional[int] = None
    message_id: str
    query_text: Optional[str] = None
    feedback_type: str
    reason: Optional[str] = None
    comment: Optional[str] = None
    execution_metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


