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
    inferred_interests_json: str = Field(default="[]")     # JSON: [{category, score, confidence, count, ...}]
    excluded_interests_json: str = Field(default="[]")     # JSON: ["Sports", ...] topics user removed/avoided
    preferred_categories_json: str = Field(default="[]")   # JSON: ["Technology", "Economy", ...]
    preferred_perspectives_json: str = Field(default="[]") # JSON: ["Technical", "Economic", ...]
    preferred_language: str = Field(default="English")
    preferred_length: str = Field(default="detailed")
    last_profile_update: datetime = Field(default_factory=utc_now)


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
    explicit_interests: List[str]
    inferred_interests: List[Dict[str, Any]]
    excluded_interests: List[str]
    preferred_categories: List[str]
    preferred_perspectives: List[str]
    preferred_language: str
    preferred_length: str
    last_profile_update: str
    summary_text: str = ""


class PersonalizationProfileUpdate(BaseModel):
    enabled: Optional[bool] = None
    preferred_categories: Optional[List[str]] = None
    preferred_perspectives: Optional[List[str]] = None
    preferred_language: Optional[str] = None
    preferred_length: Optional[str] = None
    excluded_interests: Optional[List[str]] = None

