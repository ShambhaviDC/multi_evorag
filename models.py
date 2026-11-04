'''SQLModel definitions for EvoRAG application.

This module defines the database tables used for user authentication,
user preferences, conversations, and messages. It uses SQLModel, which
combines Pydantic models with SQLAlchemy core for easy ORM operations.
''' 

from datetime import datetime, timezone
from typing import List, Optional

from sqlmodel import Field, SQLModel


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
    timestamp: datetime = Field(default_factory=utc_now)
