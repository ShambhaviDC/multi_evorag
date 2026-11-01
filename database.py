# database.py – SQLModel engine and session utilities

"""Database setup for EvoRAG.

We use a SQLite file (``evorag.db``) stored in the project root. The
module provides a ``get_engine`` function to create the SQLModel engine
and a FastAPI dependency ``get_session`` that yields a ``Session``
instance. The engine is created once at import time to avoid repeated
connections.
"""

import os
from pathlib import Path

from sqlmodel import SQLModel, create_engine, Session
import models  # ensure SQLModel subclasses are registered

# Resolve a path relative to the project root (the current working directory)
DB_PATH = Path(os.getenv("EVORAG_DB_PATH", "evorag.db"))

# SQLite URI – echo can be toggled via env for debugging
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=bool(os.getenv("EVORAG_DB_ECHO")),
    connect_args={"check_same_thread": False},
)


def create_db_and_tables() -> None:
    """Create all tables defined in ``models.py``.

    Called during application startup. ``SQLModel.metadata`` includes the
    table definitions from any imported ``SQLModel`` subclasses.
    """
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """FastAPI dependency that provides a DB session.

    Usage::

        @app.get("/items")
        def read_items(session: Session = Depends(get_session)):
            ...
    """
    with Session(engine) as session:
        yield session
