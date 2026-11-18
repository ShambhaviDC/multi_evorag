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
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            # message table
            msg_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(message)")).fetchall()]
            if msg_cols and "sources_json" not in msg_cols:
                conn.execute(text("ALTER TABLE message ADD COLUMN sources_json TEXT"))
                conn.commit()
            if msg_cols and "components_json" not in msg_cols:
                conn.execute(text("ALTER TABLE message ADD COLUMN components_json TEXT"))
                conn.commit()

            # userpreference table
            up_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(userpreference)")).fetchall()]
            if up_cols and "preferred_categories_json" not in up_cols:
                conn.execute(text("ALTER TABLE userpreference ADD COLUMN preferred_categories_json TEXT DEFAULT '[]'"))
                conn.commit()
            if up_cols and "preferred_subcategories_json" not in up_cols:
                conn.execute(text("ALTER TABLE userpreference ADD COLUMN preferred_subcategories_json TEXT DEFAULT '[]'"))
                conn.commit()

            # personalizationprofile table
            pp_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(personalizationprofile)")).fetchall()]
            if pp_cols and "onboarding_completed" not in pp_cols:
                conn.execute(text("ALTER TABLE personalizationprofile ADD COLUMN onboarding_completed BOOLEAN DEFAULT 0"))
                conn.commit()
            if pp_cols and "preferred_subcategories_json" not in pp_cols:
                conn.execute(text("ALTER TABLE personalizationprofile ADD COLUMN preferred_subcategories_json TEXT DEFAULT '[]'"))
                conn.commit()
    except Exception as e:
        pass


def get_session() -> Session:
    """FastAPI dependency that provides a DB session.

    Usage::

        @app.get("/items")
        def read_items(session: Session = Depends(get_session)):
            ...
    """
    with Session(engine) as session:
        yield session
