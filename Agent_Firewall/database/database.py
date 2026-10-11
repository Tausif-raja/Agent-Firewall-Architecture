import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger("agent_firewall.database")

# Default SQLite file lives next to the app (Agent_Firewall/agent_firewall.db), not in
# whatever directory the server happens to be started from.
DEFAULT_SQLITE_URL = "sqlite:///" + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agent_firewall.db"))

DATABASE_URL = os.getenv("DATABASE_URL") or DEFAULT_SQLITE_URL

try:
    if DATABASE_URL.startswith("postgresql"):
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        # Test connection
        with engine.connect() as conn:
            pass
    else:
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
except Exception as e:
    logger.error(f"Could not connect to the configured database ({e.__class__.__name__}). "
                 f"FALLING BACK to local SQLite at {DEFAULT_SQLITE_URL}: data will not reach Postgres.")
    DATABASE_URL = DEFAULT_SQLITE_URL
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

