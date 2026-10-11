import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Default to a local postgres instance. 
# Ideally, this should be loaded from environment variables.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ohxeuvvhflkfugiakbul:***REMOVED_SUPABASE_ENV_VALUE***@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
