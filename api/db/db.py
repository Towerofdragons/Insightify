# Import SQLAlchemy

from sqlalchemy import(
  create_engine, Column, Integer, BigInteger, String, Text, DateTime, ForeignKey,
  JSON, Numeric, ARRAY, func)

from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB, ARRAY as PG_ARRAY
from sqlalchemy.sql import text

# For pgvector
from pgvector.sqlalchemy import Vector

# For environment variables
from dotenv import load_dotenv
import os

# Base class for ORM models
Base = declarative_base()

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_URL = os.getenv("DB_URL")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    url = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # ORM: one source has many articles
    articles = relationship("Article", back_populates="source")

class Article(Base):
    __tablename__ = "articles"

    id = Column(BigInteger, primary_key=True)
    source_id = Column(Integer, ForeignKey("sources.id", ondelete="SET NULL"))

    title = Column(Text, nullable=False)
    description = Column(Text)
    url = Column(Text, unique=True, nullable=False)
    author = Column(Text)
    published_at = Column(TIMESTAMP(timezone=True))
    country = Column(String(5))
    language = Column(String(5))

    raw = Column(JSONB)

    keywords = Column(ARRAY(Text))
    categories = Column(ARRAY(Text))
    sentiment = Column(Numeric)
    embedding = Column(Vector(384))

    inserted_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # ORM: back reference
    source = relationship("Source", back_populates="articles")


"""
  Utilities

"""

def init_db():
    """
    Create tables in the database based on the ORM models.
    Run this once when bootstrapping your app.
    """
    Base.metadata.create_all(engine)


def get_session():
    """
    Creates a session and ensures it’s closed.
    Use with context managers in your code.
    """

    Connection_String = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_URL}:{DB_PORT}/{DB_NAME}"

    engine = create_engine(Connection_String, echo=True)

    # CREATE A SESSION OBJECT TO INITIATE QUERY IN DATABASE
    SessionLocal = sessionmaker(bind=engine)

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

