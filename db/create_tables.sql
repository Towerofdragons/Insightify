-- sql/create_tables.sql
-- Schema for storing news articles for Insightify
-- This script runs automatically when Postgres starts if mounted in
-- docker-entrypoint-initdb.d/

-- Drop old tables (if re-running during development)
DROP TABLE IF EXISTS articles CASCADE;
DROP TABLE IF EXISTS sources CASCADE;

-- ============================================================
-- Sources Table
-- ============================================================
-- Why: Articles can come from multiple APIs (NewsAPI, GDELT, RSS, custom scrapers).
-- Keeping them normalized avoids duplication and allows tracking.
CREATE TABLE sources (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,                   -- e.g., "NewsAPI", "GDELT", "NYTimes RSS"
    url TEXT,                             -- homepage or API URL
    created_at TIMESTAMPTZ DEFAULT NOW()
);


CREATE EXTENSION vector;

-- ============================================================
-- Articles Table
-- ============================================================
-- Stores the raw news content + metadata for later enrichment.
-- Designed for flexibility (both human-readable + raw JSON).
CREATE TABLE articles (
    id BIGSERIAL PRIMARY KEY,
    source_id INT REFERENCES sources(id) ON DELETE SET NULL,
    
    title TEXT NOT NULL,
    description TEXT,
    url TEXT UNIQUE NOT NULL,             -- avoid duplicates on re-ingestion
    author TEXT,
    published_at TIMESTAMPTZ,
    country VARCHAR(5),                   -- optional (ISO country code if available)
    language VARCHAR(5),                  -- optional (ISO language code)
    
    -- Raw payload from the API (for debugging, reproducibility)
    raw JSONB,

    -- Future-proofing: enrichment / AI analysis
    keywords TEXT[],                      -- extracted keywords
    categories TEXT[],                    -- e.g., ["politics", "protest"]
    sentiment NUMERIC,                    -- -1.0 to 1.0 sentiment score
    embedding VECTOR(384),                -- (if using pgvector for NLP later)

    inserted_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Helpful Indexes
-- ============================================================
-- Speeds up queries by publication date (e.g., "latest news").
CREATE INDEX idx_articles_published_at ON articles(published_at DESC);

-- Makes searching text faster (Postgres full-text search).
CREATE INDEX idx_articles_title_description ON articles USING GIN (
    to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,''))
);

-- Quick lookup by keyword tags (array GIN index).
CREATE INDEX idx_articles_keywords ON articles USING GIN (keywords);

-- ============================================================
-- Initial Seed (Optional)
-- ============================================================
INSERT INTO sources (name, url)
VALUES ('NewsAPI', 'https://newsapi.org'),
       ('GDELT', 'https://www.gdeltproject.org');
