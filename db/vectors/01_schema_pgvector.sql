-- =====================================================================
-- Quant_ML - Filing vectorization schema (PostgreSQL 16 + pgvector)
-- Database : quant_ml_vectors
-- Schema   : bias_research
--
-- Apply with:
--   psql -h localhost -p 5433 -U bias_research -d quant_ml_vectors -f 01_schema_pgvector.sql
--
-- EMBEDDING_DIM is fixed at 768, which covers all-mpnet-base-v2 and
-- bge-base-en-v1.5. A 384-dim model (bge-small) does NOT fit this column:
-- the data access layer rejects it instead of silently truncating.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS bias_research;

-- public stays on the path because the vector type lives there; without it,
-- every vector(768) column fails with 'type "vector" does not exist'.
SET search_path TO bias_research, public;

-- ---------------------------------------------------------------------
-- One row per filing file that was vectorized.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS filing_documents CASCADE;

CREATE TABLE filing_documents (
    document_id      BIGSERIAL PRIMARY KEY,
    symbol           VARCHAR(20)   NOT NULL,
    cik              BIGINT        NULL,
    report_type      VARCHAR(10)   NOT NULL,          -- K10 / Q10
    fiscal_year      INT           NOT NULL,
    quarter          VARCHAR(5)    NOT NULL DEFAULT '',-- '' for K10, Q1/Q2/Q3 for Q10
    portfolio        VARCHAR(100)  NULL,
    sector_code      VARCHAR(50)   NULL,
    source_folder    VARCHAR(300)  NULL,
    file_name        VARCHAR(300)  NOT NULL,
    file_path        TEXT          NULL,
    content_hash     VARCHAR(64)   NULL,              -- sha256 of the raw file (CHAR would space-pad and break equality)
    section_count    INT           NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ   NOT NULL DEFAULT now(),e
    CONSTRAINT uq_filing_document UNIQUE (symbol, report_type, fiscal_year, quarter, file_name)
);

CREATE INDEX idx_filing_documents_lookup
    ON filing_documents (report_type, fiscal_year, quarter, symbol);

CREATE INDEX idx_filing_documents_sector
    ON filing_documents (sector_code);

-- ---------------------------------------------------------------------
-- One row per chunk. The embedding model is part of the key on purpose:
-- chunking itself depends on the model, so two models produce two
-- independent sets of chunks over the same document and can be compared.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS filing_chunks CASCADE;

CREATE TABLE filing_chunks (
    chunk_id         BIGSERIAL PRIMARY KEY,
    document_id      BIGINT        NOT NULL REFERENCES filing_documents (document_id) ON DELETE CASCADE,
    embedding_model  VARCHAR(120)  NOT NULL,
    section_label    VARCHAR(60)   NOT NULL,          -- ITEM 7 - MD&A, ITEM 1A - RISK FACTORS, ...
    chunk_index      INT           NOT NULL,          -- order within the document
    chunk_text       TEXT          NOT NULL,
    word_count       INT           NOT NULL DEFAULT 0,
    embedding        vector(768)   NOT NULL,
    created_at       TIMESTAMPTZ   NOT NULL DEFAULT now(),
    CONSTRAINT uq_filing_chunk UNIQUE (document_id, embedding_model, chunk_index)
);

CREATE INDEX idx_filing_chunks_document
    ON filing_chunks (document_id, embedding_model);

CREATE INDEX idx_filing_chunks_section
    ON filing_chunks (section_label);

-- HNSW gives good recall without needing the table populated first.
-- Cosine distance, because every embedding is L2-normalized before insert.
CREATE INDEX idx_filing_chunks_embedding
    ON filing_chunks USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- Run log. This is what makes the job resumable and auditable.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS vectorization_runs CASCADE;

CREATE TABLE vectorization_runs (
    run_id           BIGSERIAL PRIMARY KEY,
    job_id           VARCHAR(60)   NULL,
    portfolio        VARCHAR(100)  NULL,
    sector_code      VARCHAR(50)   NULL,
    report_type      VARCHAR(10)   NOT NULL,
    fiscal_year      INT           NOT NULL,
    quarter          VARCHAR(5)    NOT NULL DEFAULT '',
    embedding_model  VARCHAR(120)  NOT NULL,
    status           VARCHAR(20)   NOT NULL DEFAULT 'STARTED', -- STARTED/FINISHED/ERROR
    files_found      INT           NOT NULL DEFAULT 0,
    files_processed  INT           NOT NULL DEFAULT 0,
    files_skipped    INT           NOT NULL DEFAULT 0,
    files_failed     INT           NOT NULL DEFAULT 0,
    chunks_persisted INT           NOT NULL DEFAULT 0,
    error_message    TEXT          NULL,
    started_at       TIMESTAMPTZ   NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ   NULL
);

CREATE INDEX idx_vectorization_runs_lookup
    ON vectorization_runs (report_type, fiscal_year, quarter, embedding_model);

-- ---------------------------------------------------------------------
-- Convenience view: chunk plus its document context, for semantic search.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_filing_chunks AS
SELECT c.chunk_id,
       c.embedding_model,
       c.section_label,
       c.chunk_index,
       c.chunk_text,
       c.word_count,
       c.embedding,
       d.document_id,
       d.symbol,
       d.cik,
       d.report_type,
       d.fiscal_year,
       d.quarter,
       d.sector_code,
       d.file_name
FROM filing_chunks c
JOIN filing_documents d ON d.document_id = c.document_id;
