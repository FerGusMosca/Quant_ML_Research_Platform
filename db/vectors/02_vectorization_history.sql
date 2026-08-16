-- =====================================================================
-- 02_vectorization_history.sql
-- Postgres 16 / pgvector — database quant_ml_vectors, schema bias_research
--
-- SCRIPT IDEMPOTENTE: se puede correr las veces que haga falta.
-- NO borra nada. Solo agrega columnas, indices y vistas.
--
-- Corre asi:
--   psql -h localhost -p 5433 -U bias_research -d quant_ml_vectors -f 02_vectorization_history.sql
-- =====================================================================

SET search_path TO bias_research, public;

-- ---------------------------------------------------------------------
-- 1) Columnas nuevas en vectorization_runs
--    run_source: AUTO   = la escribio el job de vectorizacion
--                MANUAL = la cargo el usuario a mano desde la pantalla
--    Las corridas viejas no tienen registro, por eso hace falta MANUAL.
-- ---------------------------------------------------------------------
ALTER TABLE vectorization_runs
    ADD COLUMN IF NOT EXISTS run_source VARCHAR(10) NOT NULL DEFAULT 'AUTO';

ALTER TABLE vectorization_runs
    ADD COLUMN IF NOT EXISTS symbols_csv TEXT NULL;

ALTER TABLE vectorization_runs
    ADD COLUMN IF NOT EXISTS notes TEXT NULL;

-- ---------------------------------------------------------------------
-- 2) Indices de lectura para la pantalla
-- ---------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_vectorization_runs_sector
    ON vectorization_runs (sector_code);

CREATE INDEX IF NOT EXISTS idx_vectorization_runs_started
    ON vectorization_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_vectorization_runs_source
    ON vectorization_runs (run_source);

CREATE INDEX IF NOT EXISTS idx_filing_documents_symbol
    ON filing_documents (symbol);

-- ---------------------------------------------------------------------
-- 3) Vistas. Se dropean primero porque CREATE OR REPLACE VIEW no puede
--    cambiarle el tipo a una columna: si el script ya corrio con otra
--    version, el replace falla. Dropear una vista no toca ningun dato.
--    El orden importa: primero las que dependen de la otra.
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS v_vectorization_by_symbol;
DROP VIEW IF EXISTS v_vectorization_by_sector;
DROP VIEW IF EXISTS v_vectorization_storage;

-- ---------------------------------------------------------------------
-- 3.a) Vista de peso de vectores.
--    Es exactamente el query del punto #1.b del documento, con el
--    contexto extra que hace falta para poder filtrar por sector,
--    modelo, tipo de reporte y anio.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_vectorization_storage AS
SELECT d.document_id,
       d.symbol,
       d.file_name,
       d.report_type,
       d.fiscal_year,
       d.quarter,
       d.sector_code,
       d.portfolio,
       c.embedding_model,
       COUNT(*)                                              AS chunks,
       SUM(pg_column_size(c.embedding))::bigint              AS bytes,
       pg_size_pretty(SUM(pg_column_size(c.embedding))::bigint) AS pretty_size,
       MIN(c.created_at)                                     AS first_chunk_at,
       MAX(c.created_at)                                     AS last_chunk_at
  FROM filing_chunks c
  JOIN filing_documents d ON d.document_id = c.document_id
 GROUP BY d.document_id, d.symbol, d.file_name, d.report_type, d.fiscal_year,
          d.quarter, d.sector_code, d.portfolio, c.embedding_model;

-- ---------------------------------------------------------------------
-- 3.b) Resumen por security. Es lo que abre la pantalla cuando elegis
--    un symbol.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_vectorization_by_symbol AS
SELECT symbol,
       sector_code,
       embedding_model,
       COUNT(DISTINCT document_id) AS documents,
       SUM(chunks)::bigint         AS chunks,
       SUM(bytes)::bigint          AS bytes,
       pg_size_pretty(SUM(bytes)::bigint) AS pretty_size,
       MIN(fiscal_year)            AS first_year,
       MAX(fiscal_year)            AS last_year,
       MAX(last_chunk_at)          AS last_vectorized_at
  FROM v_vectorization_storage
 GROUP BY symbol, sector_code, embedding_model;

-- ---------------------------------------------------------------------
-- 3.c) Resumen por sector. Es lo que abre la pantalla cuando elegis
--    un sector (el campo SECTOR de la pantalla de Document Tagger).
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_vectorization_by_sector AS
SELECT COALESCE(sector_code, 'UNCLASSIFIED') AS sector_code,
       embedding_model,
       COUNT(DISTINCT symbol)      AS securities,
       COUNT(DISTINCT document_id) AS documents,
       SUM(chunks)::bigint         AS chunks,
       SUM(bytes)::bigint          AS bytes,
       pg_size_pretty(SUM(bytes)::bigint) AS pretty_size,
       MAX(last_chunk_at)          AS last_vectorized_at
  FROM v_vectorization_storage
 GROUP BY COALESCE(sector_code, 'UNCLASSIFIED'), embedding_model;

-- ---------------------------------------------------------------------
-- 4) Chequeo final
-- ---------------------------------------------------------------------
SELECT 'vectorization_runs.run_source' AS objeto,
       COUNT(*) AS existe
  FROM information_schema.columns
 WHERE table_schema = 'bias_research'
   AND table_name   = 'vectorization_runs'
   AND column_name  = 'run_source';

SELECT table_name AS vista_creada
  FROM information_schema.views
 WHERE table_schema = 'bias_research'
   AND table_name IN ('v_vectorization_storage',
                      'v_vectorization_by_symbol',
                      'v_vectorization_by_sector')
 ORDER BY table_name;
