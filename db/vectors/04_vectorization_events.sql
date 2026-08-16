-- =====================================================================
-- 04_vectorization_events.sql
-- Postgres 16 / pgvector — database quant_ml_vectors, schema bias_research
--
-- SCRIPT IDEMPOTENTE: se puede correr las veces que haga falta.
-- SOLO LO NUEVO del punto #II.1: el registro round robin de lo que va
-- pasando adentro de la corrida de vectorizacion (que archivo va y
-- cuantos van). No toca nada de lo anterior.
--
-- Round robin quiere decir dos cosas, y las dos se aplican al insertar:
--   1) lo de dias anteriores se borra;
--   2) se guardan como maximo N registros, los mas nuevos.
-- Asi la tabla nunca crece: es una ventana de lo que esta pasando, no
-- un historico. El historico de corridas sigue siendo vectorization_runs.
--
-- Corre asi:
--   psql -h localhost -p 5433 -U bias_research -d quant_ml_vectors -f 04_vectorization_events.sql
-- =====================================================================

SET search_path TO bias_research, public;

-- Si otra sesion tiene tomada la tabla, corta a los 5 segundos en vez de
-- quedarse colgado esperando.
SET lock_timeout = '5s';

-- ---------------------------------------------------------------------
-- 1) La tabla
--    event_type:
--      RUN_START   arranca la corrida (total = archivos a procesar)
--      FILE_START  empieza un archivo (position / total)
--      FILE_DONE   archivo vectorizado (chunks = cuantos quedaron)
--      FILE_SKIP   archivo ya vectorizado, no se volvio a encodear
--      FILE_FAIL   archivo que fallo
--      RUN_END     termina la corrida
--      INFO        cualquier otra cosa que valga la pena ver
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vectorization_run_events (
    event_id    BIGSERIAL PRIMARY KEY,
    run_id      BIGINT NULL,
    job_id      VARCHAR(60) NULL,
    log_date    DATE NOT NULL DEFAULT CURRENT_DATE,
    event_type  VARCHAR(20) NOT NULL,
    sector_code VARCHAR(60) NULL,
    portfolio   VARCHAR(60) NULL,
    symbol      VARCHAR(20) NULL,
    file_name   TEXT NULL,
    report_type VARCHAR(10) NULL,
    fiscal_year INT NULL,
    quarter     VARCHAR(4) NULL DEFAULT '',
    position    INT NULL,
    total       INT NULL,
    chunks      INT NULL,
    elapsed_sec NUMERIC(10,2) NULL,
    message     TEXT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- 2) Indices. La pantalla lee siempre por corrida, por sector o por
--    fecha, y siempre lo mas nuevo primero.
-- ---------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_vect_events_created
    ON vectorization_run_events (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_vect_events_run
    ON vectorization_run_events (run_id, event_id DESC);

CREATE INDEX IF NOT EXISTS idx_vect_events_sector
    ON vectorization_run_events (sector_code);

CREATE INDEX IF NOT EXISTS idx_vect_events_date
    ON vectorization_run_events (log_date);

-- ---------------------------------------------------------------------
-- 3) Vistas
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS v_vectorization_run_progress;

-- Por donde va cada corrida: cuantos archivos lleva de cuantos, cual fue
-- el ultimo y hace cuanto. Es lo que la solapa Run History muestra al
-- lado de cada corrida.
CREATE OR REPLACE VIEW v_vectorization_run_progress AS
SELECT e.run_id,
       MAX(e.total)                                        AS total_files,
       COUNT(*) FILTER (WHERE e.event_type = 'FILE_DONE')  AS files_done,
       COUNT(*) FILTER (WHERE e.event_type = 'FILE_SKIP')  AS files_skipped,
       COUNT(*) FILTER (WHERE e.event_type = 'FILE_FAIL')  AS files_failed,
       COALESCE(SUM(e.chunks), 0)::bigint                  AS chunks,
       MAX(e.position)                                     AS last_position,
       MAX(e.created_at)                                   AS last_event_at,
       -- El ultimo symbol y archivo con nombre: el evento de cierre no trae
       -- ninguno de los dos, y sin el filtro la pantalla mostraria un guion
       -- justo cuando la corrida termina.
       (ARRAY_AGG(e.symbol    ORDER BY e.event_id DESC)
        FILTER (WHERE e.symbol IS NOT NULL))[1]              AS last_symbol,
       (ARRAY_AGG(e.file_name ORDER BY e.event_id DESC)
        FILTER (WHERE e.file_name IS NOT NULL))[1]           AS last_file_name,
       (ARRAY_AGG(e.event_type ORDER BY e.event_id DESC))[1] AS last_event_type
  FROM vectorization_run_events e
 WHERE e.run_id IS NOT NULL
 GROUP BY e.run_id;

-- ---------------------------------------------------------------------
-- 4) Chequeo
-- ---------------------------------------------------------------------
SELECT table_name AS objeto_creado
  FROM information_schema.tables
 WHERE table_schema = 'bias_research'
   AND table_name = 'vectorization_run_events';

SELECT table_name AS vista_creada
  FROM information_schema.views
 WHERE table_schema = 'bias_research'
   AND table_name = 'v_vectorization_run_progress';

SELECT COUNT(*) AS eventos_guardados FROM vectorization_run_events;
