/* ============================================================================
   db/sec_metadata/01_schema_and_sps.sql
   DB: machine_learning_research
   Idempotente: se puede correr las veces que haga falta.

   1) Columnas nuevas en dbo.SEC_Securities
   2) Tablas de tags
   3) Stored procedures que usan el DAL y la pantalla
   ========================================================================== */

SET NOCOUNT ON;
GO

/* ---------------------------------------------------------------------------
   1) Columnas nuevas
   --------------------------------------------------------------------------- */
IF COL_LENGTH('dbo.SEC_Securities', 'sic_description') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD sic_description VARCHAR(255) NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'sector_code') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD sector_code VARCHAR(20) NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'sector_name') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD sector_name VARCHAR(80) NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'industry_code') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD industry_code VARCHAR(30) NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'industry_name') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD industry_name VARCHAR(120) NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'fiscal_year_end') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD fiscal_year_end VARCHAR(8) NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'state_of_incorporation') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD state_of_incorporation VARCHAR(8) NULL;
GO
/* meta_status: PENDING / OK / ERROR / NOT_FOUND -> es el checkpoint del job */
IF COL_LENGTH('dbo.SEC_Securities', 'meta_status') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD meta_status VARCHAR(20) NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'meta_updated_at') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD meta_updated_at DATETIME NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'meta_error') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD meta_error VARCHAR(500) NULL;
GO
IF COL_LENGTH('dbo.SEC_Securities', 'meta_attempts') IS NULL
    ALTER TABLE dbo.SEC_Securities ADD meta_attempts INT NOT NULL DEFAULT(0);
GO

UPDATE dbo.SEC_Securities SET meta_status = 'PENDING' WHERE meta_status IS NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SEC_Securities_meta_status')
    CREATE INDEX IX_SEC_Securities_meta_status
        ON dbo.SEC_Securities (meta_status) INCLUDE (cik, ticker, symbol);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SEC_Securities_sector_code')
    CREATE INDEX IX_SEC_Securities_sector_code
        ON dbo.SEC_Securities (sector_code) INCLUDE (industry_code, ticker, symbol);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SEC_Securities_cik')
    CREATE INDEX IX_SEC_Securities_cik ON dbo.SEC_Securities (cik);
GO


/* ---------------------------------------------------------------------------
   2) Tags
   --------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.SEC_Tags', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.SEC_Tags (
        id         INT IDENTITY(1,1) PRIMARY KEY,
        tag_code   VARCHAR(50)  NOT NULL,
        tag_name   VARCHAR(150) NULL,
        tag_group  VARCHAR(50)  NULL,     -- SECTOR / SIZE / STYLE / THEME / CUSTOM
        color      VARCHAR(20)  NULL,
        created_at DATETIME     NOT NULL DEFAULT(GETDATE()),
        CONSTRAINT UQ_SEC_Tags_code UNIQUE (tag_code)
    );
END
GO

IF OBJECT_ID('dbo.SEC_Security_Tags', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.SEC_Security_Tags (
        id          INT IDENTITY(1,1) PRIMARY KEY,
        security_id INT      NOT NULL,
        tag_id      INT      NOT NULL,
        created_at  DATETIME NOT NULL DEFAULT(GETDATE()),
        CONSTRAINT UQ_SEC_Security_Tags UNIQUE (security_id, tag_id),
        CONSTRAINT FK_SEC_Security_Tags_tag
            FOREIGN KEY (tag_id) REFERENCES dbo.SEC_Tags(id) ON DELETE CASCADE
    );
    CREATE INDEX IX_SEC_Security_Tags_tag ON dbo.SEC_Security_Tags (tag_id);
    CREATE INDEX IX_SEC_Security_Tags_sec ON dbo.SEC_Security_Tags (security_id);
END
GO

/* Tags base */
;WITH src(tag_code, tag_name, tag_group, color) AS (
    SELECT * FROM (VALUES
        ('ENERGY',       'Energy',                 'SECTOR', '#D29922'),
        ('MATERIALS',    'Materials',              'SECTOR', '#9A7B4F'),
        ('INDUSTRIALS',  'Industrials',            'SECTOR', '#6E7681'),
        ('CONS_DISCR',   'Consumer Discretionary', 'SECTOR', '#D95F7A'),
        ('CONS_STAPLES', 'Consumer Staples',       'SECTOR', '#3FB950'),
        ('HEALTH_CARE',  'Health Care',            'SECTOR', '#58A6FF'),
        ('FINANCIALS',   'Financials',             'SECTOR', '#1F6FEB'),
        ('INFO_TECH',    'Information Technology', 'SECTOR', '#7D5BD6'),
        ('COMM_SVCS',    'Communication Services', 'SECTOR', '#C74A9E'),
        ('UTILITIES',    'Utilities',              'SECTOR', '#4FB3A5'),
        ('REAL_ESTATE',  'Real Estate',            'SECTOR', '#B58A3F'),
        ('GOVT',         'Government / Agency',    'SECTOR', '#3D444D'),
        ('UNKNOWN',      'Unclassified',           'SECTOR', '#3D444D'),
        ('US_LARGE_CAP', 'US Large Cap',           'SIZE',   '#1F6FEB'),
        ('US_MID_CAP',   'US Mid Cap',             'SIZE',   '#58A6FF'),
        ('US_SMALL_CAP', 'US Small Cap',           'SIZE',   '#3FB950'),
        ('US_MICRO_CAP', 'US Micro Cap',           'SIZE',   '#9A7B4F')
    ) v(tag_code, tag_name, tag_group, color)
)
MERGE dbo.SEC_Tags AS tgt
USING src ON tgt.tag_code = src.tag_code
WHEN MATCHED THEN
    UPDATE SET tag_name = src.tag_name, tag_group = src.tag_group, color = src.color
WHEN NOT MATCHED BY TARGET THEN
    INSERT (tag_code, tag_name, tag_group, color)
    VALUES (src.tag_code, src.tag_name, src.tag_group, src.color);
GO


/* ---------------------------------------------------------------------------
   3) Stored procedures
   --------------------------------------------------------------------------- */

/* --- Update de metadata ---------------------------------------------------- */
IF OBJECT_ID('dbo.Update_SECSecurityMetadata', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Update_SECSecurityMetadata;
GO
CREATE PROCEDURE dbo.Update_SECSecurityMetadata
    @cik                    INT,
    @sic                    VARCHAR(20)  = NULL,
    @sic_description        VARCHAR(255) = NULL,
    @exchange               VARCHAR(100) = NULL,
    @entity_type            VARCHAR(50)  = NULL,
    @sector_code            VARCHAR(20)  = NULL,
    @sector_name            VARCHAR(80)  = NULL,
    @industry_code          VARCHAR(30)  = NULL,
    @industry_name          VARCHAR(120) = NULL,
    @fiscal_year_end        VARCHAR(8)   = NULL,
    @state_of_incorporation VARCHAR(8)   = NULL
AS
BEGIN
    SET NOCOUNT ON;

    UPDATE dbo.SEC_Securities
       SET sic                    = @sic,
           sic_description        = @sic_description,
           category               = @sic_description,   -- category = sicDescription
           exchange               = @exchange,
           entity_type            = @entity_type,
           sector_code            = @sector_code,
           sector_name            = @sector_name,
           industry_code          = @industry_code,
           industry_name          = @industry_name,
           fiscal_year_end        = @fiscal_year_end,
           state_of_incorporation = @state_of_incorporation,
           meta_status            = 'OK',
           meta_error             = NULL,
           meta_updated_at        = GETDATE(),
           meta_attempts          = meta_attempts + 1
     WHERE cik = @cik;

    SELECT @@ROWCOUNT AS affected;
END
GO


/* --- Marcar fallida -------------------------------------------------------- */
IF OBJECT_ID('dbo.Mark_SECSecurityMetadataFailed', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Mark_SECSecurityMetadataFailed;
GO
CREATE PROCEDURE dbo.Mark_SECSecurityMetadataFailed
    @cik    INT,
    @status VARCHAR(20),
    @error  VARCHAR(500)
AS
BEGIN
    SET NOCOUNT ON;

    UPDATE dbo.SEC_Securities
       SET meta_status     = @status,
           meta_error      = @error,
           meta_updated_at = GETDATE(),
           meta_attempts   = meta_attempts + 1
     WHERE cik = @cik;
END
GO


/* --- Cola de pendientes (esto hace el job reanudable) ---------------------- */
IF OBJECT_ID('dbo.Get_SECSecuritiesPendingMetadata', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Get_SECSecuritiesPendingMetadata;
GO
CREATE PROCEDURE dbo.Get_SECSecuritiesPendingMetadata
    @top            INT = NULL,
    @include_errors BIT = 0
AS
BEGIN
    SET NOCOUNT ON;

    SELECT TOP (ISNULL(@top, 2147483647))
           id, cik, ticker, symbol, name, meta_status, meta_attempts
      FROM dbo.SEC_Securities
     WHERE cik IS NOT NULL
       AND (
             ISNULL(meta_status, 'PENDING') = 'PENDING'
             OR sector_code IS NULL
             OR (@include_errors = 1 AND meta_status IN ('ERROR', 'NOT_FOUND'))
           )
     ORDER BY id;
END
GO


/* --- Una sola security ----------------------------------------------------- */
IF OBJECT_ID('dbo.Get_SECSecurityByKey', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Get_SECSecurityByKey;
GO
CREATE PROCEDURE dbo.Get_SECSecurityByKey
    @symbol VARCHAR(50) = NULL,
    @cik    INT         = NULL
AS
BEGIN
    SET NOCOUNT ON;

    SELECT TOP (1) id, cik, ticker, symbol, name, meta_status, meta_attempts
      FROM dbo.SEC_Securities
     WHERE (@cik IS NOT NULL AND cik = @cik)
        OR (@symbol IS NOT NULL AND (UPPER(symbol) = UPPER(@symbol)
                                  OR UPPER(ticker) = UPPER(@symbol)))
     ORDER BY id;
END
GO


/* --- Resumen de cobertura -------------------------------------------------- */
IF OBJECT_ID('dbo.Get_SECSecuritiesMetadataSummary', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Get_SECSecuritiesMetadataSummary;
GO
CREATE PROCEDURE dbo.Get_SECSecuritiesMetadataSummary
AS
BEGIN
    SET NOCOUNT ON;

    SELECT COUNT(*)                                                          AS total,
           SUM(CASE WHEN meta_status = 'OK'        THEN 1 ELSE 0 END)        AS ok_qty,
           SUM(CASE WHEN ISNULL(meta_status,'PENDING') = 'PENDING'
                    THEN 1 ELSE 0 END)                                       AS pending_qty,
           SUM(CASE WHEN meta_status = 'ERROR'     THEN 1 ELSE 0 END)        AS error_qty,
           SUM(CASE WHEN meta_status = 'NOT_FOUND' THEN 1 ELSE 0 END)        AS not_found_qty
      FROM dbo.SEC_Securities;
END
GO


/* --- Breakdown por sector -------------------------------------------------- */
IF OBJECT_ID('dbo.Get_SECSecuritiesSectorBreakdown', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Get_SECSecuritiesSectorBreakdown;
GO
CREATE PROCEDURE dbo.Get_SECSecuritiesSectorBreakdown
AS
BEGIN
    SET NOCOUNT ON;

    SELECT sector_code, MAX(sector_name) AS sector_name, COUNT(*) AS qty
      FROM dbo.SEC_Securities
     WHERE sector_code IS NOT NULL
     GROUP BY sector_code
     ORDER BY COUNT(*) DESC;
END
GO


/* --- Volver los fallidos a la cola ----------------------------------------- */
IF OBJECT_ID('dbo.Reset_SECSecuritiesMetadataErrors', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Reset_SECSecuritiesMetadataErrors;
GO
CREATE PROCEDURE dbo.Reset_SECSecuritiesMetadataErrors
AS
BEGIN
    SET NOCOUNT ON;

    UPDATE dbo.SEC_Securities
       SET meta_status = 'PENDING', meta_error = NULL
     WHERE meta_status IN ('ERROR', 'NOT_FOUND');

    SELECT @@ROWCOUNT AS affected;
END
GO


/* --- Busqueda filtrada (la usa la pantalla) -------------------------------- */
IF OBJECT_ID('dbo.Get_SECSecuritiesFiltered', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Get_SECSecuritiesFiltered;
GO
CREATE PROCEDURE dbo.Get_SECSecuritiesFiltered
    @sector_code   VARCHAR(20)  = NULL,
    @industry_code VARCHAR(30)  = NULL,
    @tag_code      VARCHAR(50)  = NULL,
    @text          VARCHAR(100) = NULL,
    @top           INT          = 500
AS
BEGIN
    SET NOCOUNT ON;

    SELECT TOP (ISNULL(@top, 500))
           s.id, s.cik, s.ticker, s.symbol, s.name, s.exchange,
           s.sic, s.sic_description, s.entity_type,
           s.sector_code, s.sector_name, s.industry_code, s.industry_name,
           s.meta_status, s.meta_updated_at, s.meta_error,
           STUFF((SELECT ',' + t.tag_code
                    FROM dbo.SEC_Security_Tags st
                    JOIN dbo.SEC_Tags t ON t.id = st.tag_id
                   WHERE st.security_id = s.id
                   ORDER BY t.tag_code
                     FOR XML PATH(''), TYPE).value('.', 'VARCHAR(MAX)'), 1, 1, '') AS tags
      FROM dbo.SEC_Securities s
     WHERE (@sector_code   IS NULL OR s.sector_code   = @sector_code)
       AND (@industry_code IS NULL OR s.industry_code = @industry_code)
       AND (@text          IS NULL OR s.symbol LIKE '%' + @text + '%'
                                   OR s.ticker LIKE '%' + @text + '%'
                                   OR s.name   LIKE '%' + @text + '%')
       AND (@tag_code IS NULL OR EXISTS (
                SELECT 1 FROM dbo.SEC_Security_Tags st
                  JOIN dbo.SEC_Tags t ON t.id = st.tag_id
                 WHERE st.security_id = s.id AND t.tag_code = @tag_code))
     ORDER BY s.symbol;
END
GO


/* --- Tags ------------------------------------------------------------------ */
IF OBJECT_ID('dbo.Persist_SECTag', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Persist_SECTag;
GO
CREATE PROCEDURE dbo.Persist_SECTag
    @tag_code  VARCHAR(50),
    @tag_name  VARCHAR(150) = NULL,
    @tag_group VARCHAR(50)  = NULL,
    @color     VARCHAR(20)  = NULL
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (SELECT 1 FROM dbo.SEC_Tags WHERE tag_code = @tag_code)
        UPDATE dbo.SEC_Tags
           SET tag_name  = COALESCE(@tag_name,  tag_name),
               tag_group = COALESCE(@tag_group, tag_group),
               color     = COALESCE(@color,     color)
         WHERE tag_code = @tag_code;
    ELSE
        INSERT INTO dbo.SEC_Tags (tag_code, tag_name, tag_group, color)
        VALUES (@tag_code, ISNULL(@tag_name, @tag_code), ISNULL(@tag_group, 'CUSTOM'), @color);

    SELECT id FROM dbo.SEC_Tags WHERE tag_code = @tag_code;
END
GO


IF OBJECT_ID('dbo.Get_SECTags', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Get_SECTags;
GO
CREATE PROCEDURE dbo.Get_SECTags
AS
BEGIN
    SET NOCOUNT ON;

    SELECT t.id, t.tag_code, t.tag_name, t.tag_group, t.color, t.created_at,
           (SELECT COUNT(*) FROM dbo.SEC_Security_Tags st WHERE st.tag_id = t.id) AS qty
      FROM dbo.SEC_Tags t
     ORDER BY t.tag_group, t.tag_code;
END
GO


IF OBJECT_ID('dbo.Delete_SECTag', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Delete_SECTag;
GO
CREATE PROCEDURE dbo.Delete_SECTag
    @tag_code VARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;
    DELETE FROM dbo.SEC_Tags WHERE tag_code = @tag_code;
    SELECT @@ROWCOUNT AS affected;
END
GO


/* Aplica un tag a un simbolo. Devuelve 1 si matcheo, 0 si el simbolo no existe. */
IF OBJECT_ID('dbo.Persist_SECSecurityTagBySymbol', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Persist_SECSecurityTagBySymbol;
GO
CREATE PROCEDURE dbo.Persist_SECSecurityTagBySymbol
    @tag_code VARCHAR(50),
    @symbol   VARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @tag_id INT = (SELECT id FROM dbo.SEC_Tags WHERE tag_code = @tag_code);
    IF @tag_id IS NULL
    BEGIN
        INSERT INTO dbo.SEC_Tags (tag_code, tag_name, tag_group)
        VALUES (@tag_code, @tag_code, 'CUSTOM');
        SET @tag_id = SCOPE_IDENTITY();
    END

    DECLARE @matched INT = 0;

    INSERT INTO dbo.SEC_Security_Tags (security_id, tag_id)
    SELECT s.id, @tag_id
      FROM dbo.SEC_Securities s
     WHERE (UPPER(s.symbol) = UPPER(@symbol) OR UPPER(s.ticker) = UPPER(@symbol))
       AND NOT EXISTS (SELECT 1 FROM dbo.SEC_Security_Tags st
                        WHERE st.security_id = s.id AND st.tag_id = @tag_id);

    SELECT @matched = COUNT(*)
      FROM dbo.SEC_Securities s
     WHERE UPPER(s.symbol) = UPPER(@symbol) OR UPPER(s.ticker) = UPPER(@symbol);

    SELECT @matched AS matched;
END
GO


IF OBJECT_ID('dbo.Apply_SECTagBySector', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Apply_SECTagBySector;
GO
CREATE PROCEDURE dbo.Apply_SECTagBySector
    @tag_code    VARCHAR(50),
    @sector_code VARCHAR(20)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @tag_id INT = (SELECT id FROM dbo.SEC_Tags WHERE tag_code = @tag_code);
    IF @tag_id IS NULL
    BEGIN
        INSERT INTO dbo.SEC_Tags (tag_code, tag_name, tag_group)
        VALUES (@tag_code, @tag_code, 'SECTOR');
        SET @tag_id = SCOPE_IDENTITY();
    END

    INSERT INTO dbo.SEC_Security_Tags (security_id, tag_id)
    SELECT s.id, @tag_id
      FROM dbo.SEC_Securities s
     WHERE s.sector_code = @sector_code
       AND NOT EXISTS (SELECT 1 FROM dbo.SEC_Security_Tags st
                        WHERE st.security_id = s.id AND st.tag_id = @tag_id);

    SELECT @@ROWCOUNT AS affected;
END
GO


IF OBJECT_ID('dbo.Delete_SECSecurityTag', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Delete_SECSecurityTag;
GO
CREATE PROCEDURE dbo.Delete_SECSecurityTag
    @tag_code    VARCHAR(50),
    @security_id INT
AS
BEGIN
    SET NOCOUNT ON;

    DELETE st
      FROM dbo.SEC_Security_Tags st
      JOIN dbo.SEC_Tags t ON t.id = st.tag_id
     WHERE t.tag_code = @tag_code AND st.security_id = @security_id;

    SELECT @@ROWCOUNT AS affected;
END
GO


IF OBJECT_ID('dbo.Get_SECSymbolsByTag', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Get_SECSymbolsByTag;
GO
CREATE PROCEDURE dbo.Get_SECSymbolsByTag
    @tag_code      VARCHAR(50),
    @industry_code VARCHAR(30) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    SELECT s.symbol, s.ticker, s.name, s.sector_code, s.industry_code
      FROM dbo.SEC_Securities s
      JOIN dbo.SEC_Security_Tags st ON st.security_id = s.id
      JOIN dbo.SEC_Tags t           ON t.id = st.tag_id
     WHERE t.tag_code = @tag_code
       AND (@industry_code IS NULL OR s.industry_code = @industry_code)
     ORDER BY s.symbol;
END
GO
