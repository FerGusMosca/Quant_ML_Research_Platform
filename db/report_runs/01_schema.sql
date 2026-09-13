/* =============================================================================
   SCHEMA SCRIPT  -  machine_learning_research
   100% IDEMPOTENTE: se puede correr todas las veces que haga falta.
   ============================================================================= */

/* ---------------------------------------------------------------------------
   report_runs : una fila por cada corrida disparada desde el MCP / Reports Runner
   --------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.report_runs', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.report_runs
    (
        id               INT IDENTITY(1,1) NOT NULL,
        job_id           VARCHAR(64)       NULL,
        report_key       VARCHAR(100)      NOT NULL,
        portfolio        VARCHAR(100)      NULL,
        symbol           VARCHAR(50)       NULL,
        [year]           VARCHAR(50)       NULL,
        quarter          VARCHAR(20)       NULL,
        [source]         VARCHAR(100)      NULL,
        params_json      NVARCHAR(MAX)     NULL,
        [status]         VARCHAR(20)       NOT NULL,
        start_time       DATETIME          NOT NULL,
        end_time         DATETIME          NULL,
        last_error       NVARCHAR(MAX)     NULL,
        last_update_time DATETIME          NOT NULL,
        CONSTRAINT PK_report_runs PRIMARY KEY CLUSTERED (id)
    );
END
GO

/* Columnas agregadas de a una, por si la tabla ya existia de antes */
IF COL_LENGTH('dbo.report_runs', 'job_id') IS NULL
    ALTER TABLE dbo.report_runs ADD job_id VARCHAR(64) NULL;
GO

IF COL_LENGTH('dbo.report_runs', 'params_json') IS NULL
    ALTER TABLE dbo.report_runs ADD params_json NVARCHAR(MAX) NULL;
GO

IF COL_LENGTH('dbo.report_runs', 'end_time') IS NULL
    ALTER TABLE dbo.report_runs ADD end_time DATETIME NULL;
GO

IF COL_LENGTH('dbo.report_runs', 'last_error') IS NULL
    ALTER TABLE dbo.report_runs ADD last_error NVARCHAR(MAX) NULL;
GO

/* Default de status por si entra una fila sin estado */
IF NOT EXISTS (SELECT 1 FROM sys.default_constraints WHERE name = 'DF_report_runs_status')
   AND COL_LENGTH('dbo.report_runs', 'status') IS NOT NULL
BEGIN
    ALTER TABLE dbo.report_runs
        ADD CONSTRAINT DF_report_runs_status DEFAULT ('started') FOR [status];
END
GO

/* Indices de consulta: "la ultima corrida" y "las que quedaron started" */
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'IX_report_runs_start_time'
                 AND object_id = OBJECT_ID('dbo.report_runs'))
BEGIN
    CREATE INDEX IX_report_runs_start_time
        ON dbo.report_runs (start_time DESC);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'IX_report_runs_status'
                 AND object_id = OBJECT_ID('dbo.report_runs'))
BEGIN
    CREATE INDEX IX_report_runs_status
        ON dbo.report_runs ([status], start_time DESC);
END
GO
