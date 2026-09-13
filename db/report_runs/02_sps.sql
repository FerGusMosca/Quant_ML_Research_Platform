/* =============================================================================
   STORED PROCEDURES SCRIPT  -  machine_learning_research
   100% IDEMPOTENTE: todo con CREATE OR ALTER, se puede correr N veces.
   ============================================================================= */

/* ---------------------------------------------------------------------------
   persist_report_run
   @id = 0  --> INSERT (arranca la corrida, status = started)
   @id <> 0 --> UPDATE (cierra la corrida: status, end_time y error)
   Devuelve el id.
   --------------------------------------------------------------------------- */
CREATE OR ALTER PROCEDURE dbo.persist_report_run
    @id          INT,
    @job_id      VARCHAR(64)   = NULL,
    @report_key  VARCHAR(100)  = NULL,
    @portfolio   VARCHAR(100)  = NULL,
    @symbol      VARCHAR(50)   = NULL,
    @year        VARCHAR(50)   = NULL,
    @quarter     VARCHAR(20)   = NULL,
    @source      VARCHAR(100)  = NULL,
    @params_json NVARCHAR(MAX) = NULL,
    @status      VARCHAR(20)   = NULL,
    @last_error  NVARCHAR(MAX) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @now DATETIME = GETDATE();

    IF ISNULL(@id, 0) = 0
    BEGIN
        INSERT INTO dbo.report_runs
            (job_id, report_key, portfolio, symbol, [year], quarter, [source],
             params_json, [status], start_time, end_time, last_error, last_update_time)
        VALUES
            (@job_id, @report_key, @portfolio, @symbol, @year, @quarter, @source,
             @params_json, ISNULL(@status, 'started'), @now, NULL, @last_error, @now);

        SELECT CAST(SCOPE_IDENTITY() AS INT) AS id;
        RETURN;
    END

    UPDATE dbo.report_runs
       SET [status]         = ISNULL(@status, [status]),
           last_error       = ISNULL(@last_error, last_error),
           end_time         = CASE WHEN ISNULL(@status, '') IN ('finished', 'error', 'aborted')
                                   THEN @now ELSE end_time END,
           last_update_time = @now
     WHERE id = @id;

    SELECT CAST(@id AS INT) AS id;
END
GO

/* ---------------------------------------------------------------------------
   get_report_runs : las ultimas corridas, con filtro opcional
   --------------------------------------------------------------------------- */
CREATE OR ALTER PROCEDURE dbo.get_report_runs
    @top        INT          = 50,
    @status     VARCHAR(20)  = NULL,
    @report_key VARCHAR(100) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    SELECT TOP (ISNULL(@top, 50))
           r.id,
           r.job_id,
           r.report_key,
           r.portfolio,
           r.symbol,
           r.[year],
           r.quarter,
           r.[source],
           r.[status],
           r.start_time,
           r.end_time,
           DATEDIFF(SECOND, r.start_time, ISNULL(r.end_time, GETDATE())) AS elapsed_seconds,
           r.last_error,
           r.last_update_time,
           r.params_json
      FROM dbo.report_runs r
     WHERE (@status IS NULL OR r.[status] = @status)
       AND (@report_key IS NULL OR r.report_key = @report_key)
     ORDER BY r.start_time DESC, r.id DESC;
END
GO

/* ---------------------------------------------------------------------------
   reset_stuck_report_runs : pasa a 'aborted' lo que quedo colgado en 'started'
   @id = NULL y @older_than_hours = NULL --> limpia TODO lo que este en started
   Devuelve la cantidad de filas tocadas.
   --------------------------------------------------------------------------- */
CREATE OR ALTER PROCEDURE dbo.reset_stuck_report_runs
    @id               INT = NULL,
    @older_than_hours INT = NULL
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @now DATETIME = GETDATE();

    UPDATE dbo.report_runs
       SET [status]         = 'aborted',
           end_time         = ISNULL(end_time, @now),
           last_error       = ISNULL(last_error, 'Run marked as aborted by reset_stuck_report_runs'),
           last_update_time = @now
     WHERE [status] = 'started'
       AND (@id IS NULL OR id = @id)
       AND (@older_than_hours IS NULL OR start_time < DATEADD(HOUR, -@older_than_hours, @now));

    SELECT @@ROWCOUNT AS rows_reset;
END
GO

/* ---------------------------------------------------------------------------
   delete_report_run : borra una corrida puntual
   --------------------------------------------------------------------------- */
CREATE OR ALTER PROCEDURE dbo.delete_report_run
    @id INT
AS
BEGIN
    SET NOCOUNT ON;

    DELETE FROM dbo.report_runs
     WHERE id = @id;

    SELECT @@ROWCOUNT AS rows_deleted;
END
GO
