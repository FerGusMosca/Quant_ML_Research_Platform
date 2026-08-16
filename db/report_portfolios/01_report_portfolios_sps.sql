-- =====================================================================
-- 01_report_portfolios_sps.sql
-- SQL Server - base machine_learning_research
--
-- SCRIPT IDEMPOTENTE: se puede correr las veces que haga falta.
-- NO crea, NO modifica y NO borra la tabla dbo.report_portfolios.
-- Solo crea el procedimiento de lectura que usan las pantallas.
-- =====================================================================

USE machine_learning_research;
GO

IF OBJECT_ID('dbo.Get_ReportPortfolios', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Get_ReportPortfolios;
GO

CREATE PROCEDURE dbo.Get_ReportPortfolios
AS
BEGIN
    SET NOCOUNT ON;

    -- Es el catalogo de portfolios de los reportes: el mismo que llena el
    -- combo del Document Tagger y el que viaja en el argumento "portfolio"
    -- de los comandos del MCP.
    SELECT p.id,
           p.portfolio_code,
           p.name,
           p.description
      FROM dbo.report_portfolios p
     ORDER BY p.portfolio_code;
END
GO


-- ---------------------------------------------------------------------
-- Chequeo final: si esto devuelve filas, quedo bien.
-- ---------------------------------------------------------------------
EXEC dbo.Get_ReportPortfolios;
GO
