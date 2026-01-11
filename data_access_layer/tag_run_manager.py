import pyodbc
from datetime import datetime
from typing import List, Optional

from business_entities.tag_run import TagRun
from framework.common.logger.message_type import MessageType


class TagRunManager:
    """
    Data Access Layer for managing tag run records using stored procedures
    """
    def __init__(self, connection_string: str, logger):
        """
        Args:
            connection_string: SQL Server connection string
            logger: Logger instance with do_log(message, type) method
        """
        self.connection_string = connection_string
        self.logger = logger
        self._connection = None  # lazy initialization

    @property
    def connection(self):
        """Lazily creates and returns a connection (reconnects if closed)"""
        if self._connection is None or self._connection.closed:
            self._connection = pyodbc.connect(self.connection_string)
            self._connection.autocommit = False
        return self._connection

    def get_all_tag_runs(
        self,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TagRun]:
        """
        Retrieve tag runs using the get_all_tag_runs stored procedure.
        All parameters are optional filters.

        Returns:
            List of TagRun objects
        """
        runs = []
        cursor = None

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "EXEC get_all_tag_runs ?, ?, ?",
                (status, start_date, end_date)
            )

            # Use column names safely instead of hardcoded indices
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()

            for row in rows:
                row_dict = dict(zip(columns, row))
                run = TagRun(
                    id=row_dict.get("id"),
                    portfolio=row_dict.get("portfolio", ""),
                    source=row_dict.get("source", ""),
                    rank_folder=row_dict.get("rank_folder"),
                    timestamp=row_dict.get("timestamp"),
                    tag_file=row_dict.get("tag_file"),
                    tag_json=row_dict.get("tag_json"),
                    tag_model=row_dict.get("tag_model", ""),
                    doc_type=row_dict.get("doc_type", ""),
                    status=row_dict.get("status", "started"),
                    last_error=row_dict.get("last_error"),
                    last_update_time=row_dict.get("last_update_time")
                )
                runs.append(run)

            self.logger.do_log(
                f"get_all_tag_runs → Retrieved {len(runs)} records "
                f"(status={status}, {start_date} → {end_date})",
                MessageType.INFO
            )

        except Exception as e:
            self.logger.do_log(
                f"get_all_tag_runs failed: {str(e)}",
                MessageType.ERROR
            )
            raise

        finally:
            if cursor:
                cursor.close()

        return runs

    def persist_tag_run(self, run: TagRun) -> int:
        """
        Creates or updates a tag run record using the persist_tag_run stored procedure.
        Updates the run.id field with the resulting identifier.

        Args:
            run: TagRun object containing all the data

        Returns:
            The ID of the created/updated record
        """
        cursor = None
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                DECLARE @out_id INT;
                EXEC persist_tag_run 
                    @id = ?,
                    @portfolio = ?,
                    @source = ?,
                    @rank_folder = ?,
                    @tag_file = ?,
                    @tag_json = ?,
                    @tag_model = ?,
                    @doc_type = ?,
                    @status = ?,
                    @last_error = ?,
                    @last_update_time = ?,
                    @out_id = @out_id OUTPUT;
                SELECT @out_id;
            """, (
                run.id,
                run.portfolio,
                run.source,
                run.rank_folder,
                run.tag_file,
                run.tag_json,           # should be string (json serialized) if DB expects nvarchar
                run.tag_model,
                run.doc_type,
                run.status,
                run.last_error,
                run.last_update_time
            ))

            new_id = cursor.fetchone()[0]
            self.connection.commit()

            action = "CREATED" if run.id is None else "UPDATED"
            self.logger.do_log(
                f"persist_tag_run → {action} run_id={new_id} | "
                f"portfolio={run.portfolio} | status={run.status}",
                MessageType.INFO
            )

            run.id = new_id  # update the object with the real id
            return new_id

        except Exception as e:
            if self.connection:
                self.connection.rollback()
            self.logger.do_log(
                f"persist_tag_run failed for portfolio={run.portfolio}: {str(e)}",
                MessageType.ERROR
            )
            raise

        finally:
            if cursor:
                cursor.close()