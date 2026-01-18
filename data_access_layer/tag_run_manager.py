import pyodbc
from typing import Optional

from business_entities.tag_run import TagRun
from framework.common.logger.message_type import MessageType


class TagRunManager:
    """
    Data Access Layer for tag_runs table
    """

    def __init__(self, connection_string: str, logger):
        self.connection_string = connection_string
        self.logger = logger
        self._connection = None

    @property
    def connection(self):
        if self._connection is None or self._connection.closed:
            self._connection = pyodbc.connect(self.connection_string)
            self._connection.autocommit = False
        return self._connection

    def persist_tag_run(self, run: TagRun) -> int:
        """
        Persist a tag run using persist_tag_run SP.

        Rule:
        - run.id == 0  → INSERT (all fields)
        - run.id != 0  → UPDATE (status only)

        Returns:
            run id
        """
        cursor = None
        try:
            cursor = self.connection.cursor()

            cursor.execute(
                """
                EXEC persist_tag_run ?, ?, ?, ?, ?, ?,?,?, ?, ?, ?, ?, ?,?
                """,
                (
                    run.id or 0,  # @id
                    run.report,  # @report
                    run.portfolio,  # @portfolio
                    run.source,  # @source
                    run.rank_folder,  # @rank_folder
                    run.year,  # @year
                    run.quarter,  # @year
                    run.sec_processed,  # @year
                    run.tag_model,  # @tag_model
                    run.doc_type,  # @doc_type
                    run.tag_json,  # @tag_json (STRING JSON)
                    run.tag_file,  # @tag_file
                    run.status,  # @status
                    run.last_error
                )
            )

            new_id = cursor.fetchone()[0]
            self.connection.commit()

            action = "CREATED" if (run.id or 0) == 0 else "UPDATED"
            self.logger.do_log(
                f"[TAG_RUN] {action} | id={new_id} | "
                f"portfolio={run.portfolio} | status={run.status}",
                MessageType.INFO
            )

            run.id = new_id
            return new_id

        except Exception as e:
            if self.connection:
                self.connection.rollback()

            self.logger.do_log(
                f"[TAG_RUN] persist failed | portfolio={run.portfolio} | error={e}",
                MessageType.ERROR
            )
            raise

        finally:
            if cursor:
                cursor.close()

