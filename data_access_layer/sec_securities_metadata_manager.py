from typing import Optional

import pyodbc

from framework.common.logger.message_type import MessageType


class SECSecuritiesMetadataManager:
    """
    Data Access Layer for SEC_Securities metadata (sic, sicDescription,
    exchange, entityType, sector/industry) and for the tagging system.

    Kept separate from SECSecuritiesManager on purpose: that one handles the
    security insert, this one the later enrichment.

    Requires db/sec_metadata/01_schema_and_sps.sql to be applied.
    """

    def __init__(self, connection_string, logger):
        self.connection = pyodbc.connect(connection_string)
        self.logger = logger

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def __rows_to_dicts__(cursor):
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ── Work queue ────────────────────────────────────────────────────────

    def get_pending_securities(self, top: int = None, include_errors: bool = False):
        """
        Securities with no metadata yet. This is what makes the job resumable:
        state lives in the meta_status column, not in a checkpoint file.
        """
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Get_SECSecuritiesPendingMetadata ?, ?",
                           (top, 1 if include_errors else 0))
            return self.__rows_to_dicts__(cursor)

    def get_security_by_key(self, symbol: str = None, cik: int = None) -> Optional[dict]:
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Get_SECSecurityByKey ?, ?", (symbol, cik))
            rows = self.__rows_to_dicts__(cursor)
            return rows[0] if rows else None

    # ── Metadata writes ───────────────────────────────────────────────────────

    def persist_metadata(self, metadata_dto) -> int:
        """Persists a SecSecurityMetadataDTO and marks the security as OK."""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                EXEC Update_SECSecurityMetadata ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            """, (
                metadata_dto.cik,
                metadata_dto.sic,
                metadata_dto.sic_description,
                metadata_dto.exchange,
                metadata_dto.entity_type,
                metadata_dto.sector_code,
                metadata_dto.sector_name,
                metadata_dto.industry_code,
                metadata_dto.industry_name,
                metadata_dto.fiscal_year_end,
                metadata_dto.state_of_incorporation,
            ))
            row = cursor.fetchone()
            affected = row[0] if row else 0
        self.connection.commit()
        return affected

    def mark_failed(self, cik: int, status: str, error: str):
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Mark_SECSecurityMetadataFailed ?, ?, ?",
                           (cik, status[:20], (error or "")[:500]))
        self.connection.commit()

    def reset_errors(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Reset_SECSecuritiesMetadataErrors")
            row = cursor.fetchone()
            affected = row[0] if row else 0
        self.connection.commit()
        return affected

    # ── Reads for the screen ──────────────────────────────────────────────────

    def get_summary(self) -> dict:
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Get_SECSecuritiesMetadataSummary")
            rows = self.__rows_to_dicts__(cursor)
            summary = rows[0] if rows else {}

            cursor.execute("EXEC Get_SECSecuritiesSectorBreakdown")
            summary["sectors"] = self.__rows_to_dicts__(cursor)

        return summary

    def search(self, sector_code=None, industry_code=None, tag_code=None,
               text=None, top: int = 500):
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Get_SECSecuritiesFiltered ?, ?, ?, ?, ?",
                           (sector_code, industry_code, tag_code, text, top))
            return self.__rows_to_dicts__(cursor)

    # ── Tags ──────────────────────────────────────────────────────────────────

    @staticmethod
    def normalize_tag_code(tag_code: str) -> str:
        """'#us small cap' -> 'US_SMALL_CAP'"""
        return (tag_code or "").strip().lstrip("#").upper().replace(" ", "_").replace("-", "_")

    def get_tags(self):
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Get_SECTags")
            return self.__rows_to_dicts__(cursor)

    def persist_tag(self, tag_code: str, tag_name: str = None,
                    tag_group: str = "CUSTOM", color: str = None) -> int:
        code = self.normalize_tag_code(tag_code)
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Persist_SECTag ?, ?, ?, ?",
                           (code, tag_name or code, tag_group, color))
            row = cursor.fetchone()
            tag_id = row[0] if row else None
        self.connection.commit()
        return tag_id

    def delete_tag(self, tag_code: str) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Delete_SECTag ?", (self.normalize_tag_code(tag_code),))
            row = cursor.fetchone()
            affected = row[0] if row else 0
        self.connection.commit()
        return affected

    def apply_tag_to_symbols(self, tag_code: str, symbols: list) -> dict:
        """
        Applies a tag to a list of symbols.
        Returns how many matched and which ones do not exist, so the popup can
        show them.
        """
        code = self.normalize_tag_code(tag_code)
        self.persist_tag(code)

        wanted = []
        seen = set()
        for s in symbols or []:
            v = (s or "").strip().upper()
            if v and v not in seen:
                seen.add(v)
                wanted.append(v)

        tagged, not_found = 0, []

        with self.connection.cursor() as cursor:
            for symbol in wanted:
                try:
                    cursor.execute("EXEC Persist_SECSecurityTagBySymbol ?, ?", (code, symbol))
                    row = cursor.fetchone()
                    if row and row[0] and int(row[0]) > 0:
                        tagged += 1
                    else:
                        not_found.append(symbol)
                except Exception as e:
                    not_found.append(symbol)
                    self.logger.do_log(
                        f"apply_tag_to_symbols: ❌ {symbol} - {str(e)}", MessageType.ERROR)
        self.connection.commit()

        self.logger.do_log(
            f"apply_tag_to_symbols: tag {code} applied to {tagged} securities "
            f"({len(not_found)} unmatched)", MessageType.INFO)

        return {"tag_code": code, "read": len(wanted),
                "tagged": tagged, "not_found": not_found}

    def apply_tag_by_sector(self, tag_code: str, sector_code: str) -> int:
        code = self.normalize_tag_code(tag_code)
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Apply_SECTagBySector ?, ?", (code, sector_code))
            row = cursor.fetchone()
            affected = row[0] if row else 0
        self.connection.commit()
        return affected

    def remove_tag_from_security(self, tag_code: str, security_id: int) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Delete_SECSecurityTag ?, ?",
                           (self.normalize_tag_code(tag_code), security_id))
            row = cursor.fetchone()
            affected = row[0] if row else 0
        self.connection.commit()
        return affected

    def get_symbols_by_tag(self, tag_code: str, industry_code: str = None):
        with self.connection.cursor() as cursor:
            cursor.execute("EXEC Get_SECSymbolsByTag ?, ?",
                           (self.normalize_tag_code(tag_code), industry_code))
            return self.__rows_to_dicts__(cursor)
