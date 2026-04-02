"""
security_manager.py
===================
Data-access layer for the dbo.securities table.
Uses stored procedures exclusively — no raw SQL in the manager.

Usage example
-------------
    mgr = SecurityManager(connection_string)
    all_securities = mgr.get_securities()
    lecaps         = mgr.get_securities(security_type='LECAP', include_expired=False)
    mgr.persist_security(Security(...))
    mgr.delete_security('S17A6')
    count = mgr.bulk_upsert_securities([Security(...), ...])
"""

import json
import pyodbc
from datetime import date
from typing import Optional

from business_entities.security import Security


# ---------------------------------------------------------------------------
# Column indices returned by GetSecurities SP
# ---------------------------------------------------------------------------
_IDX_ID               = 0
_IDX_SYMBOL           = 1
_IDX_TYPE             = 2
_IDX_DESC             = 3
_IDX_MATURITY         = 4
_IDX_FINAL_PAYMENT    = 5
_IDX_CURRENCY         = 6
_IDX_IS_ACTIVE        = 7
_IDX_IS_EXPIRED       = 8
_IDX_DAYS_TO_MATURITY = 9
_IDX_CREATED_AT       = 10
_IDX_UPDATED_AT       = 11


def _row_to_security(row) -> Security:
    """Map a pyodbc result row to a Security dataclass."""
    maturity = row[_IDX_MATURITY]
    return Security(
        id               = int(row[_IDX_ID]),
        symbol           = str(row[_IDX_SYMBOL]),
        security_type    = str(row[_IDX_TYPE]),
        description      = str(row[_IDX_DESC] or ''),
        maturity_date    = maturity.isoformat() if isinstance(maturity, date) else str(maturity),
        final_payment    = float(row[_IDX_FINAL_PAYMENT]) if row[_IDX_FINAL_PAYMENT] is not None else 0.0,
        currency         = str(row[_IDX_CURRENCY]),
        is_active        = int(row[_IDX_IS_ACTIVE]),
        is_expired       = int(row[_IDX_IS_EXPIRED]),
        days_to_maturity = int(row[_IDX_DAYS_TO_MATURITY]),
        created_at       = str(row[_IDX_CREATED_AT]) if row[_IDX_CREATED_AT] else None,
        updated_at       = str(row[_IDX_UPDATED_AT]) if row[_IDX_UPDATED_AT] else None,
    )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SecurityManager:

    def __init__(self, connection_string: str):
        self.connection = pyodbc.connect(connection_string)

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def get_securities(
        self,
        security_type:   Optional[str] = None,
        include_expired: bool          = True,
    ) -> list[Security]:
        """
        Returns securities ordered by maturity_date ASC.

        Parameters
        ----------
        security_type   : filter by type ('LECAP', 'BONCAP', …) or None for all.
        include_expired : if False, excludes securities whose maturity_date < today.
        """
        securities = []
        with self.connection.cursor() as cursor:
            cursor.execute(
                "{CALL GetSecurities (?, ?)}",
                (security_type, 1 if include_expired else 0),
            )
            for row in cursor:
                securities.append(_row_to_security(row))
        return securities

    # ------------------------------------------------------------------
    # WRITE — single upsert
    # ------------------------------------------------------------------

    def persist_security(self, security: Security) -> None:
        """
        Insert or update a single security identified by symbol.
        Reactivates soft-deleted records automatically.
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                "{CALL PersistSecurity (?, ?, ?, ?, ?, ?)}",
                (
                    security.symbol,
                    security.security_type,
                    security.description,
                    security.maturity_date,
                    security.final_payment,
                    security.currency,
                ),
            )
            self.connection.commit()

    # ------------------------------------------------------------------
    # WRITE — soft delete
    # ------------------------------------------------------------------

    def delete_security(self, symbol: str) -> None:
        """
        Soft-deletes a security (sets is_active = 0).
        The record remains in the database for audit purposes.
        """
        with self.connection.cursor() as cursor:
            cursor.execute("{CALL DeleteSecurity (?)}", (symbol,))
            self.connection.commit()

    # ------------------------------------------------------------------
    # WRITE — bulk upsert (CSV / JSON import)
    # ------------------------------------------------------------------

    def bulk_upsert_securities(self, securities: list[Security]) -> int:
        """
        Upsert a list of securities in a single round-trip using
        BulkUpsertSecurities SP (JSON payload).

        Returns
        -------
        int : number of rows affected (inserted + updated).
        """
        payload = json.dumps([
            {
                "symbol":        s.symbol,
                "security_type": s.security_type,
                "description":   s.description,
                "maturity_date": s.maturity_date,
                "final_payment": s.final_payment,
                "currency":      s.currency,
            }
            for s in securities
        ])

        rows_affected = 0
        with self.connection.cursor() as cursor:
            cursor.execute("{CALL BulkUpsertSecurities (?)}", (payload,))
            row = cursor.fetchone()
            if row:
                rows_affected = int(row[0])
            self.connection.commit()

        return rows_affected