"""
security_manager.py
===================
Data-access layer for dbo.securities, dbo.bond_detail, dbo.bond_coupons.
Uses stored procedures exclusively.
"""

import json
import pyodbc
from datetime import date
from typing import Optional

from business_entities.security import Security
from business_entities.bond import BondDetail, BondCoupon

# ---------------------------------------------------------------------------
# Column indices — GetSecurities SP
# ---------------------------------------------------------------------------
_S_ID               = 0
_S_SYMBOL           = 1
_S_TYPE             = 2
_S_DESC             = 3
_S_MATURITY         = 4
_S_FINAL_PAYMENT    = 5
_S_CURRENCY         = 6
_S_IS_ACTIVE        = 7
_S_IS_EXPIRED       = 8
_S_DAYS_TO_MATURITY = 9
_S_CREATED_AT       = 10
_S_UPDATED_AT       = 11

# Column indices — GetBondDetail SP
_BD_SECURITY_ID  = 0
_BD_SYMBOL       = 1
_BD_TYPE         = 2
_BD_DESC         = 3
_BD_MATURITY     = 4
_BD_CURRENCY     = 5
_BD_IS_ACTIVE    = 6
_BD_LAW          = 7
_BD_PAR_SYMBOL   = 8

# Column indices — GetBondCoupons SP
_BC_ID              = 0
_BC_SECURITY_ID     = 1
_BC_SYMBOL          = 2
_BC_PAYMENT_DATE    = 3
_BC_AMOUNT          = 4
_BC_IS_PAID         = 5
_BC_DAYS_TO_PAYMENT = 6


def _row_to_security(row) -> Security:
    maturity = row[_S_MATURITY]
    return Security(
        id               = int(row[_S_ID]),
        symbol           = str(row[_S_SYMBOL]),
        security_type    = str(row[_S_TYPE]),
        description      = str(row[_S_DESC] or ''),
        maturity_date    = maturity.isoformat() if isinstance(maturity, date) else str(maturity),
        final_payment    = float(row[_S_FINAL_PAYMENT]) if row[_S_FINAL_PAYMENT] is not None else 0.0,
        currency         = str(row[_S_CURRENCY]),
        is_active        = int(row[_S_IS_ACTIVE]),
        is_expired       = int(row[_S_IS_EXPIRED]),
        days_to_maturity = int(row[_S_DAYS_TO_MATURITY]),
        created_at       = str(row[_S_CREATED_AT]) if row[_S_CREATED_AT] else None,
        updated_at       = str(row[_S_UPDATED_AT]) if row[_S_UPDATED_AT] else None,
    )


def _row_to_bond_detail(row) -> BondDetail:
    maturity = row[_BD_MATURITY]
    return BondDetail(
        security_id   = int(row[_BD_SECURITY_ID]),
        symbol        = str(row[_BD_SYMBOL]),
        security_type = str(row[_BD_TYPE]),
        description   = str(row[_BD_DESC] or ''),
        maturity_date = maturity.isoformat() if isinstance(maturity, date) else str(maturity),
        currency      = str(row[_BD_CURRENCY]),
        is_active     = int(row[_BD_IS_ACTIVE]),
        law           = str(row[_BD_LAW]),
        par_symbol    = str(row[_BD_PAR_SYMBOL]) if row[_BD_PAR_SYMBOL] else None,
    )


def _row_to_bond_coupon(row) -> BondCoupon:
    pdate = row[_BC_PAYMENT_DATE]
    return BondCoupon(
        id              = int(row[_BC_ID]),
        security_id     = int(row[_BC_SECURITY_ID]),
        symbol          = str(row[_BC_SYMBOL]),
        payment_date    = pdate.isoformat() if isinstance(pdate, date) else str(pdate),
        amount          = float(row[_BC_AMOUNT]),
        is_paid         = bool(row[_BC_IS_PAID]),
        days_to_payment = int(row[_BC_DAYS_TO_PAYMENT]),
    )


class SecurityManager:

    # paid_filter constants for get_bond_coupons
    COUPONS_FUTURE = 0
    COUPONS_PAID   = 1
    COUPONS_ALL    = 2

    def __init__(self, connection_string: str):
        self.connection = pyodbc.connect(connection_string)

    # ======================================================================
    # SECURITIES
    # ======================================================================

    def get_securities(
        self,
        security_type:   Optional[str] = None,
        include_expired: bool          = True,
    ) -> list[Security]:
        results = []
        with self.connection.cursor() as cursor:
            cursor.execute(
                "{CALL GetSecurities (?, ?)}",
                (security_type, 1 if include_expired else 0),
            )
            for row in cursor:
                results.append(_row_to_security(row))
        return results

    def persist_security(self, security: Security) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "{CALL PersistSecurity (?, ?, ?, ?, ?, ?)}",
                (security.symbol, security.security_type, security.description,
                 security.maturity_date, security.final_payment, security.currency),
            )
            self.connection.commit()

    def delete_security(self, symbol: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("{CALL DeleteSecurity (?)}", (symbol,))
            self.connection.commit()

    def bulk_upsert_securities(self, securities: list[Security]) -> int:
        payload = json.dumps([
            {"symbol": s.symbol, "security_type": s.security_type,
             "description": s.description, "maturity_date": s.maturity_date,
             "final_payment": s.final_payment, "currency": s.currency}
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

    # ======================================================================
    # BOND DETAIL
    # ======================================================================

    def get_bond_details(self, symbol: Optional[str] = None) -> list[BondDetail]:
        """Returns bond detail rows joined with securities. None = all bonds."""
        results = []
        with self.connection.cursor() as cursor:
            cursor.execute("{CALL GetBondDetail (?, ?)}", (symbol, 2))
            for row in cursor:
                results.append(_row_to_bond_detail(row))
        return results

    def get_bond_detail(self, symbol: str) -> Optional[BondDetail]:
        rows = self.get_bond_details(symbol=symbol)
        return rows[0] if rows else None

    def persist_bond_detail(
        self,
        symbol:     str,
        law:        str,
        par_symbol: Optional[str] = None,
    ) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "{CALL PersistBondDetail (?, ?, ?)}",
                (symbol, law, par_symbol),
            )
            self.connection.commit()

    # ======================================================================
    # BOND COUPONS
    # ======================================================================

    def get_bond_coupons(
        self,
        symbol:      str,
        paid_filter: int = 2,
    ) -> list[BondCoupon]:
        """
        Returns coupon schedule for a bond.

        paid_filter:
            SecurityManager.COUPONS_FUTURE (0) — future only
            SecurityManager.COUPONS_PAID   (1) — paid only
            SecurityManager.COUPONS_ALL    (2) — all (default)
        """
        results = []
        with self.connection.cursor() as cursor:
            cursor.execute("{CALL GetBondCoupons (?, ?)}", (symbol, paid_filter))
            for row in cursor:
                results.append(_row_to_bond_coupon(row))
        return results

    def persist_bond_coupon(
        self,
        symbol:       str,
        payment_date: str,
        amount:       float,
        is_paid:      bool,
    ) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "{CALL PersistBondCoupon (?, ?, ?, ?)}",
                (symbol, payment_date, amount, 1 if is_paid else 0),
            )
            self.connection.commit()

    def bulk_upsert_bond_coupons(self, coupons: list[BondCoupon]) -> int:
        """Upsert a list of BondCoupon objects. Returns rows affected."""
        payload = json.dumps([
            {
                "symbol":       c.symbol,
                "payment_date": c.payment_date,
                "amount":       c.amount,
                "is_paid":      1 if c.is_paid else 0,
            }
            for c in coupons
        ])
        rows_affected = 0
        with self.connection.cursor() as cursor:
            cursor.execute("{CALL BulkUpsertBondCoupons (?)}", (payload,))
            row = cursor.fetchone()
            if row:
                rows_affected = int(row[0])
            self.connection.commit()
        return rows_affected

    def mark_coupons_paid(self, symbol: Optional[str] = None) -> int:
        """
        Sets is_paid=1 for all coupons whose payment_date <= today.
        Pass symbol to limit to one bond, None for all bonds.
        Returns rows updated.
        """
        rows_marked = 0
        with self.connection.cursor() as cursor:
            cursor.execute("{CALL MarkCouponsPaid (?)}", (symbol,))
            row = cursor.fetchone()
            if row:
                rows_marked = int(row[0])
            self.connection.commit()
        return rows_marked