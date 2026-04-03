"""
zero_coupon_calculator.py
=========================
Financial calculations for zero-coupon / capitalising instruments
(LECAPs, BONCAPs, and any instrument with a single terminal payment).

All yields are returned as decimals [0-1].  Multiply by 100 for percentages.

Conventions
-----------
- Settlement: T+1 business day (caller is responsible for passing correct dates)
- Day-count:  Actual/365 for TNA/TIR, DAYS360 (US/NASD) for TEM
- TNA:        Nominal annual rate  = (pago_final/precio - 1) × (365/days)
- TEM:        Monthly effective    = (pago_final/precio)^(1/months) - 1
              where months = DAYS360(settlement, maturity) / 30
- TIR:        Annual effective     = (pago_final/precio)^(365/days) - 1

Usage
-----
    from common.util.financial_calculations.zero_coupon_calculator import ZeroCouponCalculator

    calc = ZeroCouponCalculator()
    result = calc.compute(
        price         = 125.40,
        final_payment = 127.486,
        settlement    = date(2026, 4, 4),   # T+1
        maturity      = date(2026, 4, 30),
    )
    # result.tna, result.tem, result.tir, result.days, result.months
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ZeroCouponResult:
    tna:    Optional[float]   # nominal annual rate     [0-1]
    tem:    Optional[float]   # monthly effective rate  [0-1]
    tir:    Optional[float]   # annual effective rate   [0-1]
    days:   int               # calendar days settlement → maturity
    months: float             # DAYS360 months settlement → maturity


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class ZeroCouponCalculator:
    """
    Computes TNA / TEM / TIR for a zero-coupon (or bullet / capitalising)
    instrument given market price and final payment.

    The instance is stateless — instantiate once and reuse freely.
    """

    def compute(
        self,
        price:         float,
        final_payment: float,
        settlement:    date,
        maturity:      date,
    ) -> ZeroCouponResult:
        """
        Compute yields for a zero-coupon instrument.

        Parameters
        ----------
        price         : market price (same unit as final_payment, e.g. per 100 VN)
        final_payment : terminal cash flow (pago final per 100 VN)
        settlement    : settlement date (T+1 business day from trade date)
        maturity      : maturity / expiry date of the instrument

        Returns
        -------
        ZeroCouponResult with tna / tem / tir as decimals [0-1].
        All fields are None if inputs are invalid or maturity <= settlement.
        """
        days   = (maturity - settlement).days
        months = self._days360(settlement, maturity) / 30

        if price <= 0 or final_payment <= 0 or days <= 0 or months <= 0:
            return ZeroCouponResult(tna=None, tem=None, tir=None,
                                    days=max(days, 0), months=max(months, 0.0))

        ratio = final_payment / price   # gross return factor

        tna = (ratio - 1) * (365 / days)
        tir = ratio ** (365 / days) - 1
        tem = ratio ** (1 / months) - 1

        return ZeroCouponResult(
            tna    = tna    if math.isfinite(tna) else None,
            tem    = tem    if math.isfinite(tem) else None,
            tir    = tir    if math.isfinite(tir) else None,
            days   = days,
            months = months,
        )

    def compute_price_from_tir(
        self,
        target_tir:    float,   # annual effective rate [0-1]
        final_payment: float,
        settlement:    date,
        maturity:      date,
    ) -> Optional[float]:
        """
        Inverse: given a target TIR, return the implied price.

        implied_price = final_payment / (1 + tir)^(days/365)
        """
        days = (maturity - settlement).days
        if days <= 0 or final_payment <= 0:
            return None
        try:
            price = final_payment / ((1 + target_tir) ** (days / 365))
            return price if math.isfinite(price) and price > 0 else None
        except (ZeroDivisionError, OverflowError):
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _days360(start: date, end: date) -> float:
        """
        DAYS360 — US / NASD method (matches Excel DAYS360(start, end, FALSE)).

        Rules
        -----
        - If d1 = 31, set d1 = 30.
        - If d2 = 31 AND d1 >= 30, set d2 = 30.
        - Result = (y2-y1)*360 + (m2-m1)*30 + (d2-d1)
        """
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day,   end.month,   end.year

        if d1 == 31:
            d1 = 30
        if d2 == 31 and d1 >= 30:
            d2 = 30

        return (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)