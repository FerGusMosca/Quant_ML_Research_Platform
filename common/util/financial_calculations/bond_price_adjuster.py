"""
bond_price_adjuster.py
======================
Utilities for computing adjusted bond prices that neutralise ex-coupon
price drops without over-adjusting for older coupons.

Background
----------
When a bond pays a coupon, the market price drops roughly by the coupon
amount on the ex-date.  Naively summing ALL historical coupons into every
bar inflates old bars, producing a false upward trend in the adjusted series.

The correct approach for a *trailing* adjusted series is:

    adjusted_price(bar) = raw_price(bar) + sum_of_coupons_paid_AFTER bar_date

This way:
  - Bars before the ex-date are untouched (no coupon has been paid yet
    relative to them).
  - Bars on/after the ex-date get the coupon added, so the level is
    restored to where it was before the drop.
  - Only the coupon that *actually caused the price drop* is reflected
    on each bar — not all historical coupons.

Example (GD30, Jan-2026 coupon = 8.27 USD per 100 VN)
-------------------------------------------------------
  raw price pre  Jan-08 ≈ 70.00
  raw price post Jan-09 ≈ 61.20   (drop ≈ 8.80, market inefficiency included)
  adjusted post  Jan-09 = 61.20 + 8.27 = 69.47  → no visible jump  ✓

Usage
-----
    from common.util.financial_calculations.bond_price_adjuster import BondPriceAdjuster

    adjuster = BondPriceAdjuster()

    adjusted_bars = adjuster.apply_trailing_adjustment(
        bars    = raw_ohlcv_bars,          # list[dict] with 'time' (unix ts) keys
        coupons = paid_coupons,            # list[dict] with 'date' (YYYY-MM-DD) and
                                           #   'amount_per_100vn' keys
    )
"""

from __future__ import annotations

from typing import NamedTuple


class CouponEvent(NamedTuple):
    date:             str    # YYYY-MM-DD  (ex-date / payment date)
    amount_per_100vn: float  # coupon amount per 100 face-value


class BondPriceAdjuster:
    """
    Computes trailing-adjusted OHLCV bars for fixed-income instruments.

    Trailing adjustment
    -------------------
    For each bar at date D, the adjustment factor is the sum of all coupons
    whose ex-date is STRICTLY AFTER D.  Older bars therefore receive a
    larger offset (they missed more future coupons), newer bars receive less,
    and bars after the *most recent* coupon receive zero adjustment.

    This is the same convention used by Bloomberg's "total return" adjusted
    price series and by most professional fixed-income charting tools.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply_trailing_adjustment(
        self,
        bars:    list[dict],
        coupons: list[dict],
    ) -> list[dict]:
        """
        Return a new list of OHLCV bars with prices adjusted for coupon drops.

        Parameters
        ----------
        bars
            List of dicts, each with keys:
                time  : int  — Unix timestamp (seconds, UTC midnight for daily bars)
                open  : float
                high  : float
                low   : float
                close : float
            Bars need not be sorted — the method sorts internally.

        coupons
            List of dicts (or CouponEvent-like objects), each with:
                date             : str   — 'YYYY-MM-DD' (ex-date or payment date)
                amount_per_100vn : float — coupon amount per 100 face value

        Returns
        -------
        list[dict]
            New list of bars with adjusted open/high/low/close values.
            The 'time' (and 'volume' if present) fields are unchanged.
            Bars are returned sorted ascending by time.
        """
        if not bars or not coupons:
            return sorted(bars, key=lambda b: b["time"])

        # Parse and sort coupons ascending
        parsed_coupons = sorted(
            [self._parse_coupon(c) for c in coupons],
            key=lambda c: c.date,
        )

        # Total of all coupons — used as the starting offset for the oldest bar
        total = sum(c.amount_per_100vn for c in parsed_coupons)

        # Sort bars ascending so we can walk them left-to-right
        sorted_bars = sorted(bars, key=lambda b: b["time"])

        ci          = 0    # coupon pointer
        running_sum = 0.0  # accumulates coupons paid ON OR BEFORE bar_date

        adjusted: list[dict] = []

        for bar in sorted_bars:
            bar_date = _timestamp_to_date(bar["time"])

            # Add every coupon whose ex-date is <= bar_date.
            # Bars on/after a coupon date get that coupon summed in — restoring
            # the price level after the ex-date drop.
            while ci < len(parsed_coupons) and parsed_coupons[ci].date <= bar_date:
                running_sum += parsed_coupons[ci].amount_per_100vn
                ci += 1

            offset = running_sum  # 0.0 for bars before the first coupon

            new_bar = dict(bar)   # shallow copy — preserves 'volume' etc.
            new_bar["open"]  = round(bar["open"]  + offset, 4)
            new_bar["high"]  = round(bar["high"]  + offset, 4)
            new_bar["low"]   = round(bar["low"]   + offset, 4)
            new_bar["close"] = round(bar["close"] + offset, 4)
            adjusted.append(new_bar)

        return adjusted

    def get_adjustment_offset(
        self,
        bar_date: str,
        coupons:  list[dict],
    ) -> float:
        """
        Return the adjustment offset (in price units per 100 VN) for a
        single bar date — i.e. the sum of all coupon amounts paid strictly
        after *bar_date*.

        Useful for displaying the offset value in tooltips or stats bars
        without recomputing the full series.

        Parameters
        ----------
        bar_date : 'YYYY-MM-DD'
        coupons  : same format as apply_trailing_adjustment()
        """
        return sum(
            c["amount_per_100vn"]
            for c in coupons
            if self._parse_coupon(c).date > bar_date
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_coupon(c) -> CouponEvent:
        """Accept either a dict or a CouponEvent / dataclass-like object."""
        if isinstance(c, CouponEvent):
            return c
        return CouponEvent(
            date             = str(c["date"]),
            amount_per_100vn = float(c["amount_per_100vn"]),
        )


# ---------------------------------------------------------------------------
# Module-level helper (no need to instantiate the class for this)
# ---------------------------------------------------------------------------

def _timestamp_to_date(ts: int) -> str:
    """
    Convert a Unix timestamp (seconds) to a 'YYYY-MM-DD' string.
    Reads as UTC so dates match coupon date strings regardless of
    the server's local timezone.
    """
    import datetime
    d = datetime.datetime.utcfromtimestamp(ts)
    return d.strftime("%Y-%m-%d")