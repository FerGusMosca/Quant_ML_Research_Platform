"""
data912_downloader.py
=====================
Reusable async downloader for the data912.com live market data API.

All pages that need live Argentine market prices (sovereign bonds, LECAPs,
BONCAPs, options, etc.) should import and use this class instead of
crafting raw httpx calls directly.

Endpoints currently supported
------------------------------
Endpoint constant        URL suffix              Typical content
--------------------     ----------------------  -----------------------
ENDPOINT_ARG_BONDS       /live/arg_bonds         Sovereign bonds (AL, GD, …)
ENDPOINT_ARG_NOTES       /live/arg_notes         LECAPs, short-term notes
ENDPOINT_ARG_OPTIONS     /live/arg_options       Listed options (future)
ENDPOINT_ARG_EQUITIES    /live/arg_equities      Local equities (future)

Usage example (inside an async FastAPI handler)
-----------------------------------------------
    from common.util.downloaders.data912_downloader import Data912Downloader

    dl = Data912Downloader()

    # Single endpoint
    bonds = await dl.fetch(Data912Downloader.ENDPOINT_ARG_BONDS)

    # Multiple endpoints concurrently
    bonds, notes = await dl.fetch_many(
        Data912Downloader.ENDPOINT_ARG_BONDS,
        Data912Downloader.ENDPOINT_ARG_NOTES,
    )

    # Filter by symbol set
    filtered = dl.filter_by_symbols(bonds, {'AL30D', 'GD35D'})

    # Extract standardised price dict for a single item
    price = dl.parse_price_item(bonds[0])
    # → {'symbol': 'AL30D', 'price': 62.91, 'bid': 62.85, 'ask': 62.97,
    #    'pct_change': 0.49, 'volume': 0}
"""

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://data912.com"

_DEFAULT_TIMEOUT_SECONDS = 15


class Data912Downloader:
    """
    Async HTTP client for data912.com live market data.

    The class is stateless — instantiate it per-request or share a single
    instance; either pattern is safe.
    """

    # ── Supported endpoint paths ─────────────────────────────────────────
    ENDPOINT_ARG_BONDS    = "/live/arg_bonds"    # sovereign bonds
    ENDPOINT_ARG_NOTES    = "/live/arg_notes"    # LECAPs / short notes
    ENDPOINT_ARG_OPTIONS  = "/live/arg_options"  # listed options (future)
    ENDPOINT_ARG_EQUITIES = "/live/arg_equities" # local equities (future)
    ENDPOINT_ARG_CORP = "/live/arg_corp"

    def __init__(self, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS):
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, endpoint: str) -> list[dict]:
        """
        Fetch a single data912 endpoint and return the raw JSON list.

        Parameters
        ----------
        endpoint : one of the ENDPOINT_* class constants, e.g.
                   Data912Downloader.ENDPOINT_ARG_BONDS

        Returns
        -------
        list[dict] : raw items from the API, or [] on error.
        """
        url = _BASE_URL + endpoint
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("[Data912] fetch error (%s): %s", endpoint, exc)
            return []

    async def fetch_many(self, *endpoints: str) -> tuple[list[dict], ...]:
        """
        Fetch multiple endpoints concurrently.

        Parameters
        ----------
        *endpoints : any number of ENDPOINT_* constants.

        Returns
        -------
        tuple of lists, one per endpoint, in the same order.
        Failed endpoints return an empty list — they do NOT raise.

        Example
        -------
            bonds, notes = await dl.fetch_many(
                Data912Downloader.ENDPOINT_ARG_BONDS,
                Data912Downloader.ENDPOINT_ARG_NOTES,
            )
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [self._fetch_with_client(client, ep) for ep in endpoints]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return tuple(
            r if isinstance(r, list) else []
            for r in results
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_symbols(items: list[dict], symbols: set[str]) -> list[dict]:
        """
        Return only items whose 'symbol' field is in *symbols*.

        Parameters
        ----------
        items   : raw list returned by fetch() / fetch_many()
        symbols : set of symbol strings to keep (e.g. {'AL30D', 'GD35D'})
        """
        return [it for it in items if it.get("symbol") in symbols]

    @staticmethod
    def parse_price_item(item: dict) -> dict:
        """
        Normalise a single raw API item into a standardised price dict.

        Raw data912 field mapping
        --------------------------
        c          → price   (last / close)
        px_bid     → bid
        px_ask     → ask
        pct_change → pct_change
        v          → volume

        Returns
        -------
        dict with keys: symbol, price, bid, ask, pct_change, volume
        All numeric fields default to 0.0 / 0 if absent or null.
        """
        return {
            "symbol":     str(item.get("symbol") or ""),
            "price":      float(item.get("c")          or 0),
            "bid":        float(item.get("px_bid")      or 0),
            "ask":        float(item.get("px_ask")      or 0),
            "pct_change": float(item.get("pct_change")  or 0),
            "volume":     int(item.get("v")             or 0),
        }

    @staticmethod
    def build_price_map(
        items:   list[dict],
        symbols: Optional[set[str]] = None,
    ) -> dict[str, dict]:
        """
        Build a symbol → parsed-price dict for fast O(1) lookups.

        Parameters
        ----------
        items   : raw list from fetch()
        symbols : optional allow-list; if None, includes all items.

        Returns
        -------
        dict[str, dict]  e.g. {'AL30D': {'symbol': 'AL30D', 'price': 62.91, …}}
        """
        result: dict[str, dict] = {}
        for item in items:
            parsed = Data912Downloader.parse_price_item(item)
            sym    = parsed["symbol"]
            if sym and (symbols is None or sym in symbols):
                result[sym] = parsed
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fetch_with_client(self, client: httpx.AsyncClient, endpoint: str) -> list[dict]:
        url = _BASE_URL + endpoint
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("[Data912] fetch error (%s): %s", endpoint, exc)
            return []