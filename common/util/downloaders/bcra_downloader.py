"""
bcra_downloader.py
==================
Reusable async downloader for BCRA (Banco Central de la República Argentina)
public statistics API v3.0.

All pages that need BCRA series (CER, UVA, UVI, BADLAR, TAMAR, reservas, etc.)
should import and use this class instead of crafting raw httpx calls directly.

Variable IDs (dbo.bcra_monetarias)
----------------------------------
 1   Reservas Internacionales (USD M)
 4   Tipo de cambio mayorista
 5   Tipo de cambio minorista
 6   BADLAR en pesos bancos privados (TNA %)
 7   TM20 en pesos bancos privados (TNA %)
15   Base Monetaria (ARS M)
27   Inflación mensual (%)
28   Inflación interanual (%)
29   Inflación esperada próx. 12 meses (%)
30   CER (Coeficiente de Estabilización de Referencia)
31   UVA (Unidad de Valor Adquisitivo)
32   UVI (Unidad de Vivienda)
34   TAMAR bancos privados (TNA %)

Usage example
-------------
    from common.util.downloaders.bcra_downloader import BcraDownloader

    dl = BcraDownloader()

    # Latest CER value
    latest = await dl.get_variable_latest(BcraDownloader.CER)
    # → {"fecha": "2026-04-15", "valor": 770.8982}

    # CER lagged 10 business days (what CER-bond flows are indexed to)
    lagged = await dl.get_variable_lagged(BcraDownloader.CER, lag_business_days=10)

    # Historical series
    series = await dl.get_variable_series(
        BcraDownloader.CER,
        desde="2026-01-01",
        hasta="2026-04-15",
    )
"""

import asyncio
import logging
import ssl
from datetime import date, datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.bcra.gob.ar/estadisticas/v3.0"
_DEFAULT_TIMEOUT_SECONDS = 15


class BcraDownloader:
    """
    Async HTTP client for BCRA statistics API v3.0.

    The class is stateless — instantiate it per-request or share a single
    instance; either pattern is safe. The caller is expected to handle
    its own caching; this downloader does NOT cache results internally.
    """

    # ── Variable IDs ─────────────────────────────────────────────────
    RESERVAS_USD_M          = 1
    TC_MAYORISTA            = 4
    TC_MINORISTA            = 5
    BADLAR_TNA              = 6
    TM20_TNA                = 7
    BASE_MONETARIA          = 15
    INFLACION_MENSUAL       = 27
    INFLACION_INTERANUAL    = 28
    INFLACION_ESPERADA_12M  = 29
    CER                     = 30
    UVA                     = 31
    UVI                     = 32
    TAMAR_TNA               = 34

    def __init__(
        self,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        verify_ssl: bool = False,
    ):
        """
        Parameters
        ----------
        timeout_seconds : request timeout (default 15s)
        verify_ssl      : BCRA's chain used to fail on some Linux boxes;
                          default False because this is a public read-only
                          endpoint and spoofing buys an attacker nothing.
                          Flip to True if your CA bundle has Sectigo root.
        """
        self._timeout = timeout_seconds
        self._verify  = verify_ssl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_variable_series(
        self,
        id_variable: int,
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
        limit: int = 3000,
        offset: int = 0,
    ) -> list[dict]:
        """
        Fetch the historical series for a BCRA variable.

        Parameters
        ----------
        id_variable : BCRA variable id (see class constants)
        desde       : ISO date 'YYYY-MM-DD' (optional)
        hasta       : ISO date 'YYYY-MM-DD' (optional)
        limit       : max rows to return (BCRA caps at 3000)
        offset      : pagination offset

        Returns
        -------
        list[dict] with keys: fecha (ISO), valor (float)
        Sorted descending by date (most recent first) — that's how BCRA
        returns it and we preserve their ordering.
        """
        path   = f"/monetarias/{id_variable}"
        params = {"limit": limit, "offset": offset}
        if desde: params["desde"] = desde
        if hasta: params["hasta"] = hasta

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                verify=self._verify,
            ) as client:
                resp = await client.get(_BASE_URL + path, params=params)
                resp.raise_for_status()
                data = resp.json()

            # BCRA wraps payload in {status, results: [{idVariable, detalle: [...]}, ...]}
            results = data.get("results") or []
            if not results:
                return []
            detalle = results[0].get("detalle") or []
            return [
                {"fecha": str(r.get("fecha")), "valor": float(r.get("valor"))}
                for r in detalle
                if r.get("valor") is not None
            ]
        except Exception as exc:
            logger.warning("[BCRA] get_variable_series(%s) error: %s", id_variable, exc)
            return []

    async def get_variable_latest(self, id_variable: int) -> Optional[dict]:
        """
        Latest published value for a BCRA variable.
        Returns {fecha, valor} or None on error / no data.
        """
        # Fetch last 30 days — more than enough to always find something
        hasta = date.today().isoformat()
        desde = (date.today() - timedelta(days=30)).isoformat()
        series = await self.get_variable_series(id_variable, desde=desde, hasta=hasta)
        if not series:
            return None
        # BCRA returns desc; first element is the latest
        return series[0]

    async def get_variable_lagged(
        self,
        id_variable: int,
        lag_business_days: int = 10,
    ) -> Optional[dict]:
        """
        Fetch the value from N business days ago.

        For CER-indexed bonds the market uses CER T-10 (10 business days
        rezagado) per prospectus — that's the canonical use case here.

        The BCRA only publishes variables on business days, so the N-th
        item counting back from the latest is the N-business-days lag.
        (Holidays also skipped; BCRA doesn't publish on ARG bank holidays.)

        Returns {fecha, valor} or None.
        """
        # 45 days window covers N=10 even with an unusually long holiday streak
        hasta = date.today().isoformat()
        desde = (date.today() - timedelta(days=45)).isoformat()
        series = await self.get_variable_series(id_variable, desde=desde, hasta=hasta)
        if len(series) <= lag_business_days:
            # Not enough history returned — fall back to oldest available
            return series[-1] if series else None
        # series is desc; index `lag_business_days` is N business days before latest
        return series[lag_business_days]

    async def get_many_latest(self, *id_variables: int) -> dict[int, Optional[dict]]:
        """
        Fetch latest values for multiple variables concurrently.

        Returns
        -------
        dict mapping id_variable → {fecha, valor} or None.
        """
        tasks = [self.get_variable_latest(v) for v in id_variables]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[int, Optional[dict]] = {}
        for vid, r in zip(id_variables, results):
            out[vid] = r if isinstance(r, dict) else None
        return out
