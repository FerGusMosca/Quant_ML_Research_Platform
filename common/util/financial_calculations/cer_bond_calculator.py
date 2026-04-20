"""
cer_bond_calculator.py
======================
Cash-flow computation for CER-indexed Argentine sovereign bonds.

CER bond mechanics
------------------
The nominal flow per 1 VN is computed from the bond's coupon schedule
(amortization % + interest rate + day-count base, each per period) and
then multiplied by the CER adjustment coefficient:

    FF_nominal    = (VR_antes × tasa_interes × base) + amortizacion
    FF_ajustado   = FF_nominal × (CER_t / CER_emision)

Where:
    VR_antes      = residual value before this period = 1 - Σ amort. previous
    tasa_interes  = period interest rate (decimal, e.g. 0.0225 = 2.25%)
    base          = day-count fraction (e.g. 0.5 for semi-annual)
    amortizacion  = fraction of principal paid this period (decimal)

Special case — DICP
-------------------
DICP accumulated interest during the grace period (2004-2014) when coupons
were capitalized. Market practice multiplies its CER coefficient by 1.27
to account for the capitalized history. PARP does NOT get this factor.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from scipy.optimize import brentq

DAYS_IN_YEAR = 365.25
_DICP_CAPITALIZED_INTEREST_FACTOR = 1.27


@dataclass
class CerBondCalculationResult:
    tir_real:   Optional[float]     # as decimal (0.07 = 7%)
    duration:   Optional[float]     # Macaulay, years
    flows_used: list = field(default_factory=list)


@dataclass
class CerBondInvestmentResult:
    symbol:          str
    effective_price: float          # per 100 VN (with costs)
    vn_bought:       float          # nominal units purchased
    monto_real:      float          # actual amount invested (ARS)
    flows:           list           # [{date, per1vn, cobro}]
    total_cobro:     float
    ganancia:        float
    tir_real:        Optional[float]
    duration:        Optional[float]


class CerBondCalculator:
    """
    Stateless calculator. Safe to instantiate once and reuse.
    """

    def compute_adjusted_flows(
        self,
        raw_flows:    list[dict],
        cer_emision:  float,
        cer_current:  float,
        symbol:       str            = "",
        today:        Optional[date] = None,
    ) -> list[dict]:
        """
        Build the list of future cash flows adjusted by CER, per 1 VN.

        Parameters
        ----------
        raw_flows    : list of {fecha, amortizacion, tasa_interes, base}
                       (exactly the schema stored in bonds_config.json)
        cer_emision  : CER value at bond issue date
        cer_current  : CER value used for indexation (market convention: T-10)
        symbol       : bond symbol — needed to detect DICP special case
        today        : reference date for filtering future flows (default: today)

        Returns
        -------
        list of {date: 'YYYY-MM-DD', amount: float}  — only future flows,
        sorted ascending by date, amounts per 1 VN already CER-adjusted.
        """
        if today is None:
            today = date.today()
        if not cer_emision or cer_emision <= 0 or not cer_current or cer_current <= 0:
            return []

        coef = cer_current / cer_emision
        if symbol.upper() == "DICP":
            coef *= _DICP_CAPITALIZED_INTEREST_FACTOR

        # First pass — walk ALL flows in order and compute VR_antes for each.
        # We can't skip past flows here because VR_antes depends on cumulative
        # amortization from the beginning of the bond's life.
        amort_acum = 0.0
        augmented: list[dict] = []
        for f in raw_flows:
            vr_antes = 1.0 - amort_acum
            amort_acum += float(f.get("amortizacion", 0) or 0)
            augmented.append({**f, "vr_antes": vr_antes})

        # Second pass — keep only future flows, compute CER-adjusted amount.
        out: list[dict] = []
        for f in augmented:
            fecha = _parse_date(f["fecha"])
            if fecha <= today:
                continue
            interes    = f["vr_antes"] * float(f.get("tasa_interes", 0) or 0) * float(f.get("base", 1) or 1)
            amort      = float(f.get("amortizacion", 0) or 0)
            ff_nominal = interes + amort
            out.append({
                "date":   fecha.isoformat(),
                "amount": ff_nominal * coef,
            })

        out.sort(key=lambda x: x["date"])
        return out

    def calculate(
        self,
        price_ars:    float,
        raw_flows:    list[dict],
        cer_emision:  float,
        cer_current:  float,
        symbol:       str            = "",
        today:        Optional[date] = None,
    ) -> CerBondCalculationResult:
        """
        Compute TIR Real (IRR on CER-adjusted flows) and Macaulay duration.

        Convention: price_ars is per 100 VN (the BYMA quote). Flows are per
        1 VN. We compare them on the same basis by using price_ars / 100.
        """
        adjusted = self.compute_adjusted_flows(
            raw_flows    = raw_flows,
            cer_emision  = cer_emision,
            cer_current  = cer_current,
            symbol       = symbol,
            today        = today,
        )
        if today is None:
            today = date.today()

        price_per_vn = (price_ars or 0) / 100.0
        if price_per_vn <= 0 or not adjusted:
            return CerBondCalculationResult(tir_real=None, duration=None, flows_used=adjusted)

        tir, dur = _tir_duration(adjusted, price_per_vn, today)
        return CerBondCalculationResult(tir_real=tir, duration=dur, flows_used=adjusted)

    def calculate_investment(
        self,
        symbol:       str,
        price_ars:    float,
        raw_flows:    list[dict],
        cer_emision:  float,
        cer_current:  float,
        monto:        float,
        arancel_pct:  float = 0.45,
        impuesto_pct: float = 0.01,
        today:        Optional[date] = None,
    ) -> CerBondInvestmentResult:
        """
        Full investment calc for the frontend calculator.

        price_ars is per 100 VN. Costs are in %; effective price adds them.
        """
        if today is None:
            today = date.today()

        eff = price_ars * (1 + (arancel_pct + impuesto_pct) / 100)
        # Position sizing: per 100 VN
        vn_bought = monto / (eff / 100.0)
        monto_real = vn_bought * (eff / 100.0)

        calc = self.calculate(
            price_ars    = eff,
            raw_flows    = raw_flows,
            cer_emision  = cer_emision,
            cer_current  = cer_current,
            symbol       = symbol,
            today        = today,
        )

        rows = [
            {
                "date":   f["date"],
                "per1vn": round(f["amount"], 6),
                "cobro":  round(vn_bought * f["amount"], 2),
            }
            for f in calc.flows_used
        ]
        total = sum(r["cobro"] for r in rows)

        return CerBondInvestmentResult(
            symbol          = symbol,
            effective_price = round(eff, 4),
            vn_bought       = round(vn_bought, 0),
            monto_real      = round(monto_real, 2),
            flows           = rows,
            total_cobro     = round(total, 2),
            ganancia        = round(total - monto_real, 2),
            tir_real        = calc.tir_real,
            duration        = calc.duration,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_date(s) -> date:
    if isinstance(s, date):
        return s
    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()


def _tir_duration(
    flows:      list[dict],
    price:      float,
    today:      date,
) -> tuple[Optional[float], Optional[float]]:
    """Newton-Raphson-safe TIR via brentq; Macaulay duration at that TIR."""
    if not flows or price <= 0:
        return None, None
    times = [
        (_parse_date(f["date"]) - today).days / DAYS_IN_YEAR
        for f in flows
    ]
    if all(t <= 0 for t in times):
        return None, None

    def npv(r: float) -> float:
        return sum(
            f["amount"] / (1 + r) ** t
            for f, t in zip(flows, times) if t > 0
        ) - price

    try:
        tir  = brentq(npv, -0.999, 20.0)
        base = 1 + tir
        pvs  = [f["amount"] / base ** t for f, t in zip(flows, times) if t > 0]
        ts   = [t                         for t in times                     if t > 0]
        total_pv = sum(pvs)
        if total_pv <= 0:
            return float(tir), None
        dur = sum(t * pv for t, pv in zip(ts, pvs)) / total_pv
        return float(tir), float(dur)
    except Exception:
        return None, None
