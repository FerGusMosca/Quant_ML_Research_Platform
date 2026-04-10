from dataclasses import dataclass
from datetime import date
from typing import Optional
from scipy.optimize import brentq

PRICE_UNIT   = 100.0
DAYS_IN_YEAR = 365.25


@dataclass
class BondCalculationResult:
    tir: Optional[float]; duration: Optional[float]; is_bullet: bool; flows_used: list


@dataclass
class InvestmentResult:
    symbol: str; effective_price: float; vn_bought: int; monto_real: float
    flows: list; total_cobro: float; ganancia: float
    tir: Optional[float]; duration: Optional[float]; is_bullet: bool


class BulletBondCalculator:

    def calculate(self, price: float, coupons: list) -> BondCalculationResult:
        flows = self._future_flows(coupons)
        tir, dur = self._tir_duration(flows, price / PRICE_UNIT)
        return BondCalculationResult(tir=tir, duration=dur, is_bullet=False, flows_used=flows)

    def calculate_investment(self, symbol: str, price: float, coupons: list,
                             monto: float, arancel_pct: float = 0.45,
                             impuesto_pct: float = 0.01) -> InvestmentResult:
        eff   = price * (1 + (arancel_pct + impuesto_pct) / PRICE_UNIT)
        vn    = int(monto / (eff / PRICE_UNIT))
        paid  = vn * (eff / PRICE_UNIT)
        flows = self._future_flows(coupons)
        rows  = [{"date": f["date"], "per100": round(f["amount"], 6),
                  "cobro": round(vn * f["amount"], 2)} for f in flows]
        total = sum(r["cobro"] for r in rows)
        tir, dur = self._tir_duration(flows, eff / PRICE_UNIT)
        return InvestmentResult(symbol=symbol, effective_price=round(eff, 4), vn_bought=vn,
                                monto_real=round(paid, 2), flows=rows, total_cobro=round(total, 2),
                                ganancia=round(total - paid, 2), tir=tir, duration=dur, is_bullet=False)

    @staticmethod
    def _future_flows(coupons: list) -> list:
        today = date.today()
        return sorted([{"date": str(c["date"]), "amount": float(c["amount"])}
                       for c in coupons
                       if (date.fromisoformat(str(c["date"])) if isinstance(c["date"], str)
                           else c["date"]) > today], key=lambda x: x["date"])

    @staticmethod
    def _tir_duration(flows: list, price: float) -> tuple:
        if not flows or price <= 0:
            return None, None
        today = date.today()
        times = [(date.fromisoformat(f["date"]) - today).days / DAYS_IN_YEAR for f in flows]
        npv   = lambda r: sum(f["amount"] / (1+r)**t for f, t in zip(flows, times)) - price
        try:
            tir  = brentq(npv, -0.999, 20.0)
            base = 1 + tir
            pvs  = [f["amount"] / base**t for f, t in zip(flows, times)]
            dur  = sum(t*pv for t, pv in zip(times, pvs)) / sum(pvs)
            return float(tir), float(dur)
        except Exception:
            return None, None