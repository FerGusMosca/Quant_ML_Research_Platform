from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class TradingStrategy:
    strategy_id:   Optional[int]
    strategy_name: str
    exchange:      str
    description:   Optional[str]
    is_active:     bool
    created_at:    Optional[datetime] = None
    updated_at:    Optional[datetime] = None
    databases:     list = field(default_factory=list)


@dataclass
class StrategyDatabase:
    db_id:         Optional[int]
    strategy_id:   int
    database_name: str
    label:         Optional[str]
    is_default:    bool
    created_at:    Optional[datetime] = None


@dataclass
class SymbolConfig:
    symbol:     str
    exchange:   str
    notes:      Optional[str]
    updated_at: Optional[datetime] = None


@dataclass
class SecuritySummary:
    symbol:        str
    trade_count:   int
    first_trade:   Optional[datetime]
    last_trade:    Optional[datetime]
    closed_trades: int
    open_trades:   int
    total_profit:  float
    avg_profit:    float


@dataclass
class TradeSignal:
    id:                   int
    strategy_name:        str
    opening_date:         datetime
    closing_date:         Optional[datetime]
    symbol:               str
    qty:                  float
    trade_direction:      str
    opening_price:        Optional[float]
    closing_price:        Optional[float]
    last_price:           Optional[float]
    total_fee:            Optional[float]
    initial_cap:          Optional[float]
    final_cap:            Optional[float]
    profit:               Optional[float]
    nominal_profit:       Optional[float]
    fee_type:             Optional[str]
    fee_value:            Optional[float]
    trendline_start_date: Optional[datetime]
    trendline_end_date:   Optional[datetime]

    @property
    def is_closed(self) -> bool:
        return self.closing_date is not None

    @property
    def duration_minutes(self) -> Optional[float]:
        if self.opening_date and self.closing_date:
            return round((self.closing_date - self.opening_date).total_seconds() / 60, 1)
        return None


@dataclass
class MonthlyPerformance:
    symbol:               str
    yr:                   int
    mo:                   int
    trade_count:          int
    total_profit:         float
    total_nominal_profit: float
    avg_profit:           float
    winning_trades:       int
    losing_trades:        int

    @property
    def win_rate(self) -> float:
        return round(self.winning_trades / self.trade_count * 100, 1) if self.trade_count else 0.0

    @property
    def month_label(self) -> str:
        import calendar
        return f"{calendar.month_abbr[self.mo]} {self.yr}"