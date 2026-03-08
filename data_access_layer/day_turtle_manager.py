import pyodbc
from typing import Optional
from datetime import datetime

from business_entities.trading_strategy_entities import (
    TradingStrategy, StrategyDatabase, SecuritySummary, TradeSignal,
    MonthlyPerformance, SymbolConfig
)


class DailyTurtleManager:

    def __init__(self, connection_string: str):
        self._cs = connection_string

    def _connect(self) -> pyodbc.Connection:
        return pyodbc.connect(self._cs)

    @staticmethod
    def _row_to_strategy(row) -> TradingStrategy:
        return TradingStrategy(
            strategy_id   = row.strategy_id,
            strategy_name = row.strategy_name,
            exchange      = row.exchange or "NYSE",
            description   = row.description,
            is_active     = bool(row.is_active),
            created_at    = row.created_at,
            updated_at    = row.updated_at,
        )

    @staticmethod
    def _row_to_strategy_db(row) -> StrategyDatabase:
        return StrategyDatabase(
            db_id         = row.db_id,
            strategy_id   = row.strategy_id,
            database_name = row.database_name,
            label         = row.label,
            is_default    = bool(row.is_default),
            created_at    = row.created_at,
        )

    @staticmethod
    def _row_to_security(row) -> SecuritySummary:
        return SecuritySummary(
            symbol        = row.symbol,
            trade_count   = int(row.trade_count),
            first_trade   = row.first_trade,
            last_trade    = row.last_trade,
            closed_trades = int(row.closed_trades),
            open_trades   = int(row.open_trades),
            total_profit  = float(row.total_profit or 0),
            avg_profit    = float(row.avg_profit   or 0),
        )

    @staticmethod
    def _row_to_trade(row) -> TradeSignal:
        def _f(v): return float(v) if v is not None else None
        return TradeSignal(
            id                   = int(row.id),
            strategy_name        = row.strategy_name,
            opening_date         = row.opening_date,
            closing_date         = row.closing_date,
            symbol               = row.symbol,
            qty                  = _f(row.qty) or 0,
            trade_direction      = row.trade_direction or "",
            opening_price        = _f(row.opening_price),
            closing_price        = _f(row.closing_price),
            last_price           = _f(row.last_price),
            total_fee            = _f(row.total_fee),
            initial_cap          = _f(row.initial_cap),
            final_cap            = _f(row.final_cap),
            profit               = _f(row.profit),
            nominal_profit       = _f(row.nominal_profit),
            fee_type             = row.fee_type,
            fee_value            = _f(row.fee_value),
            trendline_start_date = row.trendline_start_date,
            trendline_end_date   = row.trendline_end_date,
        )

    @staticmethod
    def _row_to_monthly(row) -> MonthlyPerformance:
        def _f(v): return float(v) if v is not None else 0.0
        return MonthlyPerformance(
            symbol               = row.symbol,
            yr                   = int(row.yr),
            mo                   = int(row.mo),
            trade_count          = int(row.trade_count),
            total_profit         = _f(row.total_profit),
            total_nominal_profit = _f(row.total_nominal_profit),
            avg_profit           = _f(row.avg_profit),
            winning_trades       = int(row.winning_trades),
            losing_trades        = int(row.losing_trades),
        )

    @staticmethod
    def _row_to_symbol_config(row) -> SymbolConfig:
        return SymbolConfig(
            symbol     = row.symbol,
            exchange   = row.exchange,
            notes      = row.notes,
            updated_at = row.updated_at,
        )

    # ── Strategy CRUD ─────────────────────────────────────────────────────────

    def get_all_strategies(self) -> list[TradingStrategy]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.get_all_trading_strategies")
            return [self._row_to_strategy(r) for r in cursor.fetchall()]

    def get_strategy(self, strategy_id: int) -> Optional[TradingStrategy]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.get_trading_strategy @strategy_id = ?", strategy_id)
            row = cursor.fetchone()
            return self._row_to_strategy(row) if row else None

    def persist_strategy(self, entity: TradingStrategy) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.persist_trading_strategy "
                "@strategy_id = ?, @strategy_name = ?, @exchange = ?, @description = ?, @is_active = ?",
                entity.strategy_id, entity.strategy_name,
                entity.exchange or "NYSE", entity.description,
                1 if entity.is_active else 0,
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row[0]) if row else entity.strategy_id

    def delete_strategy(self, strategy_id: int) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.delete_trading_strategy @strategy_id = ?", strategy_id)
            conn.commit()

    # ── StrategyDatabase CRUD ─────────────────────────────────────────────────

    def get_strategy_databases(self, strategy_id: int) -> list[StrategyDatabase]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.get_all_strategy_databases @strategy_id = ?", strategy_id)
            return [self._row_to_strategy_db(r) for r in cursor.fetchall()]

    def persist_strategy_database(self, entity: StrategyDatabase) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.persist_strategy_database "
                "@db_id = ?, @strategy_id = ?, @database_name = ?, @label = ?, @is_default = ?",
                entity.db_id, entity.strategy_id, entity.database_name,
                entity.label, 1 if entity.is_default else 0,
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row[0]) if row else entity.db_id

    def delete_strategy_database(self, db_id: int) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.delete_strategy_database @db_id = ?", db_id)
            conn.commit()

    # ── Symbol config ─────────────────────────────────────────────────────────

    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.get_symbol_config @symbol = ?", symbol)
            row = cursor.fetchone()
            return self._row_to_symbol_config(row) if row else None

    def get_all_symbol_configs(self) -> list[SymbolConfig]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC dbo.get_all_symbol_configs")
            return [self._row_to_symbol_config(r) for r in cursor.fetchall()]

    def persist_symbol_config(self, entity: SymbolConfig) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.persist_symbol_config @symbol = ?, @exchange = ?, @notes = ?",
                entity.symbol, entity.exchange, entity.notes,
            )
            conn.commit()

    # ── Securities ────────────────────────────────────────────────────────────

    def get_securities_for_strategy(self, strategy_name: str, database_name: str) -> list[SecuritySummary]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.get_securities_for_strategy @strategy_name = ?, @database_name = ?",
                strategy_name, database_name,
            )
            return [self._row_to_security(r) for r in cursor.fetchall()]

    # ── Trades ────────────────────────────────────────────────────────────────

    def get_trades(self, strategy_name: str, database_name: str, symbol: str,
                   date_from: Optional[datetime] = None,
                   date_to:   Optional[datetime] = None) -> list[TradeSignal]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.get_trades_for_strategy_symbol "
                "@strategy_name = ?, @database_name = ?, @symbol = ?, @date_from = ?, @date_to = ?",
                strategy_name, database_name, symbol, date_from, date_to,
            )
            return [self._row_to_trade(r) for r in cursor.fetchall()]

    def get_today_signals(self, strategy_name: str, database_name: str) -> list[TradeSignal]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.get_today_signals @strategy_name = ?, @database_name = ?",
                strategy_name, database_name,
            )
            return [self._row_to_trade(r) for r in cursor.fetchall()]

    # ── Monthly performance ───────────────────────────────────────────────────

    def get_monthly_performance(self, strategy_name: str, database_name: str,
                                 symbol: Optional[str] = None) -> list[MonthlyPerformance]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.get_monthly_performance "
                "@strategy_name = ?, @database_name = ?, @symbol = ?",
                strategy_name, database_name, symbol,
            )
            return [self._row_to_monthly(r) for r in cursor.fetchall()]