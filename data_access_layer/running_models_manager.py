"""
data_access_layer/running_models_manager.py
DAL for running_model_configs + running_model_series tables (hist_data DB).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pyodbc


# ── DTOs ──────────────────────────────────────────────────────────────────────

@dataclass
class RunningModelConfigDTO:
    model_id:               int
    model_name:             str
    algo_type:              str          # XGBOOST | RF | LSTM
    model_path:             str
    symbol:                 str
    bias:                   str
    d_from:                 str          # ISO date string YYYY-MM-DD
    d_to:                   str
    lower_percentile_limit: float
    n_flip:                 int
    make_stationary:        bool
    draw_predictions:       bool
    init_portf_size:        float
    trade_comm:             float
    display_order:          int
    is_active:              bool
    series_csv:             str          # comma-separated series symbols


# ── Manager ───────────────────────────────────────────────────────────────────

class RunningModelsManager:
    """
    CRUD for running_model_configs and running_model_series.
    Uses hist_data connection string (same DB as EconomicSeriesManager).
    """

    def __init__(self, connection_string: str, logger=None):
        self.connection_string = connection_string
        self.logger = logger
        self._conn: Optional[pyodbc.Connection] = None

    # ── connection ────────────────────────────────────────────────────────────

    @property
    def conn(self) -> pyodbc.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = pyodbc.connect(self.connection_string)
        return self._conn

    # ── READ ──────────────────────────────────────────────────────────────────

    def get_running_model_configs(self, is_active: bool = True) -> List[RunningModelConfigDTO]:
        """Returns all model configs (active by default)."""
        rows = []
        with self.conn.cursor() as cur:
            cur.execute("{CALL get_running_model_configs (?)}", (1 if is_active else 0,))
            rows = cur.fetchall()
        return [self._row_to_dto(r) for r in rows]

    def get_running_model_by_id(self, model_id: int) -> Optional[RunningModelConfigDTO]:
        """Returns a single model config by PK, or None."""
        with self.conn.cursor() as cur:
            cur.execute("{CALL get_running_model_by_id (?)}", (model_id,))
            row = cur.fetchone()
        return self._row_to_dto(row) if row else None

    # ── WRITE ─────────────────────────────────────────────────────────────────

    def persist_running_model_config(
        self,
        model_name:             str,
        algo_type:              str,
        model_path:             str,
        symbol:                 str,
        bias:                   str,
        d_from:                 str,
        d_to:                   str,
        lower_percentile_limit: float,
        n_flip:                 int,
        make_stationary:        bool,
        draw_predictions:       bool,
        init_portf_size:        float,
        trade_comm:             float,
        series_csv:             str,
        display_order:          int = 0,
        is_active:              bool = True,
    ) -> int:
        """
        Upsert model config (by model_name) and replace its series.
        Returns the model_id.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "{CALL persist_running_model_config (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)}",
                (
                    model_name, algo_type, model_path, symbol, bias,
                    d_from, d_to,
                    lower_percentile_limit, n_flip,
                    1 if make_stationary else 0,
                    1 if draw_predictions else 0,
                    init_portf_size, trade_comm,
                    display_order,
                    1 if is_active else 0,
                ),
            )
            row = cur.fetchone()
            model_id = int(row[0]) if row else None
            self.conn.commit()

        if model_id and series_csv:
            self._replace_series(model_id, series_csv)

        return model_id

    def delete_running_model_config(self, model_id: int) -> None:
        """Hard-deletes model config + its series (CASCADE)."""
        with self.conn.cursor() as cur:
            cur.execute("{CALL delete_running_model_config (?)}", (model_id,))
            self.conn.commit()

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _replace_series(self, model_id: int, series_csv: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "{CALL persist_running_model_series (?,?)}",
                (model_id, series_csv),
            )
            self.conn.commit()

    @staticmethod
    def _row_to_dto(row) -> RunningModelConfigDTO:
        return RunningModelConfigDTO(
            model_id               = int(row[0]),
            model_name             = str(row[1]),
            algo_type              = str(row[2]),
            model_path             = str(row[3]),
            symbol                 = str(row[4]),
            bias                   = str(row[5]),
            d_from                 = str(row[6])[:10],   # keep only date part
            d_to                   = str(row[7])[:10],
            lower_percentile_limit = float(row[8]),
            n_flip                 = int(row[9]),
            make_stationary        = bool(row[10]),
            draw_predictions       = bool(row[11]),
            init_portf_size        = float(row[12]),
            trade_comm             = float(row[13]),
            display_order          = int(row[14]),
            is_active              = bool(row[15]),
            series_csv             = str(row[16]) if row[16] else "",
        )