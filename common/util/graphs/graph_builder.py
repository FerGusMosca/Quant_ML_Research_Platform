import os

import matplotlib
import plotly.graph_objs as go
from common.enums.side import Side
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
import numpy as np

class GraphBuilder:

    @staticmethod
    def _select_safe_backend(interactive: bool):
        """
        Use a non-blocking backend (Agg) for headless environments when interactive=False.
        If interactive=True, do not force Agg so a window can be shown with show().
        """
        if interactive:
            return
        try:
            # Headless on Linux/WSL: no DISPLAY -> force Agg
            if os.environ.get("DISPLAY") is None:
                matplotlib.use("Agg", force=True)
        except Exception:
            matplotlib.use("Agg", force=True)


    @staticmethod
    def booster_importances_aligned(booster, feature_names, importance_types=("gain", "total_gain", "weight")):
        """
        Return a list of importances aligned to 'feature_names' using the native XGBoost Booster.
        Tries multiple importance types in order until it finds a non-all-zero vector.
        Handles both real feature names and 'f0', 'f1', ... keys.

        Parameters
        ----------
        booster : xgboost.Booster
        feature_names : list[str]
        importance_types : tuple[str]
            e.g., ("gain", "total_gain", "weight")

        Returns
        -------
        list[float]
        """
        def as_aligned(importance_dict):
            # Map both real names and f{idx} format to the provided feature_names
            values = []
            for i, name in enumerate(feature_names):
                if name in importance_dict:
                    v = float(importance_dict.get(name, 0.0))
                else:
                    v = float(importance_dict.get(f"f{i}", 0.0))
                values.append(v)
            return values

        chosen = None
        chosen_type = None
        for itype in importance_types:
            raw = booster.get_score(importance_type=itype) or {}
            vals = as_aligned(raw)
            if any(v != 0.0 for v in vals):
                chosen = vals
                chosen_type = itype
                break

        # If still all zeros, keep the last attempt (likely empty) but return zeros
        if chosen is None:
            chosen = [0.0] * len(feature_names)
            chosen_type = importance_types[-1]

        # Optional: normalize for readability if values are huge (do not change ranking)
        s = sum(chosen)
        normalized = [v / s if s > 0 else 0.0 for v in chosen]

        print(f"[DEBUG] importance_type used: {chosen_type}; sum={s:.6f}; max={max(chosen) if chosen else 0:.6f}")
        return normalized  # normalized preserves order and avoids “flat near-zero” visuals


    @staticmethod
    def build_candles_graph(min_candles_df, mov_avg_unit=20):
        # Ensure that the 'date' column is of type datetime
        min_candles_df['date'] = pd.to_datetime(min_candles_df['date'])

        # Calculate the moving average based on the 'close' price
        min_candles_df['moving_avg'] = min_candles_df['close'].rolling(window=mov_avg_unit).mean()

        # Create the candlestick chart using Plotly's go.Figure
        fig = go.Figure(data=[go.Candlestick(x=min_candles_df['date'],
                                             open=min_candles_df['open'],
                                             high=min_candles_df['high'],
                                             low=min_candles_df['low'],
                                             close=min_candles_df['close'],
                                             name='Candlesticks')])

        # Add the moving average line (in red) to the chart
        fig.add_trace(go.Scatter(x=min_candles_df['date'],
                                 y=min_candles_df['moving_avg'],
                                 mode='lines',
                                 line=dict(color='red', width=2),
                                 name=f'Moving Avg ({mov_avg_unit} min)'))

        # Update chart layout for better readability
        fig.update_layout(
            title='Candlestick Chart with Moving Average',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False  # Hide the range slider for simplicity
        )

        # Show the final chart
        fig.show()

    @staticmethod
    def plot_prices_with_trades(symbol_prices_df, summary_dict_arr, strategy_name):
        # Ensure 'date' column is in datetime format
        symbol_prices_df['date'] = pd.to_datetime(symbol_prices_df['date'])

        # Create base price plot in blue
        plt.figure(figsize=(16, 6))
        plt.plot(symbol_prices_df['date'], symbol_prices_df['close'], label='Price', color='blue')

        # Loop through all trade summaries
        for summary in summary_dict_arr:
            for pos in summary[strategy_name].portf_pos_summary:
                start = pos.date_open
                end = pos.date_close
                color = 'green' if pos.side == Side.LONG.value else 'red'

                # Highlight the segment corresponding to each trade
                mask = (symbol_prices_df['date'] >= start) & (symbol_prices_df['date'] <= end)
                plt.plot(
                    symbol_prices_df.loc[mask, 'date'],
                    symbol_prices_df.loc[mask, 'close'],
                    linewidth=3,
                    color=color
                )

        plt.title("Prices with Trade Periods")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True)

    @staticmethod
    def plot_long_probability_distributions(prob,threshold):
        # Log distribution of predicted probabilities
        plt.hist(prob, bins=50)
        plt.title(f"Distribution of LONG probabilities (threshold={threshold})")
        plt.xlabel("Probability of LONG")
        plt.ylabel("Frequency")
        plt.savefig("long_prob_distribution.png")

    @staticmethod
    def plot_feature_importances_from_values(importances, feature_names, output_path="feature_importance.png"):
        """
        Plot feature importances given as a list/array of numeric values.
        - Validates shape
        - Sorts descending
        - Prints ranking
        - Saves PNG
        - Shows the plot and blocks execution until the window is closed
        """

        importances = np.asarray(importances, dtype=float)
        if len(feature_names) != importances.shape[0]:
            raise ValueError(
                f"[ERROR] Feature mismatch: got {importances.shape[0]} importances "
                f"but {len(feature_names)} feature names."
            )

        idx = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in idx]

        print("\n[FEATURE IMPORTANCE RANKING]")
        for i in idx:
            print(f"{feature_names[i]:<35} → {importances[i]:.6f}")

        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[idx], align="center")
        plt.xticks(range(len(importances)), sorted_features, rotation=45, ha="right")
        plt.tight_layout()

        # Save PNG
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

        # Show plot and wait until closed
        #plt.show()

    @staticmethod
    def plot_feature_importances(model, feature_names, output_path="feature_importance.png"):


        # Validate consistency between feature names and model internals
        if len(feature_names) != len(model.feature_importances_):
            raise ValueError(
                f"[ERROR] Feature mismatch: model has {len(model.feature_importances_)} features "
                f"but received {len(feature_names)} feature names."
            )

        # Compute importance ranking
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]

        # Log feature importance values
        print("\n[FEATURE IMPORTANCE RANKING]")
        for i in indices:
            print(f"{feature_names[i]:<35} → {importances[i]:.4f}")

        # Plot bar chart
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), sorted_features, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path)  # Use plt.show() if running interactively

    @staticmethod
    def plot_2_series_overlapped(pivot_df, benchmark_df, benchmark_symbol, output_symbol):
        import matplotlib.dates as mdates

        # Ensure datetime index
        pivot_df.index = pd.to_datetime(pivot_df["date"])

        # Create base figure
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Lightweight Indicator (left axis)
        ax1.plot(
            pivot_df.index,
            pivot_df[output_symbol],
            label=output_symbol,
            color="blue"
        )
        ax1.set_ylabel("Lightweight Indicator", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        # Plot Benchmark (right axis)
        if benchmark_df is not None:
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
            benchmark_pivot = benchmark_df.pivot(index="date", columns="symbol", values="close")

            # Align to common date range
            common_start = pivot_df.index.min()
            common_end = pivot_df.index.max()
            benchmark_pivot = benchmark_pivot[
                (benchmark_pivot.index >= common_start) & (benchmark_pivot.index <= common_end)
                ]

            ax2 = ax1.twinx()
            ax2.plot(
                benchmark_pivot.index,
                benchmark_pivot[benchmark_symbol],
                label=benchmark_symbol,
                color="orange",
                linestyle="--"
            )
            ax2.set_ylabel(benchmark_symbol, color="orange")
            ax2.tick_params(axis='y', labelcolor='orange')

        # Combined legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        if benchmark_df is not None:
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
        else:
            ax1.legend(loc="upper left")

        # Improve X-axis date formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        fig.autofmt_xdate()

        # Final tweaks
        ax1.set_title("Lightweight Indicator vs Benchmark (dual axis)")
        ax1.grid(True)

        # Enable interactive hover (optional)
        try:
            mplcursors.cursor(ax1.lines + (ax2.lines if benchmark_df is not None else []), hover=True)
        except Exception:
            pass

        plt.tight_layout()
        plt.show()


