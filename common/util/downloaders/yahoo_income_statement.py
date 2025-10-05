import os
import time
import random
import yfinance as yf
import json

from common.enums.folders import Folders


class YahooIncomeStatement:
    """
    Downloader for yearly and quarterly income statements from Yahoo Finance.
    mode = "yearly" -> ticker.financials
    mode = "quarterly" -> ticker.quarterly_financials
    """

    _BASE_OUTPUT = "./output/Yahoo"

    @staticmethod
    def download(symbol,portfolio, mode="yearly", pause=1.0):
        print(f"\n[YahooIncomeStatement][DEBUG] === Processing {symbol} ({mode}) ===")
        ticker = yf.Ticker(symbol)
        time.sleep(pause + random.random())  # pacing

        # Select dataset based on mode
        if mode == "yearly":
            data_src = ticker.financials.T
            folder = "yearly_income_statement"
        elif mode == "quarterly":
            data_src = ticker.quarterly_financials.T
            folder = "quarterly_income_statement"
        else:
            raise ValueError("mode must be 'yearly' or 'quarterly'")

        if data_src is None or data_src.empty or len(data_src.index) == 0:
            print(f"[YahooIncomeStatement][DEBUG] Index is empty for {symbol} ({mode})")
            print(f"[YahooIncomeStatement][DEBUG] DataFrame shape: {data_src.shape}")
            return []

        # Log available periods
        print(
            f"[YahooIncomeStatement][DEBUG] Available periods for {symbol}: {[str(d.date()) for d in data_src.index]}")

        downloaded = []

        for date in data_src.index:
            # Handle yearly vs quarterly labels
            if mode == "yearly":
                # Fiscal year closes in the following year → subtract 1
                year = date.year - 1
                period_label = str(year)
            else:  # quarterly
                year = date.year
                quarter = ((date.month - 1) // 3) + 1
                period_label = f"{year}_Q{quarter}"

            # Ensure folder structure ./output/Yahoo/<mode>/<year>
            output_dir = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/Yahoo/year_income_statement/{year}"
            os.makedirs(output_dir, exist_ok=True)

            # Build file path
            out_path = os.path.join(output_dir, f"{symbol}_{period_label}_IncomeStatement.json")

            # Save JSON
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "symbol": symbol,
                    "period": period_label,
                    "report_type": f"{mode.capitalize()}IncomeStatement",
                    "items": data_src.loc[date].dropna().to_dict()
                }, f, indent=2, default=str)

            print(f"[YahooIncomeStatement][INFO] ✅ Saved {symbol} ({period_label}) -> {out_path}")
            downloaded.append(out_path)

        return downloaded
