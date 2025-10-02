import os
import time
import random
import yfinance as yf
import json

_BASE_OUTPUT = "./output/Yahoo/yearly_income_statement"

class YahooYearlyIncomeStatement:
    """
    Downloader for annual income statements from Yahoo Finance.
    Iterates all years available (Yahoo usually returns last 4).
    """

    @staticmethod
    def download(symbol, base_output=_BASE_OUTPUT, pause=1.0):
        print(f"\n[YahooYearlyIncomeStatement][DEBUG] === Processing {symbol} ===")
        ticker = yf.Ticker(symbol)
        time.sleep(pause + random.random())  # pacing

        # Try to get financials
        try:
            income = ticker.financials.T
        except Exception as e:
            raise RuntimeError(f"[YahooYearlyIncomeStatement] Failed fetching financials for {symbol}: {e}")

        if income is None or income.empty:
            raise FileNotFoundError(f"[YahooYearlyIncomeStatement] No annual income statements returned for {symbol}")

        # Log periods
        print(f"[YahooYearlyIncomeStatement][DEBUG] Available periods for {symbol}: {[str(d.date()) for d in income.index]}")

        downloaded = []
        for date in income.index:
            year = date.year-1
            output_dir = os.path.join(base_output, str(year))
            os.makedirs(output_dir, exist_ok=True)

            data = income.loc[date].dropna().to_dict()
            out_path = os.path.join(output_dir, f"{symbol}_{year}_IncomeStatement.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "symbol": symbol,
                    "year": year,
                    "report_type": "YearlyIncomeStatement",
                    "items": data
                }, f, indent=2, default=str)

            print(f"[YahooYearlyIncomeStatement][INFO] ✅ Saved {symbol} ({year}) -> {out_path}")
            downloaded.append(out_path)

        return downloaded
