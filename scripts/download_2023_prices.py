import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

tickers = ["BFAM","ZBH","VRTX","D","YOU","DTG","UNH","SBRA","IMO","SYK"]

# fechas objetivo
d1 = pd.Timestamp("2023-12-15")
d2 = pd.Timestamp("2024-12-31")

def prev_trading_close(series, target_date):
    # si no hay dato exactamente en target_date, toma el último anterior
    dates = series.index
    if target_date in dates:
        return series.loc[target_date]
    before = dates[dates <= target_date]
    if len(before) == 0:
        return float('nan')
    return series.loc[before.max()]

rows = []
for t in tickers:
    # bajamos un rango amplio para cubrir feriados y gaps
    start = "2023-12-01"
    end   = "2025-01-15"
    try:
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            rows.append({"symbol": t, "price_2023_12_15": None, "price_2024_12_31": None, "return_%": None, "note": "sin datos"})
            continue
        # usamos Adj Close (incluye efectos de splits/dividendos)
        s = df["Adj Close"]
        p1 = prev_trading_close(s, d1)
        p2 = prev_trading_close(s, d2)
        ret = None
        note = ""
        if pd.notna(p1) and pd.notna(p2) and p1 != 0:
            ret = (p2/p1 - 1) * 100
        if pd.isna(p1): note += "no p1; "
        if pd.isna(p2): note += "no p2; "
        rows.append({"symbol": t, "price_2023_12_15": None if pd.isna(p1) else round(float(p1), 4),
                     "price_2024_12_31": None if pd.isna(p2) else round(float(p2), 4),
                     "return_%": None if ret is None else round(float(ret), 2),
                     "note": note.strip()})
    except Exception as e:
        rows.append({"symbol": t, "price_2023_12_15": None, "price_2024_12_31": None, "return_%": None, "note": f"error: {e}"})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("biotech_10_prices_2023_12_15_to_2024_12_31.csv", index=False)
print("\n>>> Archivo guardado: biotech_10_prices_2023_12_15_to_2024_12_31.csv")
