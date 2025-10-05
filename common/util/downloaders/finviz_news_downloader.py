import os
import time
import random
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

from common.enums.folders import Folders


class FinVizNewsDownloader:
    """
    Downloader for FinViz headlines (per ticker).
    Saves daily headlines in JSON under yearly folder.
    """

    @staticmethod
    def download(symbol,portfolio, pause=1.0):

        base_output = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/Finviz/news/"
        today = datetime.today().strftime("%Y-%m-%d")
        year = datetime.today().year

        # yearly folder
        output_dir = os.path.join(base_output, str(year))
        os.makedirs(output_dir, exist_ok=True)

        url = f"https://finviz.com/quote.ashx?t={symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; NewsScraper/1.0; contact@example.com)"
        }

        print(f"[FinVizNewsDownloader][DEBUG] Fetching news for {symbol} -> {url}")
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"[FinVizNewsDownloader] Failed request for {symbol}, status={resp.status_code}")
        time.sleep(pause + random.random())

        soup = BeautifulSoup(resp.text, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")
        if news_table is None:
            raise RuntimeError(f"[FinVizNewsDownloader] No news table found for {symbol}")

        headlines = []
        for row in news_table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            # FinViz puts either full datetime or only hour if same day
            raw_ts = cols[0].text.strip()
            title = cols[1].text.strip()
            link = cols[1].a["href"] if cols[1].a else None

            # Normalize → if it only shows time, add today's date
            if ":" in raw_ts and len(raw_ts) <= 8:
                ts_date = today
                ts_time = raw_ts
            else:
                # Example: "Oct-02-25 09:30AM"
                try:
                    parsed = datetime.strptime(raw_ts, "%b-%d-%y %I:%M%p")
                    ts_date = parsed.strftime("%Y-%m-%d")
                    ts_time = parsed.strftime("%H:%M")
                except:
                    ts_date, ts_time = today, raw_ts

            headlines.append({
                "date": ts_date,
                "time": ts_time,
                "title": title,
                "link": link
            })

        if not headlines:
            raise RuntimeError(f"[FinVizNewsDownloader] No headlines scraped for {symbol}")

        out_path = os.path.join(output_dir, f"{symbol}_{today}_news.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "symbol": symbol,
                "date": today,
                "headlines": headlines
            }, f, indent=2)

        print(f"[FinVizNewsDownloader][INFO] ✅ Saved {len(headlines)} headlines for {symbol} -> {out_path}")
        return out_path
