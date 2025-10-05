import os
import time
import random
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

from common.enums.folders import Folders


class FinVizFullNewsDownloader:
    """
    Extended FinViz downloader.
    Step 1: fetches headlines (same as FinVizNewsDownloader).
    Step 2: fetches full article text for each link (if available).
    """

    @staticmethod
    def download(symbol, portfolio, pause=1.0):

        base_output = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/Finviz/full_news/{symbol}/"
        today = datetime.today().strftime("%Y-%m-%d")
        output_dir = os.path.join(base_output, today)
        os.makedirs(output_dir, exist_ok=True)

        url = f"https://finviz.com/quote.ashx?t={symbol}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsScraper/1.0; contact@example.com)"}

        print(f"[FinVizFullNewsDownloader][DEBUG] Fetching news headers for {symbol} -> {url}")
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"[FinVizFullNewsDownloader] Failed request for {symbol}, status={resp.status_code}")
        time.sleep(pause + random.random())

        soup = BeautifulSoup(resp.text, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")
        if news_table is None:
            raise RuntimeError(f"[FinVizFullNewsDownloader] No news table found for {symbol}")

        headlines = []
        for row in news_table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            raw_ts = cols[0].text.strip()
            title = cols[1].text.strip()
            link = cols[1].a["href"] if cols[1].a else None

            # Normalize timestamps
            if ":" in raw_ts and len(raw_ts) <= 8:
                ts_date = today
                ts_time = raw_ts
            else:
                try:
                    parsed = datetime.strptime(raw_ts, "%b-%d-%y %I:%M%p")
                    ts_date = parsed.strftime("%Y-%m-%d")
                    ts_time = parsed.strftime("%H:%M")
                except:
                    ts_date, ts_time = today, raw_ts

            # --- Step 2: try to fetch the full content ---
            content = None
            if link and link.startswith("http"):
                try:
                    art_resp = requests.get(link, headers=headers, timeout=10)
                    if art_resp.status_code == 200:
                        page_soup = BeautifulSoup(art_resp.text, "html.parser")
                        paras = page_soup.find_all("p")
                        full_text = " ".join([p.get_text(strip=True) for p in paras])
                        content = full_text[:20000] if full_text else None  # limit text size
                    else:
                        print(f"[FinVizFullNewsDownloader][WARN] Failed to fetch content ({art_resp.status_code}) for {link}")
                except Exception as e:
                    print(f"[FinVizFullNewsDownloader][WARN] Exception fetching {link} -> {e}")

            headlines.append({
                "date": ts_date,
                "time": ts_time,
                "title": title,
                "link": link,
                "content": content
            })

        # --- Save results ---
        out_path = os.path.join(output_dir, f"{symbol}_{today}_full_news.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "symbol": symbol,
                "date": today,
                "articles": headlines
            }, f, indent=2, ensure_ascii=False)

        print(f"[FinVizFullNewsDownloader][INFO] ✅ Saved {len(headlines)} articles (with content if available) -> {out_path}")
        return out_path
