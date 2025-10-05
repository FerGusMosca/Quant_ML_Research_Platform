import os
import time
import random
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from common.enums.folders import Folders


class FinVizFullNewsDownloader:
    """
    Extended FinViz downloader with automatic Selenium fallback for 403-blocked articles.
    """

    # === Main method ===
    @staticmethod
    def download(symbol, portfolio, pause=1.0):
        today = datetime.today().strftime("%Y-%m-%d")
        year = datetime.today().year
        base_output = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/Finviz/full_news/{symbol}/"
        output_dir = os.path.join(base_output, str(year))
        os.makedirs(output_dir, exist_ok=True)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finviz.com/"
        }

        url = f"https://finviz.com/quote.ashx?t={symbol}"
        print(f"[FinVizFullNewsDownloader][DEBUG] Fetching news list for {symbol} -> {url}")
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"[FinVizFullNewsDownloader] Failed request for {symbol}, status={resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")
        if news_table is None:
            raise RuntimeError(f"[FinVizFullNewsDownloader] No news table found for {symbol}")

        articles = []
        for row in news_table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            raw_ts = cols[0].text.strip()
            title = cols[1].text.strip()
            link = cols[1].a["href"] if cols[1].a else None
            if not link:
                continue

            # normalize timestamp
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

            # normalize link
            full_link = FinVizFullNewsDownloader._normalize_link(link)
            content = FinVizFullNewsDownloader._fetch_article(full_link, headers)

            # save text if available
            if content:
                safe_name = f"{symbol}_{ts_date}_{ts_time.replace(':', '-')}.txt"
                txt_path = os.path.join(output_dir, safe_name)
                if not os.path.exists(txt_path):
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(content)
                else:
                    print(f"[SKIP] Already exists: {txt_path}")

            articles.append({
                "date": ts_date,
                "time": ts_time,
                "title": title,
                "link": full_link,
                "content": content
            })
            time.sleep(0.5 + random.random() * 0.8)

        # === Save JSON ===
        now_ts = datetime.now().strftime("%H-%M-%S")
        out_path = os.path.join(output_dir, f"{symbol}_{today}_{now_ts}_full_news.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"symbol": symbol, "date": today, "articles": articles}, f, indent=2, ensure_ascii=False)

        print(f"[FinVizFullNewsDownloader][INFO] ✅ Saved {len(articles)} items -> {out_path}")
        return out_path

    # === Helper: normalize link ===
    @staticmethod
    def _normalize_link(link: str) -> str:
        if link.startswith("/"):
            return f"https://finviz.com{link}"
        elif link.startswith("http"):
            return link
        return f"https://finviz.com/{link}"

    # === Helper: fetch article content ===
    @staticmethod
    def _fetch_article(url, headers):
        """Try direct request first, then fallback to Selenium if blocked."""
        try:
            art_resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
            if art_resp.status_code == 200 and art_resp.text:
                return FinVizFullNewsDownloader._extract_article_content(art_resp.text)

            elif art_resp.status_code in (401, 403):
                print(f"[WARN] Blocked ({art_resp.status_code}) for {url} — using Selenium fallback...")
                return FinVizFullNewsDownloader._fetch_via_browser(url)

            else:
                print(f"[WARN] {url} -> HTTP {art_resp.status_code}")
        except Exception as e:
            print(f"[WARN] Exception fetching {url} -> {e}")
        return None

    # === Helper: extract article content ===
    @staticmethod
    def _extract_article_content(html):
        soup = BeautifulSoup(html, "html.parser")

        # priority: og:description → articleBody → <article> → divs → all <p>
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            return og["content"].strip()

        ab = soup.find(attrs={"itemprop": "articleBody"})
        if ab:
            return " ".join(p.get_text(strip=True) for p in ab.find_all("p"))

        art = soup.find("article")
        if art:
            return " ".join(p.get_text(strip=True) for p in art.find_all("p"))

        for sel in ["div.article-body", "div.article-content", "div.story-body", "div.post-content"]:
            block = soup.select_one(sel)
            if block:
                text = " ".join(p.get_text(strip=True) for p in block.find_all("p"))
                if text:
                    return text

        paras = soup.find_all("p")
        long_paras = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 80]
        if long_paras:
            return " ".join(long_paras)
        return None

    # === Helper: Selenium fallback ===
    @staticmethod
    def _fetch_via_browser(url):
        """Open blocked pages using Chrome and extract text (visible logs for debugging)."""
        from selenium.common.exceptions import WebDriverException

        print(f"[FALLBACK] Launching Chrome to bypass block -> {url}")

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1200,800")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-infobars")

        driver = None
        html = None

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            print(f"[FALLBACK][INFO] ✅ Chrome launched successfully.")

            driver.set_page_load_timeout(15)
            driver.get(url)
            print(f"[FALLBACK][INFO] Page loaded, current title: {driver.title}")

            time.sleep(3)
            html = driver.page_source

        except WebDriverException as we:
            print(f"[FALLBACK][ERROR] ❌ WebDriver failed to start -> {we}")
        except Exception as e:
            print(f"[FALLBACK][ERROR] ❌ Selenium error while fetching {url} -> {e}")
        finally:
            try:
                if driver:
                    driver.quit()
                    print("[FALLBACK][INFO] ✅ Browser closed cleanly.")
            except Exception as e:
                print(f"[FALLBACK][WARN] Failed to close driver -> {e}")

        if html:
            return FinVizFullNewsDownloader._extract_article_content(html)
        else:
            print(f"[FALLBACK][FAIL] ❌ No HTML retrieved for {url}")
            return None

