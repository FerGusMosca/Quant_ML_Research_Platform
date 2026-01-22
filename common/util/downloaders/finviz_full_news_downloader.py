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
    All logs structured for real-time streaming in the UI.
    """

    @staticmethod
    def download(symbol, portfolio, pause=1.0,logger=None,job_id=None):



        today = datetime.today().strftime("%Y-%m-%d")
        year = datetime.today().year
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Output folder
        base_output = os.path.normpath(
            os.path.join(
                Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                portfolio,
                "Finviz",
                "full_news",
                f"{symbol}_{timestamp}"
            )
        )
        output_dir = os.path.join(base_output, str(year))
        os.makedirs(output_dir, exist_ok=True)

        # Headers
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

        logger.do_log_light(json.dumps({
            "event": "start",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False),job_id)

        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"[FinViz] Request failed for {symbol} status={resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")
        if news_table is None:
            raise RuntimeError(f"[FinViz] No news table for {symbol}")

        articles = []
        rows = news_table.find_all("tr")
        total = len(rows)

        # MAIN LOOP
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            raw_ts = cols[0].text.strip()
            title = cols[1].text.strip()
            link = cols[1].a["href"] if cols[1].a else None
            if not link:
                continue

            full_link = FinVizFullNewsDownloader._normalize_link(link)

            # Structured progress event
            logger.do_log_light(json.dumps({
                "event": "progress",
                "symbol": symbol,
                "title": title,
                "link": full_link,
                "current": len(articles) + 1,
                "total": total,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False),job_id)

            # Timestamp normalization
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

            # Fetch article
            content = FinVizFullNewsDownloader._fetch_article(full_link, headers,logger,job_id)

            # Save .txt
            if content:
                safe_name = f"{symbol}_{ts_date}_{ts_time.replace(':', '-')}.txt"
                txt_path = os.path.join(output_dir, safe_name)
                if not os.path.exists(txt_path):
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(content)

            articles.append({
                "date": ts_date,
                "time": ts_time,
                "title": title,
                "link": full_link,
                "content": content
            })

            time.sleep(0.5 + random.random() * 0.8)

        # Save JSON
        now_ts = datetime.now().strftime("%H-%M-%S")
        out_path = os.path.normpath(
            os.path.join(output_dir, f"{symbol}_{today}_{now_ts}_full_news.json")
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "symbol": symbol,
                "date": today,
                "articles": articles
            }, f, indent=2, ensure_ascii=False)

        logger.do_log_light(json.dumps({
            "event": "saved",
            "symbol": symbol,
            "items": len(articles),
            "path": out_path
        }, ensure_ascii=False),job_id)

        return out_path

    # Helpers
    @staticmethod
    def _normalize_link(link: str) -> str:
        if link.startswith("/"):
            return f"https://finviz.com{link}"
        elif link.startswith("http"):
            return link
        return f"https://finviz.com/{link}"

    @staticmethod
    def _fetch_article(url, headers,logger,job_id=None):
        try:
            r = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
            if r.status_code == 200 and r.text:
                return FinVizFullNewsDownloader._extract_article_content(r.text)
            elif r.status_code in (401, 403):
                logger.do_log_light(f"[WARN] Blocked {r.status_code} -> Selenium fallback",job_id)
                return FinVizFullNewsDownloader._fetch_via_browser(url,logger,job_id)
        except Exception as e:
            logger.do_log_light(f"[WARN] Error fetching {url} -> {e}")
        return None

    @staticmethod
    def _extract_article_content(html):
        soup = BeautifulSoup(html, "html.parser")

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

    @staticmethod
    def _fetch_via_browser(url,logger,job_id=None):
        from selenium.common.exceptions import WebDriverException

        logger.do_log_light(f"[FALLBACK] Chrome fallback -> {url}",job_id)

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1200,800")

        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(15)
            driver.get(url)
            time.sleep(3)
            html = driver.page_source
            return FinVizFullNewsDownloader._extract_article_content(html)

        except Exception as e:
            logger.do_log_light(f"[FALLBACK][ERROR] {e}",job_id)
            return None

        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
