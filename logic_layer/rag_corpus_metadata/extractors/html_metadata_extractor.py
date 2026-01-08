import os
from bs4 import BeautifulSoup

class HTMLMetadataExtractor:
    def __init__(self, logger):
        self.logger = logger

    def extract(self, html_path):
        try:
            # --- File exists ---
            if not os.path.exists(html_path):
                self.logger.do_log(f"[META] ❌ File not found: {html_path}", 1)
                return {"path": html_path, "skipped": True}

            # --- File > 0 bytes ---
            if os.path.getsize(html_path) < 32:
                self.logger.do_log(f"[META] ❌ Empty or corrupt HTML: {html_path}", 1)
                return {"path": html_path, "skipped": True}

            # --- Read raw HTML ---
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_html = f.read()

            soup = BeautifulSoup(raw_html, "lxml")

            # --- Basic metadata ---
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            body = soup.body
            body_text = body.get_text(separator="\n", strip=True) if body else ""

            return {
                "path": html_path,
                "filename": os.path.basename(html_path),
                "folder": os.path.dirname(html_path),
                "html_length": len(raw_html),
                "text_length": len(body_text),
                "title_guess": title[:180],
                "full_text": raw_html,
                "skipped": False
            }

        except Exception as e:
            self.logger.do_log(f"[META] ❌ Fatal HTML metadata error: {html_path} | {e}", 1)
            return {"path": html_path, "skipped": True}
