import requests
import json
import re
from bs4 import BeautifulSoup

URL = "https://seekingalpha.com/symbol/INTC/income-statement"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/json",
    "Referer": "https://seekingalpha.com/symbol/INTC",
}

r = requests.get(URL, headers=headers, timeout=15)
print("HTTP:", r.status_code)

html = r.text
soup = BeautifulSoup(html, "html.parser")

# --- Strategy A: look for JSON inside <script> tags ---
scripts = soup.find_all("script")
found = False

for i, s in enumerate(scripts):
    txt = s.string or ""
    if not txt:
        continue

    # heuristic: large JSON blobs often contain "income" / "financial" / "statement"
    if any(k in txt.lower() for k in ["income", "statement", "financial"]):
        # try to extract JSON objects
        candidates = re.findall(r'\{.*\}', txt, re.DOTALL)
        for c in candidates:
            try:
                data = json.loads(c)
                print(f"\n--- JSON FOUND in <script> #{i} ---")
                print("Top-level keys:", list(data.keys())[:20])
                found = True
                break
            except Exception:
                pass
    if found:
        break

if not found:
    print("\nNo obvious JSON found in script tags.")

# --- Strategy B: dump a small HTML sample for manual inspection ---
print("\n--- HTML SAMPLE (first 800 chars) ---")
print(html[:800])
