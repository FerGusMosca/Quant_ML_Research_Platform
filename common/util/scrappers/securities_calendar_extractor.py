import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
from bs4 import BeautifulSoup
from collections import Counter


class SecuritiesCalendarExtractor:

    @staticmethod
    def _parse_date(text: str) -> Optional[datetime]:
        patterns = [
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+([0-9]{1,2}),?\s+([0-9]{4})',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+([0-9]{1,2}),?\s+([0-9]{4})',
            r'([0-9]{4}-\d{2}-\d{2})',
        ]
        months = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
                  'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                g = m.groups()
                try:
                    if len(g)==3 and not g[0].isdigit():
                        mon = months.get(g[0].lower(), months.get(g[0].lower()[:3]))
                        return datetime(int(g[2]), mon, int(g[1]))
                    elif len(g)==1:
                        return datetime.strptime(g[0], '%Y-%m-%d')
                except: pass
        return None

    @staticmethod
    def extract_filing_date_from_file(file_path: Path) -> Optional[datetime]:
        if not file_path.exists(): return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            candidates = []

            # 1. Busca fechas cerca de firmas o en celdas de tabla
            for td in soup.find_all('td'):
                txt = td.get_text()
                date = SecuritiesCalendarExtractor._parse_date(txt)
                if date: candidates.append(date)

            # 2. "Date:" explícito
            date_line = re.search(r'Date[:\s]+(.+)', soup.get_text(), re.I)
            if date_line:
                d = SecuritiesCalendarExtractor._parse_date(date_line.group(1))
                if d: candidates.append(d)

            # 3. Fecha más frecuente = filing date
            if candidates:
                return Counter(candidates).most_common(1)[0][0]

            return None
        except Exception as e:
            print(f"Error {file_path}: {e}")
            return None

    @staticmethod
    def process_k10_q10_directories(k10_dir: Path, q10_dir: Path) -> Tuple[Optional[datetime], Dict[int, Optional[datetime]]]:
        k10_date = None
        q10_dates = {1: None, 2: None, 3: None}

        if k10_dir.exists():
            files = list(k10_dir.glob("*.html")) + list(k10_dir.glob("*.htm"))
            if files:
                k10_date = SecuritiesCalendarExtractor.extract_filing_date_from_file(files[0])

        if q10_dir.exists():
            for f in q10_dir.glob("*.html"):
                name = f.name.upper()
                q = None
                if '_Q1_' in name or 'Q1' in name: q = 1
                elif '_Q2_' in name or 'Q2' in name: q = 2
                elif '_Q3_' in name or 'Q3' in name: q = 3
                if q:
                    q10_dates[q] = SecuritiesCalendarExtractor.extract_filing_date_from_file(f)

        return k10_date, q10_dates