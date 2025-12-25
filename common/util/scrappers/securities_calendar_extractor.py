import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
from bs4 import BeautifulSoup


class SecuritiesCalendarExtractor:
    """
    Utility class to extract real filing dates from downloaded 10-K and 10-Q HTML files.
    Prioritizes director/CFO signatures and 'Date:' lines over XBRL metadata.
    """

    @staticmethod
    def _parse_date_from_text(text: str) -> Optional[datetime]:
        """
        Parse a date from a text string using common formats found in SEC filings.
        Supported formats:
        - February 14, 2025
        - Aug. 27, 2025
        - 2025-08-27 (ISO)
        """
        patterns = [
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),\s+(\d{4})',
            r'(\d{4}-\d{2}-\d{2})',  # ISO format fallback
        ]

        months_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 3:  # Month name/day/year format
                        month_name, day_str, year_str = match.groups()
                        month = months_map.get(month_name.lower(), months_map.get(month_name.lower()[:3]))
                        if month:
                            return datetime(int(year_str), month, int(day_str))
                    elif len(match.groups()) == 1:  # ISO format
                        return datetime.fromisoformat(match.group(1))
                except (ValueError, AttributeError):
                    continue
        return None

    @staticmethod
    def extract_filing_date_from_file(file_path: Path) -> Optional[datetime]:
        """
        Extract the actual filing date from a 10-K or 10-Q HTML file.
        Priority:
        1. Most common date in signature lines (directors/CFO)
        2. Explicit 'Date:' line in signature block
        3. Fallback to XBRL metadata (not used for date here)
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()

            # Step 1: Look for dates in lines containing month names (typical in signatures)
            date_mentions = []
            lines = text.splitlines()
            month_keywords = ['january', 'february', 'march', 'april', 'may', 'june',
                              'july', 'august', 'september', 'october', 'november', 'december',
                              'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

            for line in lines:
                if any(kw in line.lower() for kw in month_keywords):
                    parsed = SecuritiesCalendarExtractor._parse_date_from_text(line)
                    if parsed:
                        date_mentions.append(parsed)

            # Use the most frequent date (all directors usually sign on the same day)
            if date_mentions:
                most_common = max(set(date_mentions), key=date_mentions.count)
                return most_common

            # Step 2: Look for explicit "Date:" line (common in 10-Q)
            date_match = re.search(r'Date:\s*(.+)', text, re.IGNORECASE)
            if date_match:
                parsed = SecuritiesCalendarExtractor._parse_date_from_text(date_match.group(1))
                if parsed:
                    return parsed

            return None

        except Exception as e:
            print(f"[Extractor] Error processing {file_path}: {e}")
            return None

    @staticmethod
    def process_k10_q10_directories(
        k10_dir: Path,
        q10_dir: Path
    ) -> Tuple[Optional[datetime], Dict[int, Optional[datetime]]]:
        """
        Process K10 and Q10 directories and return:
        - Filing date of the 10-K (or None)
        - Dict with filing dates for Q1, Q2, Q3 (or None if missing)
        """
        k10_date = None
        q10_dates = {1: None, 2: None, 3: None}

        # Process 10-K directory (usually only one file)
        if k10_dir.exists() and k10_dir.is_dir():
            html_files = list(k10_dir.glob("*.html")) + list(k10_dir.glob("*.htm"))
            if html_files:
                k10_date = SecuritiesCalendarExtractor.extract_filing_date_from_file(html_files[0])

        # Process 10-Q directory (multiple files: Q1, Q2, Q3)
        if q10_dir.exists() and q10_dir.is_dir():
            for q_file in q10_dir.glob("*.html"):
                filename_upper = q_file.name.upper()
                quarter = None
                if '_Q1_' in filename_upper or 'Q1' in filename_upper.replace(' ', ''):
                    quarter = 1
                elif '_Q2_' in filename_upper or 'Q2' in filename_upper.replace(' ', ''):
                    quarter = 2
                elif '_Q3_' in filename_upper or 'Q3' in filename_upper.replace(' ', ''):
                    quarter = 3

                if quarter:
                    date = SecuritiesCalendarExtractor.extract_filing_date_from_file(q_file)
                    q10_dates[quarter] = date

        return k10_date, q10_dates