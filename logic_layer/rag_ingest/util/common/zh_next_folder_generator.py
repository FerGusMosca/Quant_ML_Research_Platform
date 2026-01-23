# logic_layer/rag_ingest/util/zh_next_folder_generator.py
# All comments in English
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


class ZHNextFolderGenerator:
    """
    Generates the next chronological folder paths based on a given ZeroHedge folder.
    Does NOT check if folders exist on disk — only generates valid calendar-based paths.

    The base path is dynamically extracted from the input folder:
    Everything up to and including "/Archives/" is considered the base.

    Example:
        Input: "/dropbox_sync/Archives/2025/November/Nov 8"
        Base path extracted: "/dropbox_sync/Archives/"
        Next folders:
            "/dropbox_sync/Archives/2025/November/Nov 9"
            "/dropbox_sync/Archives/2025/November/Nov 10"
            etc.
    Handles month and year rollover automatically.
    """

    def __init__(self):
        self.month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        self.month_abbrs = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        self.abbr_to_num = {abbr.lower(): num for num, abbr in self.month_abbrs.items()}

    def _extract_base_path(self, folder_path: str,dest_root:str) -> Optional[str]:
        try:
            p = Path(folder_path).resolve()
            parts = p.parts

            if dest_root not in parts:
                return None

            idx = parts.index(dest_root)
            return str(Path(*parts[:idx + 1]))
        except Exception:
            return None

    def _parse_date_from_path(self, folder_path: str,dest_root:str) -> Optional[datetime]:
        """
        Parses year, month, and day from the full folder path.
        Returns datetime object or None if parsing fails.
        """
        try:
            normalized = folder_path.replace("\\", "/")
            parts = normalized.split("/")
            if dest_root not in parts:
                return None
            archives_idx = parts.index(dest_root)

            # Year is right after Archives
            year_str = parts[archives_idx + 1]
            year = int(year_str)

            # Day folder is the last part: "Nov 8"
            day_folder = parts[-1].strip()
            match = re.match(r"^([A-Za-z]+)\s+(\d+)$", day_folder, re.IGNORECASE)
            if not match:
                return None

            month_abbr, day_str = match.groups()
            month = self.abbr_to_num.get(month_abbr.lower()[:3])
            if not month:
                return None

            day = int(day_str)
            if day < 1 or day > 31:
                return None

            return datetime(year, month, day)

        except (ValueError, IndexError):
            return None

    def _build_path_from_date(self, base_path: str, dt: datetime) -> str:
        """
        Builds the full folder path using the extracted base path and a datetime.
        Cross-platform: Windows + Linux.
        """
        year = str(dt.year)
        month_name = self.month_names[dt.month]
        day_folder = f"{self.month_abbrs[dt.month]} {dt.day}"

        return os.path.join(base_path, year, month_name, day_folder)

    def generate_next_folders(self, current_folder: str,dest_root:str, n: int = 10) -> List[str]:
        """
        Generates the next n chronological folder paths after the given current_folder.
        Base path is dynamically extracted — no hardcoding.

        :param current_folder: Full path e.g. "/dropbox_sync/Archives/2025/November/Nov 8"
        :param n: Number of next suggestions
        :return: List of next folder paths
        """
        base_path = self._extract_base_path(current_folder,dest_root)
        if base_path is None:
            raise ValueError(f"Cannot extract base path (missing '{dest_root}') from: {current_folder}")

        base_date = self._parse_date_from_path(current_folder,dest_root)
        if base_date is None:
            raise ValueError(f"Cannot parse date from folder path: {current_folder}")

        suggestions = []
        next_date = base_date

        for _ in range(n):
            next_date += timedelta(days=1)
            path = self._build_path_from_date(base_path, next_date)
            suggestions.append(path)

        return suggestions

    def find_next_folder(self,last_sucessful_folder):

        folder_gen = ZHNextFolderGenerator()
        suggestions = folder_gen.generate_next_folders(last_sucessful_folder, n=4)

        found = False
        for candidate in suggestions:
            if os.path.exists(candidate) and os.path.isdir(candidate):
                pdfs = [os.path.join(r, f) for r, _, fs in os.walk(candidate) for f in fs if
                        f.lower().endswith('.pdf')]
                if pdfs:
                    source_path = candidate
                    pdf_list = pdfs
                    self.logger.do_log(f"[RAG] Next Folder Found: {candidate} ({len(pdfs)} PDFs)", 1)
                    found = True
                    break

        if not found:
            self.logger.do_log(f"[RAG] No folders with PDFs → next to {last_sucessful_folder} --> Nothing to process.",
                               1)
            return