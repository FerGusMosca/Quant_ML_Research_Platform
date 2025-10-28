import os
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup
import spacy
from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from framework.common.logger.message_type import MessageType


class CompetitionSummaryReport:
    def __init__(self, year, logger, report_type=ReportFolder.K10.value,
                 portfolio=None, dest_folder=None, rank_folder=None):
        self.year = year
        self.logger = logger
        self.report_type = report_type.upper()
        self.portfolio = portfolio
        self.dest_folder = dest_folder
        self.rank_folder = rank_folder

        self.input_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            portfolio, report_type, str(year)
        )
        self.output_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            dest_folder,
            f"{report_type}_competition_summary_report",
            str(year)
        )
        # Normalize paths for cross-platform compatibility
        self.input_dir = self.input_dir.replace("\\", "/")
        self.output_dir = self.output_dir.replace("\\", "/")

        os.makedirs(self.output_dir, exist_ok=True)
        self.nlp = spacy.load("en_core_web_sm")

    # ---------------------------------------------------------
    # Main runner
    # ---------------------------------------------------------
    def run(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".html")]
        self.logger.do_log(f"[COMP] Found {len(files)} {self.report_type} files for {self.year}", MessageType.INFO)
        for i, file in enumerate(sorted(files), start=1):
            try:
                symbol = file.split("_")[0]
                self._process_file(symbol, file)
                self.logger.do_log(f"[COMP][{i}/{len(files)}] ✅ {symbol} processed", MessageType.INFO)
            except Exception as e:
                self.logger.do_log(f"[COMP][{i}/{len(files)}] ❌ Failed for {file}: {e}", MessageType.ERROR)

    # ---------------------------------------------------------
    # File processor
    # ---------------------------------------------------------
    def _process_file(self, symbol, file_name):
        path = os.path.join(self.input_dir, file_name).replace("\\", "/")
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        period = self._extract_period_from_filename(file_name)

        # --- Extract competition section
        low = text.lower()
        if "competition" not in low:
            self.logger.do_log(f"[COMP] ⚠ No competition section for {symbol}", MessageType.WARNING)
            return

        start = low.index("competition")
        stop_candidates = ["regulation", "intellectual property", "employees", "risk factor"]
        stop_positions = [low.find(t, start + 50) for t in stop_candidates if low.find(t, start + 50) != -1]
        end = min(stop_positions) if stop_positions else start + 6000
        section = text[start:end]

        # --- Named Entity Recognition
        doc = self.nlp(section)
        competitors = {}
        for ent in doc.ents:
            if ent.label_ == "ORG":
                competitors[ent.text] = competitors.get(ent.text, 0) + 1

        curated_text = (
            f"Symbol: {symbol}. Year: {self.year}. Period: {period}. ReportType: {self.report_type}. "
            "Section: Competition and Market Position. "
            "This section discusses competitors, market dynamics, and market position. "
            "Competitors mentioned: " + ", ".join(competitors.keys() or ["None"]) + ". "
            "Context excerpt: " + section[:800]
        )

        summary = {
            "symbol": symbol,
            "year": self.year,
            "period": period,
            "curated_text": curated_text,
            "competition_summary": [
                {"competitor": k, "mentions": v, "context": section[:400]} for k, v in competitors.items()
            ],
        }

        out_path = os.path.join(self.output_dir, f"{symbol}_{self.year}_{period}_competition.json").replace("\\", "/")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # ---------------------------------------------------------
    # Period extractor
    # ---------------------------------------------------------
    def _extract_period_from_filename(self, filename: str) -> str:
        name = Path(filename).stem
        m_q = re.search(r'_Q([1-4])_', name, re.IGNORECASE)
        if m_q:
            return f"Q{m_q.group(1)}"
        m_y = re.search(r'_(\d{4})_', name)
        if m_y:
            return "Y" + m_y.group(1)
        return "UNKNOWN"

    # ---------------------------------------------------------
    # Consolidation and ranking
    # ---------------------------------------------------------
    @staticmethod
    def consolidate_year(year, report_type, portfolio, logger, dest_folder, rank_folder):
        base_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            dest_folder,
            f"{report_type}_competition_summary_report",
            str(year)
        ).replace("\\", "/")
        if not os.path.isdir(base_dir):
            logger.do_log(f"[COMP] ⚠ Missing folder: {base_dir}", MessageType.WARNING)
            return ""

        all_reports = []
        for fn in os.listdir(base_dir):
            if fn.endswith("_competition.json"):
                with open(os.path.join(base_dir, fn), "r", encoding="utf-8") as f:
                    all_reports.append(json.load(f))

        rank_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            rank_folder,
            f"{report_type}_competition_summary_report_rank",
            str(year)
        ).replace("\\", "/")
        os.makedirs(rank_dir, exist_ok=True)

        out_path = os.path.join(rank_dir, f"competition_summary_all_{year}.json").replace("\\", "/")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2)

        logger.do_log(f"[COMP] ✅ Consolidated -> {out_path} ({len(all_reports)} filers)", MessageType.INFO)
        return out_path

    @staticmethod
    def rank(consolidated_json, out_csv, logger):
        import pandas as pd
        if not os.path.exists(consolidated_json):
            logger.do_log(f"[COMP] ❌ Consolidated not found: {consolidated_json}", MessageType.ERROR)
            return

        with open(consolidated_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            logger.do_log(f"[COMP] ⚠ No data in {consolidated_json}", MessageType.WARNING)
            return

        rows = []
        for d in data:
            n_comp = len(d.get("competition_summary", []))
            rows.append({
                "symbol": d.get("symbol"),
                "year": d.get("year"),
                "competitors_count": n_comp,
                "top_competitors": ", ".join([c.get("competitor") for c in d.get("competition_summary", [])[:3]]),
                "excerpt_len": len(d.get("curated_text", "")),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            logger.do_log(f"[COMP] ⚠ Nothing to rank", MessageType.WARNING)
            return

        df["competition_intensity_score"] = df["competitors_count"] + 0.001 * df["excerpt_len"]
        df = df.sort_values("competition_intensity_score", ascending=False)

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.do_log(f"[COMP] ✅ Ranking -> {out_csv} ({len(df)} filers)", MessageType.INFO)
