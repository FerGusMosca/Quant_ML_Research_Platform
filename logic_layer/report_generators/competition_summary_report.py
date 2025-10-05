import os
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
import spacy

from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from framework.common.logger.message_type import MessageType


class CompetitionSummaryReport:

    def __init__(self, year, logger, report_type= ReportFolder.K10.value,portfolio=None):
        """
        :param year: filing year
        :param logger: logger instance
        :param report_type: "K10" or "Q10"
        """
        self.year = year
        self.logger = logger
        self.report_type = report_type.upper()

        self.input_dir=f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{report_type}/{year}"
        self.output_dir = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{report_type}_competition_summary_report/{year}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.nlp = spacy.load("en_core_web_sm")

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for file in os.listdir(self.input_dir):
            if file.endswith(".html"):
                symbol = file.split("_")[0]
                self._process_file(symbol, file)

        '''
        self.consolidate_competition_reports(
            input_dir=f"{self.output_dir}",
            output_file=f"{self.output_dir}/competition_summary_all_{self.year}.json"
        )
        '''

    def consolidate_competition_reports(self,input_dir, output_file):
        all_reports = []

        for file in os.listdir(input_dir):
            if file.endswith("_competition.json"):
                path = os.path.join(input_dir, file)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_reports.append(data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2)

        print(f"[COMP] ✅ Consolidated {len(all_reports)} reports into {output_file}")

    def _extract_period_from_filename(self,filename: str) -> str:
        """
        Extract fiscal period (Q1/Q2/Q3/Q4 or YEAR) from SEC filing filename.
        Example:
          'AAPL_2024_Q1_10-Q.html' -> 'Q1'
          'AAPL_2022_10-K.html'   -> '2022'
        """
        # Get only the file name (strip directories)
        name = Path(filename).stem  # e.g. AAPL_2024_Q1_10-Q

        # Look for quarter pattern
        match_quarter = re.search(r'_Q([1-4])_', name, re.IGNORECASE)
        if match_quarter:
            return f"Q{match_quarter.group(1)}"

        # Look for year pattern (4 digits)
        match_year = re.search(r'_(\d{4})_', name)
        if match_year:
            return "Y"+match_year.group(1)

        # Fallback if no pattern found
        return "UNKNOWN"

    def _process_file(self, symbol, file_name):
        path = os.path.join(self.input_dir, file_name)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        period=self._extract_period_from_filename(filename=file_name)

        # Simple heuristic: extract "Competition" section
        lower_text = text.lower()
        if "competition" not in lower_text:
            self.logger.do_log(f"[COMP] No competition section found for {symbol}", MessageType.WARNING)
            return

        start = lower_text.index("competition")
        # stop at next common section title
        end = lower_text.find("regulation", start)
        section = text[start:end] if end != -1 else text[start:]

        # Run NER
        doc = self.nlp(section)
        competitors = {}
        for ent in doc.ents:
            if ent.label_ == "ORG":
                competitors[ent.text] = competitors.get(ent.text, 0) + 1

        summary = {
            "symbol": symbol,
            "year": self.year,
            "period":period,
            "competition_summary": [
                {"competitor": k, "mentions": v, "context": section[:500]}
                for k, v in competitors.items()
            ]
        }

        out_path = os.path.join(self.output_dir, f"{symbol}_{self.year}_{period}_competition.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.do_log(f"[COMP] ✅ Competition summary created for {symbol}", MessageType.INFO)
