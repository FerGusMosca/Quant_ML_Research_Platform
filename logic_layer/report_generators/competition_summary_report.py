import os
import json
from bs4 import BeautifulSoup
import spacy

from framework.common.logger.message_type import MessageType


class CompetitionSummaryReport:
    input_dir = f"./output/K10/"
    output_dir = f"./output/K10/competition_summary_report"
    def __init__(self, year, logger):
        self.input_dir = f"{CompetitionSummaryReport.input_dir}{year}"
        self.output_dir = CompetitionSummaryReport.output_dir
        self.year = year
        self.logger = logger
        self.nlp = spacy.load("en_core_web_sm")

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for file in os.listdir(self.input_dir):
            if file.endswith(".html"):
                symbol = file.split("_")[0]
                self._process_file(symbol, file)

        self.consolidate_competition_reports(
            input_dir=f"{self.output_dir}",
            output_file=f"{self.output_dir}/competition_summary_all_{self.year}.json"
        )

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

    def _process_file(self, symbol, file_name):
        path = os.path.join(self.input_dir, file_name)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)

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
            "competition_summary": [
                {"competitor": k, "mentions": v, "context": section[:500]}
                for k, v in competitors.items()
            ]
        }

        out_path = os.path.join(self.output_dir, f"{symbol}_{self.year}_competition.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.do_log(f"[COMP] ✅ Competition summary created for {symbol}", MessageType.INFO)
