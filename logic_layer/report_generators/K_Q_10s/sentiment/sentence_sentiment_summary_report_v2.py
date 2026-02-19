import os
import re
import json
from pathlib import Path
from typing import List

from common.enums.sec_reports import SECReports
from common.util.extractors.K_Q_10.k_q_10_mdna_extractor import KQ10MDNAExtractor
from common.util.std_in_out.root_locator import RootLocator
from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from framework.common.logger.message_type import MessageType
from logic_layer.report_generators.K_Q_10s.sentiment.base_sentiment_summary_report import SentimentAnalysisBase


class SentimentSummaryReportV2(SentimentAnalysisBase):
    """
    V2 – Financial sentiment analysis for SEC filings (MD&A section).
    Batch processing for entire portfolios.
    Inherits all core logic from SentimentAnalysisBase.
    """

    def __init__(
            self,
            year: int,
            logger,
            report_type: str = ReportFolder.K10.value,
            portfolio: str = None,
            filers_whitelist: List[str] = None,
            universe_key: str = None,
            dest_folder: str = None,
            rank_folder: str = None,
    ):
        """Initialize batch processor with file system paths."""
        # Initialize base class (LM dict + VADER)
        super().__init__(logger)

        self.year = year
        self.report_type = report_type.upper()
        self.portfolio = portfolio
        self.dest_folder = dest_folder
        self.rank_folder = rank_folder
        self.whitelist = set(t.upper() for t in filers_whitelist) if filers_whitelist else None

        # Input directory (where raw HTML filings are)
        self.input_dir = (
                self.root_dir
                / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                / portfolio
                / report_type
                / str(year)
        )

        # Output directory (where sentiment JSONs go)
        year_dir = (
                self.root_dir
                / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                / dest_folder
                / f"{self.report_type}_sentiment_summary_report"
                / str(year)
        )
        self.output_dir = year_dir / universe_key if universe_key else year_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.do_log(
            f"[SENT-V2] Initialized for {year} {report_type} | Input: {self.input_dir}",
            MessageType.INFO,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Process all HTML filings with detailed per-symbol logging."""
        files = [f for f in self.input_dir.glob("*.html")]
        if self.whitelist:
            files = [f for f in files if f.name.split("_")[0].upper() in self.whitelist]

        total_files = len(files)
        self.logger.do_log(f"[SENT-V2] 🚀 Processing {total_files} filings for {self.year} ({self.report_type})",
                           MessageType.INFO)

        success_count = 0
        failed_count = 0
        skipped_count = 0
        failed_symbols = []  # Para listar al final los que fallaron

        for i, file_path in enumerate(sorted(files), 1):
            symbol = file_path.name.split("_")[0].upper()
            self.logger.do_log(f"[SENT-V2][{i}/{total_files}] 🔄 Processing {symbol}...", MessageType.INFO)

            try:
                text = self._html_to_text(file_path)
                self.logger.do_log(f"[SENT-V2][{symbol}] 📄 Text loaded: {len(text):,} chars", MessageType.DEBUG)

                mdna_extractor=KQ10MDNAExtractor(self.logger)

                if(self.report_type==SECReports.K10.value):
                    mdna=mdna_extractor._extract_10k(text, symbol)
                elif self.report_type==SECReports.Q10.value:
                    mdna=mdna_extractor._extract_10q(text, symbol)
                else:
                    raise Exception(f"Invalid report type extracting MDNA section:{self.report_type}")

                #mdna = self._extract_mdna(text, symbol)
                if not mdna or len(mdna.strip()) < 500:
                    self.logger.do_log(f"[SENT-V2][{symbol}] ❌ MD&A FAILED – {len(mdna)} chars extracted",
                                       MessageType.WARNING)
                    failed_count += 1
                    failed_symbols.append(symbol)
                    continue

                self.logger.do_log(f"[SENT-V2][{symbol}] ✅ MD&A OK – {len(mdna):,} chars", MessageType.INFO)

                result = self._score_mdna(mdna)
                period = self._extract_period_from_filename(file_path.name)

                metrics = result["metrics"]
                self.logger.do_log(
                    f"[SENT-V2][{symbol}] 📊 Sentiment={metrics['mdna_sentiment']:.3f} | Sentences={metrics['financial_sentences']} | Forward={metrics['forward_ratio']:.1%} | Hedge={metrics['hedge_ratio']:.1%}",
                    MessageType.INFO
                )

                output = {
                    "symbol": symbol,
                    "year": self.year,
                    "period": period,
                    "report_type": self.report_type,
                    "model_used": "Loughran-McDonald + VADER (calibrated)",
                    "metrics": metrics,
                    "top_positive": result["top_positive"],
                    "top_negative": result["top_negative"],
                    "forward_snippets": result["forward_snippets"],
                    "curated_text": result["curated_text"],
                }

                out_path = self.output_dir / f"{symbol}_{self.year}_{period}_sentiment.json"
                out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

                self.logger.do_log(f"[SENT-V2][{symbol}] 💾 SAVED ✅ ({period}) – {out_path.name}", MessageType.INFO)
                success_count += 1

            except Exception as e:
                self.logger.do_log(f"[SENT-V2][{symbol}] 💥 CRASH ❌ {str(e)[:100]}", MessageType.ERROR)
                failed_count += 1
                failed_symbols.append(symbol)


        self._log_summary(success_count, failed_count, skipped_count, total_files, failed_symbols)

    def consolidate_year(self,
            year: int,
            report_type:str,
            quarter: int=None,
            job_id=None

    ) -> str:
        """
        Merge all *_sentiment.json files for a given year and report type (K10 or Q10)
        into a single consolidated JSON file, saving it under the rank_folder.
        """
        self.logger.do_log(f"[SENT] 🧭 Reading from base_dir={self.output_dir}", MessageType.INFO,job_id)

        if not os.path.isdir(self.output_dir):
            self.logger.do_log(f"[SENT] ⚠ Year folder not found: {self.output_dir}", MessageType.WARNING,job_id)
            return ""

        data = []
        if report_type == ReportFolder.K10.value:
            pattern = re.compile(rf".*_{year}_Y{year}_sentiment\.json$", re.IGNORECASE)
            rank_dir = os.path.join(
                self.root_dir,
                Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                self.rank_folder,
                f"{report_type}_sentiment_summary_report_rank",
                str(year)
            )
        elif report_type == ReportFolder.Q10.value:
            pattern = re.compile(rf".*_{year}_Q{quarter}_sentiment\.json$", re.IGNORECASE)
            rank_dir = os.path.join(
                self.root_dir,
                Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                self.rank_folder,
                f"{report_type}_sentiment_summary_report_Q{quarter}_rank",
                str(year)
            )
        else:
            raise Exception(f"Invalid report type consolidating: {report_type}")

        for fn in os.listdir(self.output_dir):
            if pattern.match(fn):
                path = os.path.join(self.output_dir, fn)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        j = json.load(fh)
                    if j.get("year") == year:
                        data.append(j)
                except Exception as e:
                    self.logger.do_log(f"[SENT] ❌ Failed to read {fn} - {e}", MessageType.ERROR,job_id)

        # --- Output folder (ranked consolidated JSON) ---


        os.makedirs(rank_dir, exist_ok=True)
        self.logger.do_log(f"[SENT] 🧭 Writing to rank_dir={rank_dir}", MessageType.INFO,job_id)

        out_path = os.path.join(rank_dir, f"sentiment_summary_all_{year}.json")

        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)

        self.logger.do_log(f"[SENT] ✅ Consolidated -> {out_path} ({len(data)} filers)", MessageType.INFO,job_id)
        return out_path


    # ------------------------------------------------------------------ #
    # Consolidation
    # ------------------------------------------------------------------ #
    def consolidate_year(
            self,
            year: int,
            report_type: str,
            quarter: int = None,
            job_id=None,
    ) -> str:
        """
        Merge all *_sentiment.json files for a given year and report type
        into a single consolidated JSON file, saving it under the rank_folder.
        """
        self.logger.do_log(
            f"[SENT] 🧭 Reading from base_dir={self.output_dir}",
            MessageType.INFO,
            job_id,
        )

        if not os.path.isdir(self.output_dir):
            self.logger.do_log(
                f"[SENT] ⚠ Year folder not found: {self.output_dir}",
                MessageType.WARNING,
                job_id,
            )
            return ""

        data = []

        if report_type == ReportFolder.K10.value:
            pattern = re.compile(rf".*_{year}_Y{year}_sentiment\.json$", re.I)
            rank_dir = os.path.join(
                self.root_dir,
                Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                self.rank_folder,
                f"{report_type}_sentiment_summary_report_rank",
                str(year),
            )

        elif report_type == ReportFolder.Q10.value:
            pattern = re.compile(rf".*_{year}_Q{quarter}_sentiment\.json$", re.I)
            rank_dir = os.path.join(
                self.root_dir,
                Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                self.rank_folder,
                f"{report_type}_sentiment_summary_report_Q{quarter}_rank",
                str(year),
            )
        else:
            raise Exception(f"Invalid report type consolidating: {report_type}")

        for fn in os.listdir(self.output_dir):
            if pattern.match(fn):
                with open(os.path.join(self.output_dir, fn), "r", encoding="utf-8") as fh:
                    j = json.load(fh)
                    if j.get("year") == year:
                        data.append(j)

        os.makedirs(rank_dir, exist_ok=True)
        out_path = os.path.join(rank_dir, f"sentiment_summary_all_{year}.json")

        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)

        self.logger.do_log(
            f"[SENT] ✅ Consolidated -> {out_path} ({len(data)} filers)",
            MessageType.INFO,
            job_id,
        )

        return out_path