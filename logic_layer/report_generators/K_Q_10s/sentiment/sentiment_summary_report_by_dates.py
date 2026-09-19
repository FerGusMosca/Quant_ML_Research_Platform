"""
Sentiment summary report driven by FILING DATES instead of report type.

The older reports pick a report type (annual or quarterly) and a year, and walk
the folder of that type. This one starts from the securities calendar: it asks
which filings actually landed between two dates, and scores whatever it finds,
mixing annual and quarterly filings inside a single ranking.

Nothing about the scoring changes: the whole calculation is inherited from
SentimentAnalysisBase, exactly as the year-based reports use it.
"""

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import KQ10HtmlStructuredBlockExtractor
from framework.common.logger.message_type import MessageType
from logic_layer.report_generators.K_Q_10s.sentiment.base_sentiment_summary_report import SentimentAnalysisBase
from logic_layer.report_generators.K_Q_10s.sentiment.sentence_sentiment_summary_report import SentimentSummaryReport


class SentimentSummaryReportByDates(SentimentAnalysisBase):
    """
    Scores every filing whose filing date falls inside [date_from, date_to],
    regardless of whether it is a 10-K or a 10-Q, and leaves one consolidated
    ranking (csv + json) for the whole range.
    """

    FOLDER_TAG = "BY_DATES_sentiment_summary_report"

    # How each calendar date column maps to a filing on disk.
    # (marker looked up inside the column name, report folder, quarter)
    CALENDAR_SLOTS = [
        ("q1", ReportFolder.Q10.value, 1),
        ("q2", ReportFolder.Q10.value, 2),
        ("q3", ReportFolder.Q10.value, 3),
        ("k10", ReportFolder.K10.value, None),
    ]

    def __init__(
            self,
            date_from,
            date_to,
            logger,
            calendar_mgr,
            portfolio: str = None,
            filers_whitelist: List[str] = None,
            universe_key: str = None,
            dest_folder: str = None,
            rank_folder: str = None,
    ):
        super().__init__(logger)

        self.date_from = self._as_date(date_from)
        self.date_to = self._as_date(date_to)

        if self.date_from is None or self.date_to is None:
            raise Exception("[SENT-DATES] date_from and date_to are required (YYYY-MM-DD)")

        if self.date_to < self.date_from:
            self.date_from, self.date_to = self.date_to, self.date_from

        self.calendar_mgr = calendar_mgr
        self.portfolio = portfolio
        self.dest_folder = dest_folder
        self.rank_folder = rank_folder
        self.universe_key = universe_key
        self.whitelist = set(t.upper() for t in filers_whitelist) if filers_whitelist else None

        self.range_key = f"{self.date_from.isoformat()}_{self.date_to.isoformat()}"

        # One JSON per filing lives here.
        range_dir = (
                self.root_dir
                / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                / dest_folder
                / self.FOLDER_TAG
                / self.range_key
        )
        self.output_dir = range_dir / universe_key if universe_key else range_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Each run gets its own folder inside the rank folder, named after the
        # range, and the three files of that run live inside it.
        self.rank_dir = (
                self.root_dir
                / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                / rank_folder
                / self.range_key
        )

        self.logger.do_log(
            f"[SENT-DATES] Initialized for {self.date_from} .. {self.date_to} | "
            f"portfolio={portfolio} | output={self.output_dir}",
            MessageType.INFO,
        )

    # ------------------------------------------------------------------ #
    # Date helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_date(value) -> Optional[date]:
        """Accept a date, a datetime or a YYYY-MM-DD string."""
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value).strip()[:10]
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _pick_column(row: dict, marker: str):
        """
        The calendar rows come straight from the stored procedure, so the exact
        column names are not hardcoded here: the slot is found by the marker it
        carries (q1, q2, q3, k10) instead of by an exact name.
        """
        marker = marker.lower()
        for key, value in row.items():
            name = str(key).lower()
            if marker in name and "date" in name:
                return value
        for key, value in row.items():
            if marker in str(key).lower():
                return value
        return None

    @staticmethod
    def _row_value(row: dict, marker: str):
        for key, value in row.items():
            if str(key).lower() == marker:
                return value
        for key, value in row.items():
            if marker in str(key).lower():
                return value
        return None

    # ------------------------------------------------------------------ #
    # Work list
    # ------------------------------------------------------------------ #
    def build_worklist(self, job_id=None) -> List[Dict]:
        """
        Translate the date range into the list of filings that landed inside it.

        Returns one entry per filing: symbol, fiscal year, report type, quarter
        and the day it was actually filed.
        """
        year_from = self.date_from.year
        year_to = self.date_to.year

        rows = self.calendar_mgr.get_calendar_rows(from_year=year_from, to_year=year_to)

        self.logger.do_log(
            f"[SENT-DATES] 📅 Calendar rows for {year_from}-{year_to}: {len(rows)}",
            MessageType.INFO,
            job_id,
        )

        worklist = []

        for row in rows:
            symbol = str(self._row_value(row, "symbol") or "").strip().upper()
            if not symbol:
                continue

            if self.whitelist and symbol not in self.whitelist:
                continue

            fiscal_year = self._row_value(row, "fiscal_year")
            try:
                fiscal_year = int(fiscal_year)
            except (TypeError, ValueError):
                continue

            for marker, report_type, quarter in self.CALENDAR_SLOTS:
                filing_date = self._as_date(self._pick_column(row, marker))
                if filing_date is None:
                    continue
                if filing_date < self.date_from or filing_date > self.date_to:
                    continue

                worklist.append({
                    "symbol": symbol,
                    "fiscal_year": fiscal_year,
                    "report_type": report_type,
                    "quarter": quarter,
                    "filing_date": filing_date.isoformat(),
                })

        worklist.sort(key=lambda item: (item["symbol"], item["filing_date"]))

        self.logger.do_log(
            f"[SENT-DATES] 🎯 Filings inside the range: {len(worklist)}",
            MessageType.INFO,
            job_id,
        )

        return worklist

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, job_id=None) -> Dict:
        """Score every filing in the range and drop one JSON per filing."""
        worklist = self.build_worklist(job_id)
        total = len(worklist)

        # Same structural extractor the Document Tagger uses, asked for MD&A only
        mdna_extractor = KQ10HtmlStructuredBlockExtractor()

        success = 0
        failed = 0
        missing = 0
        failed_symbols = []
        missing_symbols = []

        for i, task in enumerate(worklist, 1):
            symbol = task["symbol"]
            report_type = task["report_type"]
            fiscal_year = task["fiscal_year"]
            quarter = task["quarter"]
            period = f"Q{quarter}" if quarter else f"Y{fiscal_year}"

            self.logger.do_log(
                f"[SENT-DATES][{i}/{total}] 🔄 {symbol} {report_type} {period} "
                f"(filed {task['filing_date']})",
                MessageType.INFO,
                job_id,
            )

            file_path = self._build_single_file_path(
                portfolio=self.portfolio,
                report_type=report_type,
                year=fiscal_year,
                symbol=symbol,
                quarter=quarter,
            )

            if not os.path.isfile(file_path):
                self.logger.do_log(
                    f"[SENT-DATES][{symbol}] ⏭ Filing not downloaded – skipped ({period} {fiscal_year})",
                    MessageType.WARNING,
                    job_id,
                )
                missing += 1
                missing_symbols.append(f"{symbol}:{period}")
                continue

            try:
                html = Path(file_path).read_text(encoding="utf-8")
                mdna = mdna_extractor.extract_mdna(html, report_type)

                if not mdna or len(mdna.strip()) < 500:
                    self.logger.do_log(
                        f"[SENT-DATES][{symbol}] ❌ MD&A FAILED – {len(mdna or '')} chars",
                        MessageType.WARNING,
                        job_id,
                    )
                    failed += 1
                    failed_symbols.append(f"{symbol}:{period}")
                    continue

                result = self._score_mdna(mdna)
                metrics = result["metrics"]

                self.logger.do_log(
                    f"[SENT-DATES][{symbol}] 📊 Sentiment={metrics['mdna_sentiment']:.3f} | "
                    f"Sentences={metrics['financial_sentences']} | "
                    f"Forward={metrics['forward_ratio']:.1%} | Hedge={metrics['hedge_ratio']:.1%}",
                    MessageType.INFO,
                    job_id,
                )

                output = {
                    "symbol": symbol,
                    "year": fiscal_year,
                    "period": period,
                    "report_type": report_type,
                    "filing_date": task["filing_date"],
                    "date_from": self.date_from.isoformat(),
                    "date_to": self.date_to.isoformat(),
                    "model_used": "Loughran-McDonald + VADER (calibrated)",
                    "metrics": metrics,
                    "top_positive": result["top_positive"],
                    "top_negative": result["top_negative"],
                    "forward_snippets": result["forward_snippets"],
                    "curated_text": result["curated_text"],
                }

                out_path = self.output_dir / f"{symbol}_{fiscal_year}_{period}_sentiment.json"
                out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

                self.logger.do_log(
                    f"[SENT-DATES][{symbol}] 💾 SAVED ✅ ({period}) – {out_path.name}",
                    MessageType.INFO,
                    job_id,
                )
                success += 1

            except Exception as e:
                self.logger.do_log(
                    f"[SENT-DATES][{symbol}] 💥 CRASH ❌ {str(e)[:120]}",
                    MessageType.ERROR,
                    job_id,
                )
                failed += 1
                failed_symbols.append(f"{symbol}:{period}")

        self._log_range_summary(total, success, failed, missing,
                                failed_symbols, missing_symbols, job_id)

        return {
            "planned": total,
            "processed": success,
            "failed": failed,
            "missing": missing,
        }

    # ------------------------------------------------------------------ #
    # Consolidation
    # ------------------------------------------------------------------ #
    def consolidate_range(self, job_id=None) -> str:
        """
        Merge every JSON produced for this range into a single consolidated
        file under the rank folder. Annual and quarterly filings live together
        here on purpose: that is the whole point of the report.
        """
        if not os.path.isdir(self.output_dir):
            self.logger.do_log(
                f"[SENT-DATES] ⚠ Range folder not found: {self.output_dir}",
                MessageType.WARNING,
                job_id,
            )
            return ""

        pattern = re.compile(r".*_sentiment\.json$", re.IGNORECASE)
        data = []

        for fn in sorted(os.listdir(self.output_dir)):
            if not pattern.match(fn):
                continue
            try:
                with open(os.path.join(self.output_dir, fn), "r", encoding="utf-8") as fh:
                    data.append(json.load(fh))
            except Exception as e:
                self.logger.do_log(
                    f"[SENT-DATES] ❌ Failed to read {fn} - {e}",
                    MessageType.ERROR,
                    job_id,
                )

        os.makedirs(self.rank_dir, exist_ok=True)
        out_path = os.path.join(self.rank_dir, f"sentiment_summary_all_{self.range_key}.json")

        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)

        self.logger.do_log(
            f"[SENT-DATES] ✅ Consolidated -> {out_path} ({len(data)} filings)",
            MessageType.INFO,
            job_id,
        )

        return out_path

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #
    def rank_range(self, consolidated_json: str, job_id=None) -> Dict[str, str]:
        """
        Ranking for the whole range, in csv and json.

        The csv is produced by the SAME ranking function the year-based reports
        already use, so the formula and the ordering are untouched. The json is
        the very same table, only re-serialized, so both files always agree.
        """
        if not consolidated_json or not os.path.exists(consolidated_json):
            self.logger.do_log(
                f"[SENT-DATES] ❌ Nothing to rank: {consolidated_json}",
                MessageType.ERROR,
                job_id,
            )
            return {"csv": "", "json": ""}

        out_csv = os.path.join(self.rank_dir, f"sentiment_summary_ranking_{self.range_key}.csv")

        SentimentSummaryReport.rank(consolidated_json, out_csv, self.logger, job_id)

        if not os.path.exists(out_csv):
            self.logger.do_log(
                "[SENT-DATES] ⚠ Ranking produced no csv, so there is no json either",
                MessageType.WARNING,
                job_id,
            )
            return {"csv": "", "json": ""}

        # The report type already travels in every row, so which filing fed each
        # score is readable straight from the ranking.
        df = pd.read_csv(out_csv)

        # Bring the filing date next to each row, so the ranking can be read
        # against the calendar without opening the consolidated file.
        try:
            with open(consolidated_json, "r", encoding="utf-8") as fh:
                consolidated = json.load(fh)

            dates = {
                (str(item.get("symbol")), int(item.get("year") or 0), str(item.get("report_type"))):
                    item.get("filing_date")
                for item in consolidated
            }

            df["filing_date"] = [
                dates.get((str(row.symbol), int(row.year or 0), str(row.report_type)))
                for row in df.itertuples()
            ]
            df.to_csv(out_csv, index=False)
        except Exception as e:
            self.logger.do_log(
                f"[SENT-DATES] ⚠ Could not attach the filing dates to the ranking - {e}",
                MessageType.WARNING,
                job_id,
            )

        out_json = os.path.join(self.rank_dir, f"sentiment_summary_ranking_{self.range_key}.json")
        df.to_json(out_json, orient="records", indent=2)

        self.logger.do_log(
            f"[SENT-DATES] ✅ Ranking -> {out_csv} | {out_json} ({len(df)} rows)",
            MessageType.INFO,
            job_id,
        )

        return {"csv": out_csv, "json": out_json}

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log_range_summary(self, total, success, failed, missing,
                           failed_symbols, missing_symbols, job_id=None) -> None:
        success_pct = (success / total * 100) if total > 0 else 0

        self.logger.do_log(
            f"[SENT-DATES] 🎯 SUMMARY {self.range_key}: {success}/{total} SUCCESS "
            f"({success_pct:.0f}%) | {failed} FAILED | {missing} NOT DOWNLOADED",
            MessageType.INFO,
            job_id,
        )

        if missing_symbols:
            self.logger.do_log(
                f"[SENT-DATES] ⏭ Not downloaded: {', '.join(sorted(set(missing_symbols)))}",
                MessageType.WARNING,
                job_id,
            )

        if failed_symbols:
            self.logger.do_log(
                f"[SENT-DATES] ❌ Failed: {', '.join(sorted(set(failed_symbols)))}",
                MessageType.WARNING,
                job_id,
            )
