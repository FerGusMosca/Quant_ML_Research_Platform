import os
import re
import json
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from framework.common.logger.message_type import MessageType


class SentimentSummaryReport:
    """
    Extract and score management sentiment/guidance from SEC filings (K10, Q10, 20-F).
    Lexicon-based sentiment (VADER) + forward/hedging cues.
    """

    # Regex cues
    FORWARD_CUES = re.compile(
        r"\b(we (expect|anticipate|believe|plan|intend|forecast|aim|target|will|continue to|remain)"
        r"|outlook|guidance|pipeline|visibility|headwind(s)?|tailwind(s)?"
        r"|margin expansion|unit economics|cash flow improvement|order backlog)\b",
        re.I,
    )
    HEDGING_CUES = re.compile(
        r"\b(may|might|could|potentially|subject to|uncertain|uncertainty|volatile|volatility)\b", re.I
    )

    # Section anchors
    SECTION_TITLES = [
        "item 7. management’s discussion and analysis",
        "item 7 management’s discussion and analysis",
        "management’s discussion and analysis",
        "md&a",
        "item 7a",
        "quantitative and qualitative disclosures",
        "item 5. operating and financial review and prospects",
        "operating and financial review and prospects",
        "outlook",
        "guidance",
        "trend",
        "trends",
    ]
    STOP_TITLES = [
        "item 8",
        "financial statements",
        "item 1a",
        "risk factors",
        "item 1b",
        "unresolved staff comments",
        "item 2",
        "properties",
        "quantitative and qualitative disclosures",
        "controls and procedures",
    ]

    def __init__(self, year: int, logger, report_type: str = ReportFolder.K10.value,portfolio: str=None, filers_whitelist: List[str] = None,
                 universe_key: str = None,dest_folder: str=None,rank_folder:str=None):
        """
        :param year: Filing year to process
        :param logger: Logger instance (must support .do_log(msg, MessageType))
        :param report_type: "K10" or "Q10"
        :param filers_whitelist: Optional list of tickers to restrict
        :param universe_key: Optional universe name for sub-folder under the year
        """
        self.report_type = report_type.upper()
        self.portfolio=portfolio
        self.dest_folder=dest_folder
        self.rank_folder=rank_folder

        # Input HTML lives in ./output/{K10|Q10}/<YEAR>
        self.input_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            portfolio,
            report_type,
            str(year)
        )
        year_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            self.dest_folder,
            f"{self.report_type}_sentiment_summary_report",
            str(year)
        )


        # Output is namespaced by report type + year (and optional universe key)
        self.output_dir = os.path.join(year_dir, universe_key) if universe_key else year_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.year = year
        self.logger = logger
        self.whitelist = set(t.upper() for t in filers_whitelist) if filers_whitelist else None

        # Load VADER
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            nltk.download("vader_lexicon")
            self.sia = SentimentIntensityAnalyzer()

    @staticmethod
    def rank(consolidated_json: str, out_csv: str, logger) -> None:
        """
        Produce ranking CSV by optimism composite score from a consolidated JSON.
        """
        import pandas as pd
        import os, json

        if not consolidated_json or not os.path.exists(consolidated_json):
            logger.do_log(f"[SENT] ❌ Consolidated JSON not found: {consolidated_json}", MessageType.ERROR)
            return

        with open(consolidated_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            logger.do_log(f"[SENT] ⚠ No data found in {consolidated_json}", MessageType.WARNING)
            return

        rows = []
        for e in data:
            m = e.get("metrics", {})
            rows.append({
                "symbol": e.get("symbol"),
                "year": e.get("year"),
                "report_type": e.get("report_type"),
                "sentiment_mdna": m.get("mdna_sentiment", 0.0),
                "sentiment_outlook": m.get("outlook_sentiment", 0.0),
                "forward_ratio": m.get("forward_ratio", 0.0),
                "hedge_ratio": m.get("hedge_ratio", 0.0),
                "optimism_score": round(
                    0.5 * m.get("mdna_sentiment", 0.0)
                    + 0.5 * m.get("outlook_sentiment", 0.0)
                    + 0.2 * m.get("forward_ratio", 0.0)
                    - 0.2 * m.get("hedge_ratio", 0.0), 6)
            })

        df = pd.DataFrame(rows)
        if df.empty:
            logger.do_log(f"[SENT] ⚠ No valid records to rank in {consolidated_json}", MessageType.WARNING)
            return

        df = df.sort_values(
            ["optimism_score", "sentiment_outlook", "sentiment_mdna"], ascending=False
        )

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)

        logger.do_log(f"[SENT] ✅ Ranking -> {out_csv} ({len(df)} filers)", MessageType.INFO)

    # -------- Public API --------
    def run(self) -> None:
        """Process filings and save one *_sentiment.json per symbol under the YEAR folder."""
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".html")]
        if self.whitelist:
            files = [f for f in files if f.split("_")[0].upper() in self.whitelist]

        self.logger.do_log(f"[SENT] Found {len(files)} {self.report_type} files to process for {self.year}",
                           MessageType.INFO)

        for i, file in enumerate(sorted(files), start=1):
            symbol = file.split("_")[0].upper()
            try:
                path = os.path.join(self.input_dir, file)
                with open(path, "r", encoding="utf-8") as fh:
                    html = fh.read()

                text = self._html_to_text(html)
                sections = self._extract_relevant_sections(text)
                result = self._score_sections(sections)
                period = self._extract_period_from_filename(file)

                curated_text = (
                        f"Symbol: {symbol}. Year: {self.year}. Period: {period}. ReportType: {self.report_type}. "
                        "Section: Management Discussion and Analysis (MD&A). "
                        "Key metrics include sentiment, forward outlook, and risk hedging. "
                        + " Top positive sentences: "
                        + " ".join(p['sent'] for p in result['top_positive'])
                        + " Top negative sentences: "
                        + " ".join(n['sent'] for n in result['top_negative'])
                        + " Forward-looking snippets: "
                        + " ".join(result['forward_snippets'])
                )

                out = {
                    "symbol": symbol,
                    "year": self.year,
                    "period": period,
                    "report_type": self.report_type,
                    "metrics": result["metrics"],
                    "top_positive": result["top_positive"],
                    "top_negative": result["top_negative"],
                    "forward_snippets": result["forward_snippets"],
                    "curated_text": curated_text
                }

                out_path = os.path.join(self.output_dir, f"{symbol}_{self.year}_{period}_sentiment.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

                self.logger.do_log(f"[SENT][{i}/{len(files)}] ✅ {symbol}: saved sentiment ({period})", MessageType.INFO)

            except Exception as e:
                self.logger.do_log(f"[SENT][{i}/{len(files)}] ❌ {symbol} failed - {e}", MessageType.ERROR)


    @staticmethod
    def consolidate_year(
            year: int,
            report_type: str,
            portfolio: str,
            logger,
            dest_folder: str,
            rank_folder: str,
            universe_key: str = None
    ) -> str:
        """
        Merge all *_sentiment.json files for a given year and report type (K10 or Q10)
        into a single consolidated JSON file, saving it under the rank_folder.
        """
        import os, re, json

        # --- Input folder (sentiment JSONs) ---
        base_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            dest_folder,
            f"{report_type}_sentiment_summary_report",
            str(year)
        )
        if universe_key:
            base_dir = os.path.join(base_dir, universe_key)

        if not os.path.isdir(base_dir):
            logger.do_log(f"[SENT] ⚠ Year folder not found: {base_dir}", MessageType.WARNING)
            return ""

        data = []
        pattern = re.compile(rf".*_{year}_(Y{year}|Q[1-4])_sentiment\.json$", re.IGNORECASE)

        for fn in os.listdir(base_dir):
            if pattern.match(fn):
                path = os.path.join(base_dir, fn)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        j = json.load(fh)
                    if j.get("year") == year:
                        data.append(j)
                except Exception as e:
                    logger.do_log(f"[SENT] ❌ Failed to read {fn} - {e}", MessageType.ERROR)

        # --- Output folder (ranked consolidated JSON) ---
        rank_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            rank_folder,
            f"{report_type}_sentiment_summary_report_rank",
            str(year)
        )
        os.makedirs(rank_dir, exist_ok=True)

        out_path = os.path.join(rank_dir, f"sentiment_summary_all_{year}.json")

        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)

        logger.do_log(f"[SENT] ✅ Consolidated -> {out_path} ({len(data)} filers)", MessageType.INFO)
        return out_path

    @staticmethod
    def consolidate_year(
            year: int,
            report_type: str,
            portfolio: str,
            logger,
            dest_folder: str,
            rank_folder: str,
            universe_key: str = None
    ) -> str:
        """
        Merge all *_sentiment.json files for a given year and report type (K10 or Q10)
        into a single consolidated JSON file, saving it under the rank_folder.
        """

        import os, re, json

        # --- Input folder (sentiment JSONs) ---
        base_dir = os.path.join(
            "/zzLotteryTicket/documents",
            dest_folder,
            f"{report_type}_sentiment_summary_report",
            str(year)
        )
        if universe_key:
            base_dir = os.path.join(base_dir, universe_key)

        base_dir = base_dir.replace("\\", "/")

        logger.do_log(f"[SENT] 🧭 Reading from base_dir={base_dir}", MessageType.INFO)

        if not os.path.isdir(base_dir):
            logger.do_log(f"[SENT] ⚠ Year folder not found: {base_dir}", MessageType.WARNING)
            return ""

        data = []
        pattern = re.compile(rf".*_{year}_(Y{year}|Q[1-4])_sentiment\.json$", re.IGNORECASE)

        for fn in os.listdir(base_dir):
            if pattern.match(fn):
                path = os.path.join(base_dir, fn)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        j = json.load(fh)
                    if j.get("year") == year:
                        data.append(j)
                except Exception as e:
                    logger.do_log(f"[SENT] ❌ Failed to read {fn} - {e}", MessageType.ERROR)

        # --- Output folder (ranked consolidated JSON) ---
        rank_dir = os.path.join(
            "/zzLotteryTicket/documents",
            rank_folder,
            f"{report_type}_sentiment_summary_report_rank",
            str(year)
        )
        rank_dir = rank_dir.replace("\\", "/")

        os.makedirs(rank_dir, exist_ok=True)
        logger.do_log(f"[SENT] 🧭 Writing to rank_dir={rank_dir}", MessageType.INFO)

        out_path = os.path.join(rank_dir, f"sentiment_summary_all_{year}.json")

        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)

        logger.do_log(f"[SENT] ✅ Consolidated -> {out_path} ({len(data)} filers)", MessageType.INFO)
        return out_path

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


    def _html_to_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)

    def _extract_relevant_sections(self, text: str) -> Dict[str, str]:
        low = text.lower()
        n = len(text)

        starts: List[int] = []
        for t in self.SECTION_TITLES:
            p = low.find(t)
            if p != -1:
                starts.append(p)
        for m in self.FORWARD_CUES.finditer(low):
            starts.append(max(0, m.start() - 800))

        if not starts:
            return {"mdna": "", "outlook": ""}

        starts = sorted(set(starts))
        s0 = starts[0]
        stops = [low.find(t, s0 + 20) for t in self.STOP_TITLES if low.find(t, s0 + 20) != -1]
        e0 = min(stops) if stops else min(n, s0 + 20000)
        mdna = text[s0:e0]

        outlook_chunks = []
        for p in starts[1:]:
            e = min(n, p + 8000)
            outlook_chunks.append(text[p:e])
        outlook_txt = "\n".join(outlook_chunks) if outlook_chunks else mdna[:8000]

        return {"mdna": mdna, "outlook": outlook_txt}

    def _split_sentences(self, txt: str) -> List[str]:
        return re.split(r"(?<=[.!?])\s+", txt)

    def _score_sections(self, sections: Dict[str, str]) -> Dict[str, object]:
        mdna_sents = self._split_sentences(sections.get("mdna", ""))
        outl_sents = self._split_sentences(sections.get("outlook", ""))

        def score_block(sents: List[str]) -> Dict[str, object]:
            pos, neg = [], []
            forward_hits = hedge_hits = cues_total = 0
            scored = []
            for s in sents:
                if not s or len(s) < 30:
                    continue
                v = self.sia.polarity_scores(s)["compound"]
                scored.append((s, v))
                if self.FORWARD_CUES.search(s):
                    cues_total += 1
                    forward_hits += 1
                if self.HEDGING_CUES.search(s):
                    hedge_hits += 1
                if v >= 0.4:
                    pos.append((s, v))
                if v <= -0.4:
                    neg.append((s, v))

            avg = round(sum(v for _, v in scored) / len(scored), 4) if scored else 0.0
            forward_ratio = round(forward_hits / max(1, cues_total), 3) if cues_total else 0.0
            hedge_ratio = round(hedge_hits / max(1, len(scored)), 3) if scored else 0.0

            pos = sorted(pos, key=lambda x: -x[1])[:5]
            neg = sorted(neg, key=lambda x: x[1])[:5]
            return {
                "avg": avg,
                "forward_ratio": forward_ratio,
                "hedge_ratio": hedge_ratio,
                "top_pos": [{"sent": s, "score": round(v, 4)} for s, v in pos],
                "top_neg": [{"sent": s, "score": round(v, 4)} for s, v in neg],
                "forward_snips": [s for s, _ in scored if self.FORWARD_CUES.search(s)][:10],
            }

        mdna = score_block(mdna_sents)
        outl = score_block(outl_sents)

        metrics = {
            "mdna_sentiment": mdna["avg"],
            "outlook_sentiment": outl["avg"],
            "forward_ratio": max(mdna["forward_ratio"], outl["forward_ratio"]),
            "hedge_ratio": max(mdna["hedge_ratio"], outl["hedge_ratio"]),
        }

        return {
            "metrics": metrics,
            "top_positive": (mdna["top_pos"][:3] + outl["top_pos"][:3])[:5],
            "top_negative": (mdna["top_neg"][:3] + outl["top_neg"][:3])[:5],
            "forward_snippets": (mdna["forward_snips"][:5] + outl["forward_snips"][:5])[:8],
        }
