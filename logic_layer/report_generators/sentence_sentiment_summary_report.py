import os
import re
import json
from typing import Dict, List

from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from framework.common.logger.message_type import MessageType


class SentimentSummaryReport:
    """
    Extract and score management sentiment/guidance from 10-K / 20-F.
    Lexicon-based sentiment (VADER) + forward/hedging cues.

    Key changes vs the original version:
    - Output is now namespaced by year: ./output/K10/sentiment_summary_report/<YEAR>[/<universe_key>]
    - New consolidate_year(year, ...) that only merges files from that year directory
      and double-checks the 'year' field inside each JSON.
    - A legacy consolidate(...) method remains for backwards-compatibility, but it
      is recommended to switch calls to consolidate_year(...).
    """

    # Base folders (do not include year here)
    input_base: str = "./output/K10/"
    output_base: str = "./output/K10/sentiment_summary_report"

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

    def __init__(self, year: int, logger, filers_whitelist: List[str] = None, universe_key: str = None):
        """
        :param year: Filing year to process
        :param logger: Logger instance (must support .do_log(msg, MessageType))
        :param filers_whitelist: Optional list of tickers to restrict
        :param universe_key: Optional universe name for sub-folder under the year
        """
        # Input HTML lives in ./output/K10/<YEAR>
        self.input_dir = os.path.join(SentimentSummaryReport.input_base, str(year))

        # >>> Output is now namespaced by year (and optional universe key) <<<
        year_dir = os.path.join(SentimentSummaryReport.output_base, str(year))
        self.output_dir = os.path.join(year_dir, universe_key) if universe_key else year_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.year = year
        self.logger = logger
        self.whitelist = set(t.upper() for t in filers_whitelist) if filers_whitelist else None

        # Load VADER (download lexicon on first run if needed)
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            nltk.download("vader_lexicon")
            self.sia = SentimentIntensityAnalyzer()

    # -------- Public API --------
    def run(self) -> None:
        """Process HTML filings and save one *_sentiment.json per symbol under the YEAR folder."""
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".html")]
        if self.whitelist:
            files = [f for f in files if f.split("_")[0].upper() in self.whitelist]

        self.logger.do_log(f"[SENT] Processing {len(files)} file(s) for year {self.year}", MessageType.INFO)

        for i, file in enumerate(sorted(files), start=1):
            try:
                symbol = file.split("_")[0].upper()
                with open(os.path.join(self.input_dir, file), "r", encoding="utf-8") as fh:
                    html = fh.read()

                text = self._html_to_text(html)
                sections = self._extract_relevant_sections(text)
                result = self._score_sections(sections)

                out = {
                    "symbol": symbol,
                    "year": self.year,
                    "metrics": result["metrics"],
                    "top_positive": result["top_positive"],
                    "top_negative": result["top_negative"],
                    "forward_snippets": result["forward_snippets"],
                }
                out_path = os.path.join(self.output_dir, f"{symbol}_{self.year}_sentiment.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

                self.logger.do_log(f"[SENT][{i}] ✅ {symbol} saved", MessageType.INFO)

            except Exception as e:
                self.logger.do_log(f"[SENT][{i}] ❌ {file} failed - {str(e)}", MessageType.ERROR)

    @staticmethod
    def consolidate_year(year: int, logger, universe_key: str = None) -> str:
        """
        Merge all *_{year}_sentiment.json for a single year into one JSON file.

        It reads from: ./output/K10/sentiment_summary_report/<year>[/<universe_key>]
        and writes:     ./output/K10/sentiment_summary_report/<year>[/<universe_key>]/sentiment_summary_all_<year>.json

        Double-checks the embedded 'year' field inside each file to avoid any mix-ups.
        """
        base = os.path.join(SentimentSummaryReport.output_base, str(year))
        if universe_key:
            base = os.path.join(base, universe_key)

        data = []
        if not os.path.isdir(base):
            logger.do_log(f"[SENT] ⚠ Year folder not found: {base}", MessageType.WARNING)
        else:
            for fn in os.listdir(base):
                if fn.endswith(f"_{year}_sentiment.json"):
                    path = os.path.join(base, fn)
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            j = json.load(fh)
                        if j.get("year") == year:
                            data.append(j)
                        else:
                            logger.do_log(f"[SENT] ⚠ Skipped (wrong embedded year): {fn}", MessageType.WARNING)
                    except Exception as e:
                        logger.do_log(f"[SENT] ❌ Failed to read {fn} - {e}", MessageType.ERROR)

        out_path = os.path.join(base, f"sentiment_summary_all_{year}.json")
        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
        logger.do_log(f"[SENT] Consolidated -> {out_path} ({len(data)} filers)", MessageType.INFO)
        return out_path

    @staticmethod
    def consolidate(input_dir: str, out_path: str, logger, year: int = None) -> None:
        """
        Legacy consolidator kept for backwards compatibility.
        NOTE: Prefer consolidate_year(year, logger, universe_key) to avoid mixing years.

        If 'year' is provided, it will only include files ending with _{year}_sentiment.json
        and also check the embedded 'year' field inside each JSON.
        If 'year' is None, it will fall back to merging every *_sentiment.json in input_dir.
        """
        data = []
        try:
            for f in os.listdir(input_dir):
                if not f.endswith("_sentiment.json"):
                    continue

                if year is not None and not f.endswith(f"_{year}_sentiment.json"):
                    continue  # skip other years when explicit filter is requested

                with open(os.path.join(input_dir, f), "r", encoding="utf-8") as fh:
                    j = json.load(fh)

                if year is not None:
                    if j.get("year") != year:
                        logger.do_log(f"[SENT] ⚠ Skipped (wrong embedded year): {f}", MessageType.WARNING)
                        continue

                data.append(j)

            with open(out_path, "w", encoding="utf-8") as out:
                json.dump(data, out, indent=2)

            logger.do_log(f"[SENT] (legacy) Consolidated -> {out_path} ({len(data)} filers)", MessageType.INFO)
            if year is None:
                logger.do_log("[SENT] ⚠ Using legacy consolidate without year may mix multiple years. "
                              "Prefer consolidate_year(...).", MessageType.WARNING)

        except Exception as e:
            logger.do_log(f"[SENT] ❌ consolidate failed - {e}", MessageType.ERROR)

    @staticmethod
    def rank(consolidated_json: str, out_csv: str, logger) -> None:
        """Produce ranking CSV by optimism composite score from a consolidated JSON."""
        import pandas as pd

        with open(consolidated_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for e in data:
            m = e.get("metrics", {})
            rows.append(
                {
                    "symbol": e.get("symbol"),
                    "year": e.get("year"),
                    "sentiment_mdna": m.get("mdna_sentiment", 0.0),
                    "sentiment_outlook": m.get("outlook_sentiment", 0.0),
                    "forward_ratio": m.get("forward_ratio", 0.0),
                    "hedge_ratio": m.get("hedge_ratio", 0.0),
                    "optimism_score": round(
                        0.5 * m.get("mdna_sentiment", 0)
                        + 0.5 * m.get("outlook_sentiment", 0)
                        + 0.2 * m.get("forward_ratio", 0)
                        - 0.2 * m.get("hedge_ratio", 0),
                        6,
                    ),
                }
            )

        df = pd.DataFrame(rows).sort_values(
            ["optimism_score", "sentiment_outlook", "sentiment_mdna"], ascending=False
        )
        df.to_csv(out_csv, index=False)
        logger.do_log(f"[SENT] Ranking -> {out_csv} ({len(df)} filers)", MessageType.INFO)

    # -------- Internals --------
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)

    def _extract_relevant_sections(self, text: str) -> Dict[str, str]:
        """Locate MD&A and Outlook sections using anchors and cues."""
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
        """Split text into sentences by '.', '!' or '?'."""
        return re.split(r"(?<=[.!?])\s+", txt)

    def _score_sections(self, sections: Dict[str, str]) -> Dict[str, object]:
        """Score MD&A and Outlook blocks and aggregate metrics/snippets."""
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
