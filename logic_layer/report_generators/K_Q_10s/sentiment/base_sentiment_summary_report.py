import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from common.util.std_in_out.root_locator import RootLocator

nltk.download('punkt', quiet=True)

from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from framework.common.logger.message_type import MessageType


# ===== sentiment_analysis_base.py =====

import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict

from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from common.util.std_in_out.root_locator import RootLocator
from framework.common.logger.message_type import MessageType

nltk.download("punkt", quiet=True)


class SentimentAnalysisBase:
    """
    BASE CLASS – DO NOT CHANGE LOGIC.
    This class preserves EXACT behavior from the old SentimentSummaryReportV2.
    """

    LEGAL_BLACKLIST = re.compile(
        r"(indenture|exhibit|articles of incorporation|bylaws|trustee|supplemental indenture|code of ethics)",
        re.I,
    )

    FORWARD_CUES = re.compile(
        r"(we (expect|anticipate|believe|plan|intend|will|continue)|outlook|guidance|pipeline|visibility)",
        re.I,
    )

    HEDGING_CUES = re.compile(
        r"(may|might|could|uncertain|uncertainty|volatile|volatility|subject to)",
        re.I,
    )

    def __init__(self, logger):
        self.logger = logger
        self.root_dir = RootLocator.get_root()

        lm_dict_path = (
            self.root_dir
            / "static"
            / "dictionaries"
            / "Loughran-McDonald_MasterDictionary_1993-2024.csv"
        )

        self.lm_dict = self._load_loughran_mcdonald(lm_dict_path)
        self.vader = SentimentIntensityAnalyzer()

        self.logger.do_log(
            "[SENT-BASE] Loughran-McDonald + VADER initialized",
            MessageType.INFO,
        )

    # ------------------------------------------------------------------ #
    # Dictionary loading
    # ------------------------------------------------------------------ #
    def _load_loughran_mcdonald(self, csv_path: Path) -> Dict[str, Dict[str, int]]:
        """Load Loughran-McDonald master dictionary safely from CSV."""
        if not csv_path.exists():
            raise FileNotFoundError(f"LM Dictionary not found at {csv_path}")

        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

        # Ensure required columns exist
        required_cols = ["Word", "Negative", "Positive", "Uncertainty"]
        for col in required_cols:
            if col not in df.columns:
                self.logger.do_log(f"[SENT-V2] Column '{col}' missing – filling with 0", MessageType.WARNING)
                df[col] = 0

        df = df[required_cols].copy()
        df["Word"] = df["Word"].astype(str).str.strip().str.lower()
        df = df[df["Word"].str.len() > 0]
        df[["Negative", "Positive", "Uncertainty"]] = df[["Negative", "Positive", "Uncertainty"]].fillna(0).astype(int)

        word_scores = {}
        for word, neg, pos, unc in zip(df["Word"], df["Negative"], df["Positive"], df["Uncertainty"]):
            word_scores[word] = {
                "positive": int(pos),
                "negative": int(neg),
                "uncertainty": int(unc),
            }

        self.logger.do_log(f"[SENT-V2] Loaded {len(word_scores)} words from LM dictionary", MessageType.INFO)
        return word_scores


    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def _html_to_text(self, file_path: Path) -> str:
        return BeautifulSoup(
            file_path.read_text(encoding="utf-8"),
            "html.parser",
        ).get_text(" ", strip=True)

    # ------------------------------------------------------------------ #
    # Core processing methods
    # ------------------------------------------------------------------ #
    def _html_to_text(self, file_path: Path) -> str:
        """Extract clean text from HTML filing."""
        return BeautifulSoup(file_path.read_text(encoding="utf-8"), "html.parser").get_text(" ", strip=True)

    def _extract_mdna(self, text: str, symbol: str) -> str:
        """Extract MD&A with DETAILED logging per symbol."""
        text = re.sub(r'\s+', ' ', text)

        # Comprehensive start patterns (covers 95%+ of 10-Ks)
        start_patterns = [
            r"Part\s+II\s*Item\s*7",  # Part II Item 7
            r"Item\s*7[.:]?\s*(Management|MD&A)",
            r"Item\s*7[.:]?\s*Management['’s]?\s+Discussion\s+and\s+Analysis",
            r"MANAGEMENT['’S]?\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION",
            r"Management['’s]?\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition",
            r"MD&A",  # Simple MD&A
            r"Financial\s+Review|Operating\s+and\s+Financial\s+Review",
        ]

        start_pos = len(text)
        matched_pattern = "NONE"
        for pattern in start_patterns:
            match = re.search(pattern, text, re.I)
            if match and match.start() < start_pos:
                start_pos = match.start()
                matched_pattern = pattern[:50] + "..."

        if start_pos == len(text):
            self.logger.do_log(f"[SENT-V2][{symbol}] 🔍 NO START PATTERN matched (tried {len(start_patterns)})",
                               MessageType.WARNING)

            # ULTIMATE FALLBACK: keyword-based
            fallback = re.search(r"\b(revenue|operating\s+results?|financial\s+condition|results\s+of\s+operations)\b",
                                 text[:100000], re.I)
            if fallback:
                start_pos = max(0, fallback.start() - 5000)
                matched_pattern = "FALLBACK-revenue"
                self.logger.do_log(f"[SENT-V2][{symbol}] 🔄 FALLBACK start @ {start_pos:,} (revenue keyword)",
                                   MessageType.WARNING)
            else:
                return ""
        else:
            self.logger.do_log(f"[SENT-V2][{symbol}] 🎯 START '{matched_pattern}' @ {start_pos:,}", MessageType.DEBUG)

        # Stop patterns
        tail = text[start_pos:start_pos + 30000]  # Limit search
        stop_pos = len(text)
        stop_patterns = [
            r"Item\s*7A[.:]?\s*Quantitative", r"Item\s*8[.:]?\s*Financial",
            r"FINANCIAL\s+STATEMENTS", r"Report\s+of\s+Independent",
            r"ITEM\s*8\.", r"Part\s+II\s*Item\s*7A",
        ]

        for pattern in stop_patterns:
            match = re.search(pattern, tail, re.I)
            if match:
                stop_pos = start_pos + match.start()
                self.logger.do_log(f"[SENT-V2][{symbol}] 🛑 STOP '{pattern[:30]}...' @ {stop_pos:,}", MessageType.DEBUG)
                break

        mdna_text = text[start_pos:stop_pos].strip()

        # Quality check
        if len(mdna_text) < 2000:
            self.logger.do_log(f"[SENT-V2][{symbol}] ⚠️  MD&A SHORT {len(mdna_text):,} chars → FALLBACK 20k",
                               MessageType.WARNING)
            mdna_text = text[start_pos:start_pos + 20000]
        else:
            self.logger.do_log(f"[SENT-V2][{symbol}] 📏 MD&A RAW {len(mdna_text):,} chars", MessageType.DEBUG)

        return re.sub(r'\s+', ' ', mdna_text)


    def _extract_period_from_filename(self, filename: str) -> str:
        """Detect quarter (Q1-Q4) or annual (YXXXX) from filename."""
        name = Path(filename).stem
        if m := re.search(r"_Q([1-4])_", name):
            return f"Q{m.group(1)}"
        if m := re.search(r"_(\d{4})_", name):
            return f"Y{m.group(1)}"
        return "UNKNOWN"


    # ------------------------------------------------------------------ #
    # LM sentence scoring (calibrated)
    # ------------------------------------------------------------------ #
    def _lm_score_sentence(self, sentence: str) -> float:
        """Calculate calibrated LM tone score for a single sentence."""
        words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
        matches = [w for w in words if w in self.lm_dict]

        if not matches:
            return 0.0

        pos = sum(self.lm_dict[w]["positive"] for w in matches)
        neg = sum(self.lm_dict[w]["negative"] for w in matches)
        unc = sum(self.lm_dict[w]["uncertainty"] for w in matches)
        total = len(matches)

        tone = (pos - neg) / total
        tone = tone * (1 - unc / total)  # Uncertainty attenuates tone

        # Reasonable bounds to avoid outliers
        tone = max(min(tone, 3.0), -3.0)

        return tone



    # ------------------------------------------------------------------ #
    # MD&A scoring (calibrated hybrid)
    # ------------------------------------------------------------------ #
    def _score_mdna(self, text: str) -> Dict:
        """Score the MD&A section with calibrated LM + VADER hybrid."""
        sentences = sent_tokenize(text)
        scored_sentences = []
        forward_snippets = []
        hedge_count = 0
        seen = set()

        for s in sentences:
            if len(s) < 40 or self.LEGAL_BLACKLIST.search(s) or s in seen:
                continue
            seen.add(s)

            lm_score = self._lm_score_sentence(s)
            vader_score = self.vader.polarity_scores(s)["compound"]

            # Hybrid: LM dominates financial tone, VADER adds general nuance
            compound = 0.6 * lm_score + 0.4 * vader_score
            scored_sentences.append((s, compound))

            if self.FORWARD_CUES.search(s):
                forward_snippets.append(s)
            if self.HEDGING_CUES.search(s):
                hedge_count += 1

        if not scored_sentences:
            avg = 0.0
            top_pos = top_neg = []
        else:
            avg = round(sum(score for _, score in scored_sentences) / len(scored_sentences), 4)
            sorted_scored = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
            top_pos = sorted_scored[:5]
            top_neg = sorted(sorted_scored, key=lambda x: x[1])[:5]

        curated = (
            "Key MD&A sentiment: "
            + " ".join(s for s, _ in top_pos[:2])
            + " "
            + " ".join(s for s, _ in top_neg[:2])
        )

        return {
            "metrics": {
                "mdna_sentiment": avg,
                "outlook_sentiment": avg,
                "forward_ratio": round(len(forward_snippets) / max(1, len(sentences)), 3),
                "hedge_ratio": round(hedge_count / max(1, len(scored_sentences)), 3),
                "financial_sentences": len(scored_sentences),
                "hedge_sentences": hedge_count,
            },
            "top_positive": [{"sent": s, "score": round(v, 4)} for s, v in top_pos],
            "top_negative": [{"sent": s, "score": round(v, 4)} for s, v in top_neg],
            "forward_snippets": forward_snippets[:8],
            "curated_text": curated,
        }


    def _log_summary(self, success: int, failed: int, skipped: int, total: int, failed_symbols: List[str]) -> None:
        """Private method to log the final execution summary."""
        success_pct = (success / total * 100) if total > 0 else 0

        self.logger.do_log(
            f"[SENT-V2] 🎯 SUMMARY: {success}/{total} SUCCESS ({success_pct:.0f}%) | "
            f"{failed} FAILED | {skipped} SKIPPED",
            MessageType.INFO
        )

        if failed_symbols:
            self.logger.do_log(f"[SENT-V2] ❌ Failed symbols: {', '.join(sorted(set(failed_symbols)))}",
                               MessageType.WARNING)
        else:
            self.logger.do_log("[SENT-V2] ✅ All filings processed successfully!", MessageType.INFO)

    def _build_single_file_path(
            self,
            portfolio: str,
            report_type: str,
            year: int,
            symbol: str,
            quarter: Optional[int] = None
    ) -> Path:
        """Build the expected path for a filing HTML."""
        base_path = (
                self.root_dir
                / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                / portfolio
                / report_type
                / str(year)
        )

        # Pattern: SYMBOL_YEAR_QQUARTER_10Q.html or SYMBOL_YEAR_10K.html
        if quarter and report_type == ReportFolder.Q10.value:
            pattern = f"{symbol}_{year}_Q{quarter}_10-Q.html"
        else:
            pattern = f"{symbol}_{year}_10-K.html"

        return base_path / pattern