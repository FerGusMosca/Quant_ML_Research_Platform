import os
import re
import json
from typing import Dict, List
from bs4 import BeautifulSoup
import pandas as pd

from framework.common.logger.message_type import MessageType


class FinancialRatiosSummaryReport:
    """
    Extract key financial line items (Revenue, COGS, GP, Assets, Debt, Equity, etc.)
    from SEC filings (K10, Q10) and compute accounting ratios.

    - GPA (Gross Profit / Assets)
    - Asset Turnover (Revenue / Assets)
    - Operating Margin (OpIncome / Revenue)
    - Debt/Equity
    - Floating (if disclosed, shares outstanding available for trading)
    """

    # Regex anchors for financial tables
    FIN_SECTION_TITLES = [
        "consolidated statements of operations",
        "consolidated statements of income",
        "consolidated statements of financial condition",
        "consolidated balance sheets",
        "financial data",
    ]

    # -------------------------
    # Dictionary of synonyms
    # -------------------------
    SYNONYMS = {
        "revenue": [
            r"Net (revenue|sales)",
            r"Total Revenues?",
            r"Net Sales",
            r"Operating Revenues?",
            r"Total net (sales|revenues?)",
            r"Sales and other operating revenues?",
        ],
        "gross_profit": [
            r"Gross Profit",
            r"Gross Margin",
            r"Total Gross Profit",
            r"Operating Gross Profit",
            r"Gross Income",
        ],
        "cost_of_revenue": [
            r"Cost of (revenue|sales)",
            r"Cost of goods sold",
            r"Total cost of (sales|revenues?)",
            r"Operating costs?",
            r"Cost of products sold",
        ],
    }

    def __init__(self, year: int, logger, report_type: str = "K10",
                 filers_whitelist: List[str] = None, universe_key: str = None):
        self.report_type = report_type.upper()
        self.input_dir = os.path.join(f"./output/{self.report_type}", str(year))
        year_dir = os.path.join(f"./output/{self.report_type}/financial_ratios_report", str(year))
        self.output_dir = os.path.join(year_dir, universe_key) if universe_key else year_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.year = year
        self.logger = logger
        self.whitelist = set(t.upper() for t in filers_whitelist) if filers_whitelist else None

    # -------------------------
    # Public API
    # -------------------------
    def run(self) -> None:
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".html")]
        if self.whitelist:
            files = [f for f in files if f.split("_")[0].upper() in self.whitelist]

        self.logger.do_log(
            f"[RATIOS] Processing {len(files)} {self.report_type} file(s) for year {self.year}",
            MessageType.INFO,
        )

        for i, file in enumerate(sorted(files), start=1):
            try:
                symbol = file.split("_")[0].upper()
                with open(os.path.join(self.input_dir, file), "r", encoding="utf-8") as fh:
                    html = fh.read()

                items = self._extract_financial_items(html)
                out = {
                    "symbol": symbol,
                    "year": self.year,
                    "report_type": self.report_type,
                    "raw_items": items,
                }
                out_path = os.path.join(self.output_dir, f"{symbol}_{self.year}_ratios.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

                self.logger.do_log(f"[RATIOS][{i}] ✅ {symbol} saved", MessageType.INFO)

            except Exception as e:
                self.logger.do_log(f"[RATIOS][{i}] ❌ {file} failed - {str(e)}", MessageType.ERROR)

    @staticmethod
    def consolidate_year(year: int, report_type: str, logger, universe_key: str = None) -> str:
        """Merge all *_{year}_ratios.json for a single year + report type into one JSON file."""
        base = os.path.join(f"./output/{report_type.upper()}/financial_ratios_report", str(year))
        if universe_key:
            base = os.path.join(base, universe_key)

        data = []
        if not os.path.isdir(base):
            logger.do_log(f"[RATIOS] ⚠ Year folder not found: {base}", MessageType.WARNING)
        else:
            for fn in os.listdir(base):
                if fn.endswith(f"_{year}_ratios.json"):
                    path = os.path.join(base, fn)
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            j = json.load(fh)
                        if j.get("year") == year:
                            data.append(j)
                        else:
                            logger.do_log(f"[RATIOS] ⚠ Skipped (wrong embedded year): {fn}",
                                          MessageType.WARNING)
                    except Exception as e:
                        logger.do_log(f"[RATIOS] ❌ Failed to read {fn} - {e}",
                                      MessageType.ERROR)

        out_path = os.path.join(base, f"financial_ratios_all_{year}.json")
        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
        logger.do_log(f"[RATIOS] Consolidated -> {out_path} ({len(data)} filers)", MessageType.INFO)
        return out_path

    @staticmethod
    def rank(consolidated_json: str, out_csv: str, logger) -> None:
        """Produce ranking CSV by GPA, ROA, Operating Margin, etc."""
        with open(consolidated_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for e in data:
            m = e.get("metrics", {})
            rows.append(
                {
                    "symbol": e.get("symbol"),
                    "year": e.get("year"),
                    "report_type": e.get("report_type"),
                    "gpa": m.get("gpa"),
                    "asset_turnover": m.get("asset_turnover"),
                    "operating_margin": m.get("operating_margin"),
                    "debt_equity": m.get("debt_equity"),
                    "floating": m.get("floating"),
                }
            )

        df = pd.DataFrame(rows).sort_values(
            ["gpa", "asset_turnover", "operating_margin"], ascending=False
        )
        df.to_csv(out_csv, index=False)
        logger.do_log(f"[RATIOS] Ranking -> {out_csv} ({len(df)} filers)", MessageType.INFO)

    # -------- Internals --------
    def _html_to_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)

        # -------------------------
        # Internals
        # -------------------------

    def _extract_financial_items(self, html: str) -> Dict[str, float]:
        soup = BeautifulSoup(html, "html.parser")
        items = {}

        # Revenue (using synonyms)
        items["revenue"] = self._grab_from_tables_revenue(soup, "Revenue", self.SYNONYMS["revenue"])

        #Cost of Revenue
        items["cost_of_revenue"] = self._grab_from_tables_cost_of_revenue(soup, "Cost of Revenue", self.SYNONYMS["cost_of_revenue"])

        # Gross Profit (using synonyms)
        items["gross_profit"] = self._grab_from_tables_gross_profit(soup, "Gross Profit", self.SYNONYMS["gross_profit"])

        self.logger.do_log(f"[RATIOS][DEBUG] Extracted items -> {items}", MessageType.INFO)
        return items

    def _grab_from_tables_gross_profit(self, soup, label: str, patterns: List[str]) -> float:
        """
        Try to locate Gross Profit (or synonyms) in financial tables.
        Returns the numeric value if found, otherwise None.
        """
        tables = []
        for idx, table in enumerate(soup.find_all("table")):
            txt = table.get_text(" ", strip=True)
            if re.search(r"[\d$]", txt):
                tables.append((idx, table))
            else:
                self.logger.do_log(
                    f"[RATIOS][DEBUG] Skipping table {idx} (no numbers detected)",
                    MessageType.DEBUG,
                )

        for idx, table in tables:
            self.logger.do_log(
                f"[RATIOS][DEBUG] Scanning table {idx} for {label}…",
                MessageType.DEBUG,
            )
            for pat in patterns:
                self.logger.do_log(f"[RATIOS][DEBUG] Trying pattern: {pat}", MessageType.DEBUG)
                for cell in table.find_all("td"):
                    text = cell.get_text(strip=True)
                    if re.search(pat, text, re.I):
                        val_td = cell.find_next("td")
                        if val_td:
                            raw = val_td.get_text(" ", strip=True)
                            try:
                                val = float(re.sub(r"[^\d\-.]", "", raw))
                                self.logger.do_log(
                                    f"[RATIOS][DEBUG] {label} -> {val} (pattern={pat})",
                                    MessageType.INFO,
                                )
                                return val
                            except Exception as e:
                                self.logger.do_log(
                                    f"[RATIOS][DEBUG] {label} parse fail {raw} - {e}",
                                    MessageType.WARNING,
                                )

        self.logger.do_log(f"[RATIOS][DEBUG] {label} not found", MessageType.INFO)
        return None

    def _grab_from_tables_revenue(self, soup, label: str, patterns: List[str]) -> float:
        # Collect tables that look numeric (contain digits or '$')
        tables = []
        for idx, table in enumerate(soup.find_all("table")):
            txt = table.get_text(" ", strip=True)
            if re.search(r"[\d$]", txt):
                tables.append((idx, table))
            else:
                self.logger.do_log(
                    f"[RATIOS][DEBUG] Skipping table {idx} (no numbers detected)",
                    MessageType.DEBUG,
                )

        # Iterate through tables and patterns
        for idx, table in tables:
            self.logger.do_log(
                f"[RATIOS][DEBUG] Scanning table {idx} for {label}…",
                MessageType.DEBUG,
            )
            for pat in patterns:
                self.logger.do_log(
                    f"[RATIOS][DEBUG] Trying pattern: {pat}",
                    MessageType.DEBUG,
                )
                for cell in table.find_all("td"):
                    text = cell.get_text(strip=True)
                    if re.search(pat, text, re.I):
                        siblings = cell.find_next_siblings("td")
                        # Check up to 4 following cells (since values may not be immediate)
                        for offset, sib in enumerate(siblings[:4], start=1):
                            raw = sib.get_text(" ", strip=True)
                            cleaned = re.sub(r"[^\d\-.()]", "", raw)
                            if cleaned:
                                # Handle negatives in parentheses
                                if "(" in raw and ")" in raw:
                                    cleaned = "-" + re.sub(r"[^\d.]", "", raw)
                                try:
                                    val = float(cleaned)
                                    self.logger.do_log(
                                        f"[RATIOS][DEBUG] {label} -> {val} "
                                        f"(pattern={pat}, table={idx}, raw='{raw}')",
                                        MessageType.INFO,
                                    )
                                    return val
                                except Exception as e:
                                    self.logger.do_log(
                                        f"[RATIOS][DEBUG] {label} parse fail raw='{raw}' - {e}",
                                        MessageType.WARNING,
                                    )

        # Nothing matched
        self.logger.do_log(f"[RATIOS][DEBUG] {label} not found", MessageType.INFO)
        return None

    def _grab_from_tables_cost_of_revenue(self, soup, label: str, patterns: List[str]) -> float:
        # Collect only tables that contain numeric data or '$'
        tables = []
        for idx, table in enumerate(soup.find_all("table")):
            txt = table.get_text(" ", strip=True)
            if re.search(r"[\d$]", txt):
                tables.append((idx, table))

        # Scan through filtered tables
        for idx, table in tables:
            self.logger.do_log(
                f"[RATIOS][DEBUG] Scanning table {idx} for {label}…",
                MessageType.DEBUG,
            )
            for pat in patterns:
                self.logger.do_log(f"[RATIOS][DEBUG] Trying pattern: {pat}", MessageType.DEBUG)
                for cell in table.find_all("td"):
                    text = cell.get_text(strip=True)
                    if re.search(pat, text, re.I):
                        val_td = cell.find_next("td")
                        if val_td:
                            raw = val_td.get_text(" ", strip=True)
                            try:
                                val = float(re.sub(r"[^\d\-.]", "", raw))
                                self.logger.do_log(
                                    f"[RATIOS][DEBUG] {label} -> {val} (pattern={pat})",
                                    MessageType.INFO,
                                )
                                return val
                            except Exception as e:
                                self.logger.do_log(
                                    f"[RATIOS][DEBUG] {label} parse fail {raw} - {e}",
                                    MessageType.WARNING,
                                )
        self.logger.do_log(f"[RATIOS][DEBUG] {label} not found", MessageType.INFO)
        return None

    def _compute_ratios(self, items: Dict[str, float]) -> Dict[str, float]:
        rev = items.get("revenue") or 0
        gp = items.get("gross_profit") or 0
        assets = items.get("total_assets") or 0
        ni = items.get("net_income") or 0
        debt = items.get("debt")
        eq = items.get("equity")
        floating = items.get("floating")

        ratios = {}
        ratios["gpa"] = round(gp / assets, 3) if gp and assets else None
        ratios["asset_turnover"] = round(rev / assets, 3) if rev and assets else None
        ratios["operating_margin"] = round(ni / rev, 3) if rev and ni else None
        if debt is not None and eq not in (None, 0):
            ratios["debt_equity"] = round(debt / eq, 3)
        else:
            ratios["debt_equity"] = None
        ratios["floating"] = floating

        return ratios
