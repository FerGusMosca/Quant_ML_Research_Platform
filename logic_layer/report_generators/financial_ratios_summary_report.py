import os
import re
import json
from typing import Dict, List
import pandas as pd
from lxml import etree

from framework.common.logger.message_type import MessageType


class FinancialRatiosSummaryReport:
    USGAAP_TAGS = {
        "revenue": [
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "OperatingRevenues",
            "SalesAndServiceRevenueNet",
            "InterestAndDividendIncomeOperating",
            "FinancialServicesRevenue"
        ],
        "cost_of_revenue": [
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold",
            "CostOfSales",
            "OperatingCostsAndExpenses",
            "CostOfServices",
            "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization"
        ],
        "gross_profit": [
            "GrossProfit",
            "GrossMargin",
            "GrossIncome"
        ],
        "total_assets": [
            "Assets",
            "AssetsCurrentAndNoncurrent",
            "AssetsFairValueDisclosure",
            "AssetsIncludingAssetsFromDiscontinuedOperations"
        ],
        "net_income": [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "NetIncomeLossAvailableToCommonStockholdersDiluted"
        ],
        "debt": [
            "Liabilities",
            "LongTermDebtNoncurrent",
            "LongTermDebtCurrent",
            "LongTermAndShortTermDebt",
            "DebtInstrumentCarryingAmount",
            "DebtObligations"
        ],
        "equity": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "PartnersCapital",
            "MembersEquity",
            "TemporaryEquityRedemptionValue",
            "CommonStockValue",
            "AdditionalPaidInCapital"
        ]
    }

    def __init__(self, year: int, logger, report_type: str = "K10", universe_key: str = None):
        self.report_type = report_type.upper()
        self.input_dir = os.path.join(f"./output/{self.report_type}", str(year))
        year_dir = os.path.join(f"./output/{self.report_type}/financial_ratios_report", str(year))
        self.output_dir = os.path.join(year_dir, universe_key) if universe_key else year_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.year = year
        self.logger = logger

    # -------------------
    # Public API
    # -------------------
    def run(self) -> None:
        files = [f for f in os.listdir(self.input_dir)
                 if f.lower().endswith(".xml") and "htm" in f.lower()]
        for i, file in enumerate(sorted(files), start=1):
            try:
                symbol = file.split("_")[0].upper()
                fpath = os.path.join(self.input_dir, file)
                items = self._extract_xbrl_items(fpath)
                ratios = self._compute_ratios(items)

                curated_text = (
                    f"Symbol: {symbol}. Year: {self.year}. ReportType: {self.report_type}. "
                    "This document summarizes key financial ratios derived from XBRL filings, "
                    "including revenue, net income, assets, debt, and equity. "
                    f"Metrics include GPA ({ratios.get('gpa')}), asset turnover ({ratios.get('asset_turnover')}), "
                    f"operating margin ({ratios.get('operating_margin')}), and debt-to-equity ({ratios.get('debt_equity')})."
                )

                out = {
                    "symbol": symbol,
                    "year": self.year,
                    "report_type": self.report_type,
                    "items": items,
                    "curated_text": curated_text,
                    "metrics": ratios,
                }
                out_path = os.path.join(self.output_dir, f"{symbol}_{self.year}_ratios.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

                self.logger.do_log(f"[RATIOS][{i}] ✅ {symbol} saved", MessageType.INFO)

            except Exception as e:
                self.logger.do_log(f"[RATIOS][{i}] ❌ {file} failed - {str(e)}", MessageType.ERROR)

    @staticmethod
    def consolidate_year(year: int, report_type: str, logger, universe_key: str = None) -> str:
        """Merge all *_ratios.json for a year+report_type into one JSON."""
        base = os.path.join(f"./output/{report_type.upper()}/financial_ratios_report", str(year))
        if universe_key:
            base = os.path.join(base, universe_key)

        data = []
        if os.path.isdir(base):
            for fn in os.listdir(base):
                if fn.endswith(f"_{year}_ratios.json"):
                    try:
                        with open(os.path.join(base, fn), "r", encoding="utf-8") as fh:
                            j = json.load(fh)
                        if j.get("year") == year:
                            data.append(j)
                    except Exception as e:
                        logger.do_log(f"[RATIOS] ❌ Failed {fn} - {e}", MessageType.ERROR)
        out_path = os.path.join(base, f"financial_ratios_all_{year}.json")
        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
        logger.do_log(f"[RATIOS] Consolidated -> {out_path} ({len(data)} filers)", MessageType.INFO)
        return out_path

    @staticmethod
    def rank(consolidated_json: str, out_csv: str, logger) -> None:
        """Ranking CSV con ratios calculados."""
        with open(consolidated_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for e in data:
            m = e.get("metrics", {})
            rows.append({
                "symbol": e.get("symbol"),
                "year": e.get("year"),
                "revenue": e.get("items", {}).get("revenue"),
                "gross_profit": e.get("items", {}).get("gross_profit"),
                "assets": e.get("items", {}).get("total_assets"),
                "net_income": e.get("items", {}).get("net_income"),
                "debt": e.get("items", {}).get("debt"),
                "equity": e.get("items", {}).get("equity"),
                "gpa": m.get("gpa"),
                "asset_turnover": m.get("asset_turnover"),
                "operating_margin": m.get("operating_margin"),
                "debt_equity": m.get("debt_equity"),
            })

        df = pd.DataFrame(rows).sort_values(
            ["gpa", "asset_turnover", "operating_margin"], ascending=False
        )
        df.to_csv(out_csv, index=False)
        logger.do_log(f"[RATIOS] Ranking -> {out_csv} ({len(df)} filers)", MessageType.INFO)

    # -------------------
    # Internals
    # -------------------
    def _extract_xbrl_items(self, fpath: str) -> Dict[str, float]:
        items = {}
        tree = etree.parse(fpath)
        root = tree.getroot()
        nsmap = {k: v for k, v in root.nsmap.items() if k}
        nsmap["us-gaap"] = [v for k, v in nsmap.items() if "us-gaap" in v][0]

        for key, tags in self.USGAAP_TAGS.items():
            val = None
            for tag in tags:
                els = root.findall(f".//us-gaap:{tag}", namespaces=nsmap)
                for el in els:
                    raw = (el.text or "").replace(",", "").strip()
                    if raw:
                        cleaned = raw.replace("(", "-").replace(")", "")
                        try:
                            val = float(cleaned)
                            break
                        except:
                            continue
                if val is not None: break
            items[key] = val
        return items

    def _compute_ratios(self, items: Dict[str, float]) -> Dict[str, float]:
        rev = items.get("revenue") or 0
        gp = items.get("gross_profit") or 0
        assets = items.get("total_assets") or 0
        ni = items.get("net_income") or 0
        debt = items.get("debt")
        eq = items.get("equity")

        ratios = {}
        ratios["gpa"] = round(gp / assets, 3) if gp and assets else None
        ratios["asset_turnover"] = round(rev / assets, 3) if rev and assets else None
        ratios["operating_margin"] = round(ni / rev, 3) if rev and ni else None
        ratios["debt_equity"] = round(debt / eq, 3) if debt and eq else None
        return ratios
