import json
import os
import shutil
from datetime import datetime
import asyncio

import traceback

from business_entities.tag_run import TagRun
from common.dto.mcp.bootstrap_registry import  build_mcp_registry_reports
from common.dto.mcp.dispatcher import JsonRpcDispatcher
from common.dto.mcp.progress_bus import ProgressBus
from common.dto.sec_w_file import SecurityWithFile
from common.dto.security_report_calendar import SecurityReportCalendar
from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from common.enums.report_type import ReportType
from common.enums.sec_reports import SECReports
from common.util.date_mgmt.date_range_handler import DateRangeHandler
from common.util.downloaders.K_Q_10.k8_downloader import K8Downloader
from common.util.downloaders.finviz_full_news_downloader import FinVizFullNewsDownloader
from common.util.downloaders.finviz_offline_sentiment_analyzer import FinvizOfflineSentimentAnalyzer
from common.util.downloaders.ib_income_statement import IBIncomeStatement
from common.util.downloaders.K_Q_10.k10_downloader import K10Downloader
from common.util.downloaders.K_Q_10.q10_downloader import Q10Downloader
from common.util.downloaders.thirteen_F.thirteen_F_graph_downloader import ThirteenFGraphDownloader
from common.util.downloaders.yahoo_income_statement import YahooIncomeStatement
from common.util.scrappers.securities_calendar_extractor import SecuritiesCalendarExtractor
from common.util.std_in_out.K_Q_10_file_locator import KQ10FileLocator
from common.util.std_in_out.file_locators import FileLocators
from common.util.std_in_out.root_locator import RootLocator
from data_access_layer.neo4j.graph_holding_mgr import HoldingsGraphManager
from data_access_layer.portfolio_securities_manager import PortfolioSecuritiesManager
from data_access_layer.report_securities_manager import ReportSecuritiesManager
from data_access_layer.securities_calendar_manager import SecuritiesCalendarManager
from data_access_layer.tag_run_manager import TagRunManager
from framework.common.logger.message_type import MessageType
from logic_layer.indicator_algos.financial_ratios_calcualtor import FinancialRatiosCalculator
from logic_layer.rag_corpus_metadata.tagger.transformers_single_security_topic_tagger import \
    TransformersSingleSecurityTopicTagger
from logic_layer.rag_corpus_metadata.tagger.transformers_topic_tagger import TransformersTopicTagger
from logic_layer.report_generators.K_Q_10s.K_Q_10_competition_graph import KQ10CompetitionGraph
from logic_layer.report_generators.K_Q_10s.competition_summary_report import CompetitionSummaryReport
from logic_layer.report_generators.K_Q_10s.sentiment.single_stock_sentiment_summary_report_v2 import \
    SentimentSingleSecurity
from logic_layer.report_generators.query_match_report import QueryMatchReportK10Q10
from logic_layer.report_generators.K_Q_10s.sentiment.sentence_sentiment_summary_report import SentimentSummaryReport
from logic_layer.report_generators.K_Q_10s.sentiment.sentence_sentiment_summary_report_v2 import SentimentSummaryReportV2
from logic_layer.report_generators.thirtieen_F.thirteen_f_graph_processor import ThirteenFGraphProcessor
from service_layer.client.seeking_alpha.sa_financial_client import SAFinancialsClient
from service_layer.server.mcp_server import MCPServer


class ReportsOrchestationLogic:
    def __init__(self,hist_data_conn_str,ml_reports_conn_str,mcp_server=None,mcp_port=None,p_classification_map_key=None,
                 logger=None,neo4j_config=None):

        self.logger=logger

        self.report_securities_mgr = ReportSecuritiesManager(ml_reports_conn_str, logger)

        self.portfolio_securities_mgr = PortfolioSecuritiesManager(ml_reports_conn_str,logger)

        self.sec_cal_mgr =SecuritiesCalendarManager(ml_reports_conn_str)

        self.tag_runs_mgr = TagRunManager(ml_reports_conn_str,logger)

        self.mcp_server=mcp_server
        self.mcp_port=mcp_port

        if neo4j_config is not None:
            self.neo_holding_graph_mgr=HoldingsGraphManager(neo4j_config.uri, neo4j_config.user, neo4j_config.pwd)

    '''
    def _run_financial_ratios_report(self, year, report_type="K10", universe=None):
        """
        Build financial ratios summaries from SEC filings (K10 or Q10).
        Extract balance sheet / income statement fields, compute ratios,
        and consolidate into one JSON + optional CSV.

        :param year: Filing year
        :param report_type: "K10" or "Q10"
        :param universe: optional universe key for subfolder
        """
        # Instantiate processor
        gen = FinancialRatiosSummaryReport(
            year=year,
            report_type=report_type,
            logger=self.logger,
            universe_key=universe
        )
        gen.run()

        # Consolidate (year + report_type aware)
        consolidated = FinancialRatiosSummaryReport.consolidate_year(year, report_type, self.logger,
                                                                     universe_key=universe)

        # Ranking opcional: GPA, ROA, Debt/Equity, etc. (si lo implementás igual que sentiment.rank)
        ranking_csv = os.path.join(os.path.dirname(consolidated), f"financial_ratios_ranking_{year}.csv")
        FinancialRatiosSummaryReport.rank(consolidated, ranking_csv, self.logger)

        self.logger.do_log(
            f"[RATIOS] ✅ Financial ratios summary completed ({report_type}, scope={universe or 'ALL'})",
            MessageType.INFO
        )
    '''

    def _run_download_k10(self, year, portfolio, job_id):
        """
        Download 10-K filings for a given portfolio and year range.
        Emits a FINAL structured completion event so clients can safely transition.
        """

        # Resolve year range
        years = DateRangeHandler.handle_date_range(year, self.logger)
        single_year = len(years) == 1

        # Global summary for the whole job
        summary = {
            "years": {},
            "total_securities": 0,
            "downloaded": 0,
            "skipped_exists": 0,
            "not_found": 0,
            "errors": 0,
        }

        for y in years:
            base_path = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{ReportFolder.K10.value}/{y}"

            self.logger.do_log(
                f"[REPORT] Downloading K10 to {base_path}",
                MessageType.INFO,
                job_id
            )

            # Explicit overwrite only when a single year is requested
            if single_year and os.path.exists(base_path):
                shutil.rmtree(base_path)
                self.logger.do_log(
                    f"[REPORT] Removed existing directory {base_path}",
                    MessageType.INFO,
                    job_id
                )

            os.makedirs(base_path, exist_ok=True)

            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)
            self.logger.do_log(
                f"[REPORT] Found {len(securities)} securities to process for year {y}",
                MessageType.INFO,
                job_id
            )

            # Per-year summary
            summary["years"][y] = {
                "downloaded": 0,
                "skipped_exists": 0,
                "not_found": 0,
                "errors": 0,
            }

            for i, sec in enumerate(securities):
                symbol = sec.ticker
                cik = sec.cik
                summary["total_securities"] += 1

                try:
                    result = K10Downloader.download_k10(symbol, cik, y, base_path,self.logger, job_id)

                    if result == "EXISTS":
                        summary["skipped_exists"] += 1
                        summary["years"][y]["skipped_exists"] += 1
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ⚠️ Skipped {symbol}: file already exists ({y})",
                            MessageType.INFO,
                            job_id
                        )

                    elif result == "NOT_FOUND":
                        summary["not_found"] += 1
                        summary["years"][y]["not_found"] += 1
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ❌ No 10-K available yet for {symbol} ({y})",
                            MessageType.WARNING,
                            job_id
                        )

                    else:
                        summary["downloaded"] += 1
                        summary["years"][y]["downloaded"] += 1
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded K10 for {symbol} ({y})",
                            MessageType.INFO,
                            job_id
                        )

                except Exception as e:
                    summary["errors"] += 1
                    summary["years"][y]["errors"] += 1
                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {e}",
                        MessageType.ERROR,
                        job_id
                    )

        # ---- FINAL EXPLICIT COMPLETION EVENT (CRITICAL) ----
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "download_k10",
                "portfolio": portfolio,
                "summary": summary,
            }),
            MessageType.INFO,
            job_id
        )


    def _run_download_k8(self, year, portfolio, job_id):
        """
        Download 8-K filings for a given portfolio and year range.

        Iterates over all securities in the portfolio, resolves the full date range
        for each year, and downloads available 8-K filings (market-moving events).
        Tracks per-year and global statistics (downloaded, not found, errors)
        and emits a FINAL structured completion event for safe downstream processing.
        """

        years = DateRangeHandler.handle_date_range(year, self.logger)

        summary = {
            "years": {},
            "total_securities": 0,
            "downloaded": 0,
            "not_found": 0,
            "errors": 0,
        }

        for y in years:
            base_path = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{ReportFolder.K8.value}/{y}"

            self.logger.do_log(
                f"[REPORT] Downloading K8 to {base_path}",
                MessageType.INFO,
                job_id
            )

            os.makedirs(base_path, exist_ok=True)

            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

            self.logger.do_log(
                f"[REPORT] Found {len(securities)} securities to process for year {y}",
                MessageType.INFO,
                job_id
            )

            summary["years"][y] = {
                "downloaded": 0,
                "not_found": 0,
                "errors": 0,
            }

            for i, sec in enumerate(securities):
                symbol = sec.ticker
                cik = sec.cik
                summary["total_securities"] += 1

                try:


                    k8_downloader = K8Downloader(self.logger)
                    files = k8_downloader.download_k8_range(
                        symbol, cik, y, base_path, job_id
                    )

                    if not files:
                        summary["not_found"] += 1
                        summary["years"][y]["not_found"] += 1
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ⚠️ No 8-K found for {symbol} ({y})",
                            MessageType.WARNING,
                            job_id
                        )
                    else:
                        summary["downloaded"] += len(files)
                        summary["years"][y]["downloaded"] += len(files)
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded {len(files)} K8 files for {symbol} ({y})",
                            MessageType.INFO,
                            job_id
                        )

                except Exception as e:
                    summary["errors"] += 1
                    summary["years"][y]["errors"] += 1
                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {e}",
                        MessageType.ERROR,
                        job_id
                    )

        # ---- FINAL EXPLICIT COMPLETION EVENT (CRITICAL) ----
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "download_k8",
                "portfolio": portfolio,
                "summary": summary,
            }),
            MessageType.INFO,
            job_id
        )

    def _run_download_q10(self, year, portfolio, job_id=None):
        """
        Download 10-Q filings for a given portfolio and year range.
        Emits a FINAL structured completion event so clients can safely transition.
        """

        # ---------------------------------------------------------
        # 🧠 Resolve year range
        # ---------------------------------------------------------
        years = DateRangeHandler.handle_date_range(year, self.logger)

        # ---------------------------------------------------------
        # 📊 Global summary
        # ---------------------------------------------------------
        summary = {
            "years": {},
            "total_securities": 0,
            "downloaded": 0,
            "skipped_exists": 0,
            "not_found": 0,
            "errors": 0,
        }

        # ---------------------------------------------------------
        # 🚀 Process each year
        # ---------------------------------------------------------
        for y in years:
            base_path = (
                f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/"
                f"{portfolio}/{ReportFolder.Q10.value}/{y}"
            )

            self.logger.do_log(
                f"[REPORT] Downloading Q10 to {base_path}",
                MessageType.INFO,
                job_id
            )

            os.makedirs(base_path, exist_ok=True)

            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)
            self.logger.do_log(
                f"[REPORT] Found {len(securities)} securities to process for year {y}",
                MessageType.INFO,
                job_id
            )

            # Per-year summary
            summary["years"][y] = {
                "downloaded": 0,
                "skipped_exists": 0,
                "not_found": 0,
                "errors": 0,
            }

            for i, sec in enumerate(securities):
                symbol = sec.ticker
                cik = sec.cik
                summary["total_securities"] += 1

                try:
                    result = Q10Downloader.download_q10s(symbol, cik, y, base_path,self.logger,job_id)

                    if result == "EXISTS":
                        summary["skipped_exists"] += 1
                        summary["years"][y]["skipped_exists"] += 1

                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ⚠️ Skipped {symbol}: files already exist ({y})",
                            MessageType.INFO,
                            job_id
                        )

                    elif result == "NOT_FOUND":
                        summary["not_found"] += 1
                        summary["years"][y]["not_found"] += 1

                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ❌ No 10-Q available yet for {symbol} ({y})",
                            MessageType.WARNING,
                            job_id
                        )

                    elif result == "FOUND":
                        summary["downloaded"] += 1
                        summary["years"][y]["downloaded"] += 1

                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded Q10(s) for {symbol} ({y})",
                            MessageType.INFO,
                            job_id
                        )

                    else:
                        raise Exception(f"Unknown result '{result}' returned by Q10Downloader")

                except Exception as e:
                    summary["errors"] += 1
                    summary["years"][y]["errors"] += 1

                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}] 💥 Failed for {symbol}: {e}",
                        MessageType.ERROR,
                        job_id
                    )

        # ---------------------------------------------------------
        # 🧾 FINAL EXPLICIT COMPLETION EVENT (CRITICAL)
        # ---------------------------------------------------------
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "download_q10",
                "portfolio": portfolio,
                "summary": summary,
            }),
            MessageType.INFO,
            job_id
        )

    def _get_universe_filers(self, universe_key: str):
        if not universe_key:
            return None
        dtos = self.report_securities_mgr.get_report_securities(universe_key)
        return sorted({(d.ticker or "").upper() for d in dtos if d.ticker})

    def run_query_match_report_KQ_10(self, year, report_type=ReportFolder.K10.value,
                               portfolio=None, dest_folder=None,query=None):

        years = DateRangeHandler.handle_date_range(year, self.logger)
        securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

        all_matches = []

        for y in years:
            self.logger.do_log(
                f"[QUERY-MATCH] 🚀 Starting query match scan | report={report_type} | year={y}",
                MessageType.INFO
            )

            for i, sec in enumerate(securities):
                self.logger.do_log(
                    f"[QUERY-MATCH] 🔎 Processing {sec.symbol} ({i + 1}/{len(securities)})",
                    MessageType.DEBUG
                )

                qm_rep = QueryMatchReportK10Q10(
                    logger=self.logger,
                    portfolio=portfolio,
                    report_type=report_type,
                    dest_folder=dest_folder
                )

                try:
                    # Core logic:
                    # 1) Stream documents
                    # 2) Bi-encoder filter (cheap)
                    # 3) Cross-encoder filter (expensive, only survivors)
                    matches = qm_rep.run_analysis(
                        symbol=sec.symbol,
                        query=query,
                        year=y,
                        report_type=report_type
                    )

                    if matches:
                        all_matches.extend(matches)

                except Exception as e:
                    self.logger.do_log(
                        f"[QUERY-MATCH][ERROR] {sec.symbol} {y}: {e}",
                        MessageType.ERROR
                    )

        # Optional persistence
        '''
        if dest_folder:
            QueryMatchReportWriter.write(
                matches=all_matches,
                dest_folder=dest_folder,
                logger=self.logger
            )
        '''

        return all_matches

    def _run_sentiment_summary_report(
            self,
            year,
            report_type=ReportFolder.K10.value,
            portfolio=None,
            universe=None,
            dest_folder=None,
            rank_folder=None,
            job_id=None,
    ):
        """
        Build sentiment summaries focused on management guidance/opinion.
        Emits a FINAL structured completion event for MCP clients.
        """

        # ---------------------------------------------------------
        # 🧠 Resolve year range
        # ---------------------------------------------------------
        years = DateRangeHandler.handle_date_range(year, self.logger)

        # ---------------------------------------------------------
        # 📊 Global summary
        # ---------------------------------------------------------
        summary = {
            "report": "sentiment_summary",
            "report_type": report_type,
            "portfolio": portfolio,
            "universe": universe,
            "years": {},
            "processed_years": 0,
            "successful_years": 0,
            "failed_years": 0,
        }

        self.logger.do_log(
            f"[SENT] 🚀 Starting sentiment summary report years={years}, type={report_type}",
            MessageType.INFO,
            job_id
        )

        whitelist = self._get_universe_filers(universe) if universe else None

        # ---------------------------------------------------------
        # 🚀 Process each year
        # ---------------------------------------------------------
        for y in years:
            start_time = datetime.now()
            summary["processed_years"] += 1

            summary["years"][y] = {
                "status": "started",
                "consolidated": False,
                "ranking": False,
                "error": None,
                "elapsed_sec": None,
            }

            self.logger.do_log(
                f"[SENT] ▶️ Processing year {y}",
                MessageType.INFO,
                job_id
            )

            try:
                gen = SentimentSummaryReportV2(
                    year=y,
                    report_type=report_type,
                    logger=self.logger,
                    portfolio=portfolio,
                    filers_whitelist=whitelist,
                    universe_key=universe,
                    dest_folder=dest_folder,
                    rank_folder=rank_folder,
                )

                gen.run()

                # -------------------------------------------------
                # Consolidation + ranking
                # -------------------------------------------------
                if report_type == ReportFolder.K10.value:
                    consolidated = gen.consolidate_year(y, report_type, job_id)
                    ranking_csv = os.path.join(
                        os.path.dirname(consolidated),
                        f"sentiment_summary_ranking_{y}.csv"
                    )
                    SentimentSummaryReport.rank(consolidated, ranking_csv, self.logger)
                else:
                    for quarter in [1, 2, 3]:
                        consolidated = gen.consolidate_year(y, report_type, quarter)
                        ranking_csv = os.path.join(
                            os.path.dirname(consolidated),
                            f"sentiment_summary_ranking_{y}.csv"
                        )
                        SentimentSummaryReport.rank(consolidated, ranking_csv, self.logger)

                summary["years"][y]["status"] = "completed"
                summary["years"][y]["consolidated"] = True
                summary["years"][y]["ranking"] = True
                summary["successful_years"] += 1

                elapsed = (datetime.now() - start_time).total_seconds()
                summary["years"][y]["elapsed_sec"] = round(elapsed, 2)

                self.logger.do_log(
                    f"[SENT] ✅ Year {y} completed in {elapsed:.1f}s",
                    MessageType.INFO,
                    job_id
                )

            except Exception as e:
                summary["years"][y]["status"] = "failed"
                summary["years"][y]["error"] = str(e)
                summary["failed_years"] += 1

                self.logger.do_log(
                    f"[SENT] ❌ Year {y} failed: {e}",
                    MessageType.ERROR,
                    job_id
                )

        # ---------------------------------------------------------
        # 🧾 FINAL COMPLETION EVENT (CRITICAL)
        # ---------------------------------------------------------
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "sentiment_summary",
                "summary": summary,
            }),
            MessageType.INFO,
            job_id
        )

    #
    def _run_document_single_security(
            self,
            symbol: str,
            source: str,
            year: int,
            quarter: int = None,
            tag_cfg=None,
            job_id: str = None,
    ) -> dict:
        """
        Run topic analysis for a single security.
        Returns JSON result suitable for MCP/API responses.
        """

        start_time = datetime.now()
        symbol = symbol.upper().strip()

        # ---------------------------------------------------------
        # 📊 Initialize result structure
        # ---------------------------------------------------------
        result = {
            "report": "topic_single_security",
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "source": source,
            "input_file": None,
            "tag_source": None,
            "status": "started",
            "analysis": None,
            "error_type": None,
            "message": None,
            "elapsed_sec": None,
        }

        self.logger.do_log(
            f"[TOPIC-SINGLE] 🚀 Starting analysis: {symbol} {year}" +
            (f" Q{quarter}" if quarter else ""),
            MessageType.INFO,
            job_id
        )

        try:
            # ---------------------------------------------------------
            # 🔍 Resolve input file
            # ---------------------------------------------------------
            report_suffix = "10-K" if quarter is None else f"{quarter}_10-Q"

            file_path = os.path.join(
                RootLocator.get_root(),
                Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                source,
                str(year),
                f"{symbol}_{year}_{report_suffix}.html",
            )

            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Input filing not found: {file_path}")

            result["input_file"] = os.path.basename(file_path)

            # ---------------------------------------------------------
            # 🎯 Run analysis
            # ---------------------------------------------------------
            tagger = TransformersSingleSecurityTopicTagger(
                logger=self.logger,
                tag_cfg=tag_cfg,
            )

            tag_dict = tagger.initialize_tag_dict(job_id=job_id)
            tagger.tag_dict = tag_dict

            result["tag_source"] = (
                tag_cfg.tag_file
                if tag_cfg and getattr(tag_cfg, "tag_file", None)
                else "inline/json"
            )

            analysis_result = tagger.analyze(
                security_symbol=symbol,
                file_path=file_path,
                job_id=job_id,
            )

            # ---------------------------------------------------------
            # 📦 Process result
            # ---------------------------------------------------------
            if not analysis_result or not analysis_result.get("topics"):
                result["status"] = "completed"
                result["analysis"] = {
                    "security": symbol,
                    "file": result["input_file"],
                    "topics": {},
                    "summary": "No topics showed sufficient semantic alignment with the provided tag phrases",
                }
                result["message"] = "Analysis completed (no matching topics)"

            else:
                result["status"] = "completed"
                result["analysis"] = analysis_result
                result["message"] = "Analysis completed successfully"

            self.logger.do_log(
                f"[TOPIC-SINGLE] ✅ {symbol} completed | topics="
                f"{len(result['analysis'].get('topics', {}))}",
                MessageType.INFO,
                job_id
            )

        except FileNotFoundError as e:
            result["status"] = "failed"
            result["error_type"] = "file_not_found"
            result["message"] = str(e)

            self.logger.do_log(
                f"[TOPIC-SINGLE] ❌ Filing not found for {symbol}: {str(e)}",
                MessageType.ERROR,
                job_id
            )

        except Exception as e:
            result["status"] = "failed"
            result["error_type"] = "internal_error"
            result["message"] = f"Internal error: {str(e)}"

            self._log_exc(
                "[TAGGING SINGLE SECURITY] ❌ execution failed",
                e,
                job_id,
            )

        # ---------------------------------------------------------
        # ⏱️ Calculate elapsed time
        # ---------------------------------------------------------
        elapsed = (datetime.now() - start_time).total_seconds()
        result["elapsed_sec"] = round(elapsed, 2)

        self.logger.do_log(
            f"[TOPIC-SINGLE] 🏁 {symbol} finished in {elapsed:.1f}s | Status: {result['status']}",
            MessageType.INFO,
            job_id
        )

        # ---------------------------------------------------------
        # 🧾 FINAL COMPLETION EVENT (for MCP)
        # ---------------------------------------------------------
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "topic_single_security",
                "result": result,
            }),
            MessageType.INFO,
            job_id
        )

        return result

    def _run_sentiment_single_security_report(
            self,
            symbol: str,
            year: int,
            report_type: str = ReportFolder.K10.value,
            portfolio: str = None,
            quarter: int = None,
            job_id: str = None,
    ) -> dict:
        """
        Run sentiment analysis for a single security.
        Returns JSON result suitable for MCP/API responses.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            year: Filing year
            report_type: '10K' or '10Q'
            portfolio: Portfolio name (e.g., 'US_BIGCAP')
            quarter: Required if report_type is '10Q' (1-4)
            job_id: Optional job identifier for logging

        Returns:
            dict: {
                "status": "success" | "error",
                "symbol": str,
                "year": int,
                "report_type": str,
                "analysis": {...} | None,
                "error_type": str | None,
                "message": str | None
            }
        """

        start_time = datetime.now()
        symbol = symbol.upper().strip()

        # ---------------------------------------------------------
        # 📊 Initialize result structure
        # ---------------------------------------------------------
        result = {
            "report": "sentiment_single_security",
            "symbol": symbol,
            "year": year,
            "report_type": report_type,
            "quarter": quarter,
            "portfolio": portfolio,
            "status": "started",
            "analysis": None,
            "error_type": None,
            "message": None,
            "elapsed_sec": None,
        }

        self.logger.do_log(
            f"[SENT-SINGLE] 🚀 Starting analysis: {symbol} {report_type} {year}" +
            (f" Q{quarter}" if quarter else ""),
            MessageType.INFO,
            job_id
        )

        try:
            # ---------------------------------------------------------
            # 🔍 Validate parameters
            # ---------------------------------------------------------
            if not portfolio:
                raise ValueError("Portfolio parameter is required")

            if report_type not in [ReportFolder.K10.value, ReportFolder.Q10.value]:
                raise ValueError(f"Invalid report_type: {report_type}. Must be '10K' or '10Q'")

            if report_type == ReportFolder.Q10.value and not quarter:
                raise ValueError("Quarter (1-4) is required for 10Q reports")

            if quarter and (int(quarter) < 1 or int(quarter) > 4):
                raise ValueError(f"Invalid quarter: {quarter}. Must be 1-4")

            # ---------------------------------------------------------
            # 🎯 Run analysis
            # ---------------------------------------------------------
            analyzer = SentimentSingleSecurity(self.logger)

            analysis_result = analyzer.analyze(
                symbol=symbol,
                report_type=report_type,
                year=year,
                portfolio=portfolio,
                quarter=quarter,
                job_id=job_id
            )

            # ---------------------------------------------------------
            # 📦 Process result
            # ---------------------------------------------------------
            if analysis_result.get("status") == "success":
                result["status"] = "completed"
                result["analysis"] = analysis_result.get("analysis", {})
                result["message"] = "Analysis completed successfully"

                metrics = analysis_result.get("analysis", {}).get("metrics", {})
                sentiment = metrics.get("mdna_sentiment", 0)

                self.logger.do_log(
                    f"[SENT-SINGLE] ✅ {symbol} completed | Sentiment={sentiment:.3f}",
                    MessageType.INFO,
                    job_id
                )

            else:
                # Analysis returned error
                result["status"] = "failed"
                result["error_type"] = analysis_result.get("error_type", "unknown_error")
                result["message"] = analysis_result.get("message", "Analysis failed")

                self.logger.do_log(
                    f"[SENT-SINGLE] ❌ {symbol} failed: {result['message']}",
                    MessageType.ERROR,
                    job_id
                )

        except ValueError as e:
            # Validation errors
            result["status"] = "failed"
            result["error_type"] = "validation_error"
            result["message"] = str(e)

            self.logger.do_log(
                f"[SENT-SINGLE] ❌ Validation error for {symbol}: {str(e)}",
                MessageType.ERROR,
                job_id
            )

        except FileNotFoundError as e:
            # Filing not found
            result["status"] = "failed"
            result["error_type"] = "file_not_found"
            result["message"] = str(e)

            self.logger.do_log(
                f"[SENT-SINGLE] ❌ Filing not found for {symbol}: {str(e)}",
                MessageType.ERROR,
                job_id
            )

        except Exception as e:
            # Unexpected errors
            result["status"] = "failed"
            result["error_type"] = "internal_error"
            result["message"] = f"Internal error: {str(e)}"

            self.logger.do_log(
                f"[SENT-SINGLE] 💥 Unexpected error for {symbol}: {str(e)}",
                MessageType.ERROR,
                job_id
            )

        # ---------------------------------------------------------
        # ⏱️ Calculate elapsed time
        # ---------------------------------------------------------
        elapsed = (datetime.now() - start_time).total_seconds()
        result["elapsed_sec"] = round(elapsed, 2)

        self.logger.do_log(
            f"[SENT-SINGLE] 🏁 {symbol} finished in {elapsed:.1f}s | Status: {result['status']}",
            MessageType.INFO,
            job_id
        )

        # ---------------------------------------------------------
        # 🧾 FINAL COMPLETION EVENT (for MCP)
        # ---------------------------------------------------------
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "sentiment_single_security",
                "result": result,
            }),
            MessageType.INFO,
            job_id
        )

        return result

    def _run_competition_summary_report(self, year, report_type=ReportFolder.K10.value,
                                        portfolio=None, universe=None,
                                        dest_folder=None, rank_folder=None):
        """
        Build competition summaries across a range of years or a single year.
        """
        # Parse range
        years=DateRangeHandler.handle_date_range(year,self.logger)

        for y in years:
            start_time = datetime.now()
            self.logger.do_log(f"[COMP] 🚀 Starting competition summary ({report_type}, year={y})", MessageType.INFO)

            try:
                gen = CompetitionSummaryReport(
                    year=y,
                    report_type=report_type,
                    logger=self.logger,
                    portfolio=portfolio,
                    dest_folder=dest_folder,
                    rank_folder=rank_folder
                )
                gen.run()
            except Exception as e:
                self.logger.do_log(f"[COMP] ❌ Error during run() for year {y}: {e}", MessageType.ERROR)
                continue

            try:
                consolidated = CompetitionSummaryReport.consolidate_year(
                    y, report_type, portfolio, self.logger,
                    dest_folder=dest_folder, rank_folder=rank_folder
                )
                ranking_csv = os.path.join(os.path.dirname(consolidated),
                                           f"competition_summary_ranking_{y}.csv")
                CompetitionSummaryReport.rank(consolidated, ranking_csv, self.logger)
            except Exception as e:
                self.logger.do_log(f"[COMP] ⚠️ Consolidation/Ranking failed for {y}: {e}", MessageType.WARNING)
                continue

            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.do_log(
                f"[COMP] ✅ Competition summary completed ({report_type}, year={y}) in {elapsed:.1f}s",
                MessageType.INFO
            )



    def _log_exc(self,prefix, e,job_id):
        tb = traceback.extract_tb(e.__traceback__)[-1]
        self.logger.do_log(
            f"{prefix} | {e.__class__.__name__}: {e} | line={tb.lineno} | file={tb.filename}",
            MessageType.ERROR,
            job_id
        )

    def _file_finder(self,securities,y,quarter, root_folder,source,rank_folder,tag_cfg,job_id):
        try:  # 2- Find files based on report types
            self.logger.do_log(
                f"[TAGGING] 🚀 Starting (source={source}, rank_folder={rank_folder}, year={y}) quarter?={quarter}",
                MessageType.INFO,
                job_id
            )

            file_folder = os.path.join(
                RootLocator.get_root(),
                #Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                root_folder,
                source,
                str(y))

            prev_matched_files = FileLocators.enumerate_all_files(
                file_folder,
                self.logger,
                filters=[s.symbol.lower() for s in securities if getattr(s, "symbol", None)],
                job_id=job_id
            )

            matched_files = []
            for file in prev_matched_files:
                for sec in securities:
                    if tag_cfg is not None:
                        if ( tag_cfg.evaluate_file_for_report(sec.symbol, source, file, str(y), quarter)):
                            matched_files.append(SecurityWithFile(sec, file))
                    elif SECReports.K10.value in source or SECReports.Q10.value in source: #se assume is K10
                        if(  KQ10FileLocator.find_file(source,os.path.basename(file),sec.symbol,str(y),quarter)):
                            matched_files.append(SecurityWithFile(sec, file))
                    else:
                        raise Exception(f"Missing tag_cfg or valid source matching file {file} with security {sec.symbol}")

            return  matched_files
        except Exception as e:
            self._log_exc(f"[TAGGING] ❌ file enumeration failed | year={y}", e, job_id)
            return []

    def _create_rank_folder(self,y,tag_dict,root_folder,rank_folder,tag_run,job_id):
        try:  # 3- Crate Rank Folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            tag = "_".join(tag_dict.keys())

            dest_rank_folder = os.path.join(str(ReportType.DOCUMENT_TAGGING_RANKING.value).upper(),
                                            rank_folder,
                                            f"file_taging_{tag}_rank_{timestamp}",
                                            str(y))
            rank_dir = os.path.join(
                RootLocator.get_root(),
                #Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                root_folder,
                dest_rank_folder
            )

            tag_run.rank_folder = dest_rank_folder
            self.tag_runs_mgr.persist_tag_run(tag_run)

            self.logger.do_log(f"[TAGGING] Creating Dest Rank Dir  rank_dir={rank_dir}", MessageType.INFO, job_id)

            return rank_dir

        except Exception as e:
            self._log_exc(f"[TAGGING] ❌ rank dir build failed | year={y}", e, job_id)
            raise e

    def _create_competition_folder(self, y, root_folder, dest_graph_folder, job_id):
        try:  # 3- Crate Rank Folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")


            dest_comp_folder = os.path.join(str(ReportType.COMPETITION_GRAPH.value).upper(),
                                            dest_graph_folder,
                                            f"file_competition_graph_{timestamp}",
                                            str(y))
            comp_dir = os.path.join(
                RootLocator.get_root(),
                root_folder,
                dest_comp_folder
            )

            self.logger.do_log(f"[COMP_GRAPH] Creating Dest Comp. Graph Dir  comp_dir={comp_dir}", MessageType.INFO, job_id)

            return comp_dir

        except Exception as e:
            self._log_exc(f"[COMP_GRAPH] ❌ Comp. Graph dir build failed | year={y}", e, job_id)
            raise e

    def _run_document_tagging(self, portfolio, year,quarter, source, rank_folder, tag_cfg, job_id):

        #1- Extract All Input Data
        try:
            tagger = TransformersTopicTagger(self.logger, tag_cfg)
            years = DateRangeHandler.handle_date_range(year, self.logger)
            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)
            tag_dict = tagger.initialize_tag_dict(job_id=job_id)
        except Exception as e:
            self._log_exc("[TAGGING] ❌ init failed", e,job_id)
            #Special Error inicailization
            tag_run = TagRun.initialize_tag_run(portfolio=portfolio,
                                                report_type=ReportType.DOCUMENT_TAGGING_RANKING.value,
                                                source=source, rank_folder=rank_folder, year=year,
                                                quarter=quarter,sec_processed=0,
                                                tag_cfg=tag_cfg,
                                                tag_dict=None)
            tag_run.set_error(str(e))
            self.tag_runs_mgr.persist_tag_run(tag_run)

            return

        try:
            found_files=False
            for y in years:

                tag_run = TagRun.initialize_tag_run(portfolio=portfolio,
                                                    report_type=ReportType.DOCUMENT_TAGGING_RANKING.value,
                                                    source=source, rank_folder=rank_folder, year=str(y),
                                                    quarter=quarter,sec_processed=len(securities) , tag_cfg=tag_cfg,
                                                    tag_dict=tag_dict)

                self.tag_runs_mgr.persist_tag_run(tag_run)

                try:
                    start_time = datetime.now()

                    # 2- Find files based on report types
                    matched_files=self._file_finder(securities,y,quarter,Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                                                    source,rank_folder,tag_cfg,job_id)

                    if len(matched_files)==0:
                        tag_run.set_skipped(f"No files found for portfolio {portfolio} and year(s) {years}")
                        self.tag_runs_mgr.persist_tag_run(tag_run)
                        continue
                    else:
                        found_files=True
                    # 3- Crate Rank Folder
                    rank_dir= self._create_rank_folder(y,tag_dict,Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                                                       rank_folder,tag_run,job_id)

                    try: #4- Run Rank
                        ranking_dict = tagger.rank(
                            securities,
                            matched_files,
                            rank_dir,
                            tag_dict,
                            job_id
                        )

                        self.logger.do_log(
                            f"[TAGGING] ✔ Persisted {len(ranking_dict)} rows | dir={rank_dir}",
                            MessageType.INFO,
                            job_id
                        )

                    except Exception as e:
                        self._log_exc(f"[TAGGING] ❌ ranking failed | year={y}", e,job_id)
                        raise e

                    elapsed = (datetime.now() - start_time).total_seconds()
                    tag_run.set_finished()
                    self.tag_runs_mgr.persist_tag_run(tag_run)
                    self.logger.do_log(
                        f"[TAGGING] 🏁 Completed year={y} in {elapsed:.2f}s",
                        MessageType.INFO,
                        job_id
                    )
                except Exception as e:
                    self.logger.do_log(
                        f"[TAGGING] ❌ ERROR processing year={y} :{str(e)}",MessageType.INFO,job_id
                    )
                    tag_run.set_error(str(e))
                    self.tag_runs_mgr.persist_tag_run(tag_run)

            if not found_files:
                self.logger.do_log(
                    f"[TAGGING] ⚠️ Not a single file found for portfolio {portfolio} on years year={years}",
                    MessageType.INFO,
                    job_id
                )

        except Exception as e:
            self._log_exc(f"[TAGGING] ❌ CRITICAL error running document tagging={str(e)}", e, job_id)
            #tag_run.set_error(str(e))
            #self.tag_runs_mgr.persist_tag_run(tag_run)

    def _persist_file_graph(self, graph_dir, output_file, edges):
        os.makedirs(graph_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for edge in edges:
                line = {
                    "src": edge["src"],
                    "dst": edge["dst"],
                    "type": edge["relation"],
                    "weight": edge["score"],
                    "metadata": {
                        "file": edge["file"],
                        "block_id": edge["block_id"]
                    }
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _persist_store_graph(self, output_file: str,year:int,quarter:str,job_id:str):
        batch = []
        total = 0

        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                batch.append({
                    "manager": obj["src"].replace("manager::", ""),
                    "cusip": obj["dst"].replace("asset::", ""),
                    "asset_name": obj["dst"].replace("asset::", ""),
                    "weight": obj.get("weight", 0),
                    "file": obj.get("metadata", {}).get("file"),
                })

                if len(batch) >= self.neo_holding_graph_mgr.batch_size:
                    self.neo_holding_graph_mgr.persist(batch,year,quarter)
                    total += len(batch)
                    self.logger.do_log(
                        f"[COMP_GRAPH] Inserted {total} rows",
                        MessageType.INFO,
                        job_id)
                    batch.clear()

            if batch:
                self.neo_holding_graph_mgr.persist(batch,year,quarter)
                total += len(batch)

        self.logger.do_log(
            f"[COMP_GRAPH] Done. Total rows: {total}",
            MessageType.INFO,
            job_id
        )

    def _run_competition_graph(self, portfolio, year,quarter, source, graph_folder, job_id):

        #1- Extract All Input Data
        try:
            years = DateRangeHandler.handle_date_range(year, self.logger)
            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)
            comp_grag_ctor = KQ10CompetitionGraph(self.logger)
        except Exception as e:
            self._log_exc("[COMP_GRAPH] ❌ init failed", e,job_id)
            #Special Error inicailization
            return

        try:

            for y in years:

                try:
                    start_time = datetime.now()

                    # 2- Find files based on report types
                    matched_files=self._file_finder(securities,y,quarter,Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                                                    source,graph_folder,None,job_id)

                    if len(matched_files)==0:
                        continue

                    # 3- Crate Rank Folder
                    graph_dir= self._create_competition_folder(y,Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
                                                              graph_folder,job_id)

                    graph_file=os.path.join(graph_dir,"graph.jsonl")

                    try: #4- Run Graph
                        #Initialize an run compeition graph --> graph_dir
                        for matched_file in matched_files:
                            comp_grag_ctor.extract_competition(matched_file,job_id)

                        self._persist_file_graph(graph_dir, graph_file, comp_grag_ctor.edges)

                        self.logger.do_log(
                            f"[COMP_GRAPH] ✔ Successfully created ",
                            MessageType.INFO,
                            job_id
                        )

                    except Exception as e:
                        self._log_exc(f"[COMP_GRAPH] ❌ ranking failed | year={y}", e,job_id)
                        raise e

                    elapsed = (datetime.now() - start_time).total_seconds()

                    self.logger.do_log(
                        f"[COMP_GRAPH] 🏁 Completed year={y} in {elapsed:.2f}s",
                        MessageType.INFO,
                        job_id
                    )
                except Exception as e:
                    self.logger.do_log(
                        f"[COMP_GRAPH] ❌ ERROR processing year={y} :{str(e)}",MessageType.INFO,job_id
                    )


        except Exception as e:
            self._log_exc(f"[COMP_GRAPH] ❌ CRITICAL error running competition graph={str(e)}", e, job_id)

    def _run_download_securities_calendar(self, year, portfolio, job_id):
        """
        Download and persist SEC filing calendars (K10 / Q10 dates)
        for all securities in a portfolio and year range.

        Emits a FINAL structured completion event so clients can safely transition.
        """

        # ---------------------------------------------------------
        # 🧠 Resolve year range
        # ---------------------------------------------------------
        years = DateRangeHandler.handle_date_range(year, self.logger)

        securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

        root_dir = RootLocator.get_root(markers=["bias_mgmt_console.py", "README.md"])

        # ---------------------------------------------------------
        # 📊 Global summary
        # ---------------------------------------------------------
        summary = {
            "years": {},
            "total_securities": len(securities),
            "processed": 0,
            "saved": 0,
            "errors": 0,
        }

        self.logger.do_log(
            f"[REPORT] Starting SEC calendar download for portfolio={portfolio}, years={years}",
            MessageType.INFO,
            job_id
        )


        calendars= self.sec_cal_mgr.get_calendars_by_range(years[0],years[-1])

        # ---------------------------------------------------------
        # 🚀 Process each year
        # ---------------------------------------------------------
        for y in years:
            summary["years"][y] = {
                "processed": 0,
                "saved": 0,
                "errors": 0,
            }

            for i, sec in enumerate(securities):
                summary["processed"] += 1
                summary["years"][y]["processed"] += 1

                try:

                    if (sec.symbol,y) in calendars:
                        self.logger.do_log(
                            f"[SKIP] ◻️ Skipping download for  {sec.ticker} on year {y} because it already exists",
                            MessageType.INFO,
                            job_id
                        )
                        continue

                    # -------------------------------------------------
                    # Locate downloaded reports
                    # -------------------------------------------------
                    k10_dir = (
                            root_dir
                            / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                            / portfolio
                            / "K10"
                            / str(y)
                    )

                    q10_dir = (
                            root_dir
                            / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                            / portfolio
                            / "Q10"
                            / str(y)
                    )

                    # -------------------------------------------------
                    # Extract filing dates from real files
                    # -------------------------------------------------
                    k10_filing_date, q10_filing_dates = (
                        SecuritiesCalendarExtractor.process_k10_q10_directories(
                            sec.ticker,
                            k10_dir,
                            q10_dir
                        )
                    )

                    # -------------------------------------------------
                    # Build calendar entry
                    # -------------------------------------------------
                    entry = SecurityReportCalendar(
                        cik=sec.cik,
                        symbol=sec.ticker,
                        fiscal_year=y,
                        q1=q10_filing_dates.get(1),
                        q2=q10_filing_dates.get(2),
                        q3=q10_filing_dates.get(3),
                        k10=k10_filing_date,
                    )

                    self.sec_cal_mgr.upsert_calendar_entry(entry)

                    summary["saved"] += 1
                    summary["years"][y]["saved"] += 1

                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}][{y}] ✅ Calendar saved for {sec.ticker}",
                        MessageType.INFO,
                        job_id
                    )

                except Exception as e:
                    summary["errors"] += 1
                    summary["years"][y]["errors"] += 1

                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}][{y}] ❌ {sec.ticker} failed: {e}",
                        MessageType.ERROR,
                        job_id
                    )

        # ---------------------------------------------------------
        # 🧾 FINAL COMPLETION EVENT (CRITICAL)
        # ---------------------------------------------------------
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "download_securities_calendar",
                "portfolio": portfolio,
                "summary": summary,
            }),
            MessageType.INFO,
            job_id
        )

    def financial_ratios_report_SA(self, symbol=None, job_id=None):
        """
        Generate a simple financial ratios report using Seeking Alpha data.
        This method is fully guarded to prevent uncontrolled exceptions.
        """

        if not symbol:
            self.logger.do_log(
                "[REPORT][SA] Missing symbol parameter",
                MessageType.WARNING,
                job_id
            )
            return

        self.logger.do_log(
            f"[REPORT][SA] Starting financial ratios report | symbol={symbol}",
            MessageType.INFO,
            job_id
        )

        try:
            data = SAFinancialsClient.fetch_fundamentals(
                symbol=symbol,
                logger=self.logger,
                job_id=job_id
            )

            if not data:
                self.logger.do_log(
                    f"[REPORT][SA] No fundamentals data returned | symbol={symbol}",
                    MessageType.WARNING,
                    job_id
                )
                return

            ratios = FinancialRatiosCalculator.compute(
                data=data,
                logger=self.logger,
                job_id=job_id
            )

            self.logger.do_log(
                json.dumps({
                    "event": "completed",
                    "report": "financial_ratios_sa",
                    "symbol": symbol,
                    "ratios": ratios
                }),
                MessageType.INFO,
                job_id
            )

        except Exception as e:
            # Absolute safety net: no exception escapes the report runner
            self.logger.do_log(
                f"[REPORT][SA] Unhandled error | symbol={symbol} | error={e}",
                MessageType.ERROR,
                job_id
            )

    def _run_fin_viz_news_downloader(self,portfolio,symbol=None,job_id=None):

        if portfolio!="SINGLE_STOCKS":
            # ✅ Get securities from portfolio
            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

            self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process", MessageType.INFO)

            for i, sec in enumerate(securities):
                symbol = sec.ticker
                try:
                    out_file = FinVizFullNewsDownloader.download(symbol,portfolio,logger=self.logger,job_id=job_id)

                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded news for {symbol} -> {out_file}",
                        MessageType.INFO
                    )
                except Exception as e:
                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {str(e)}",
                        MessageType.ERROR,job_id
                    )
        else:
            try:
                FinVizFullNewsDownloader.download(symbol, portfolio,logger= self.logger,job_id=job_id)
                pass


            except Exception as e:
                self.logger.do_log(f"[REPORT] ❌ Failed for {symbol}: {str(e)}",MessageType.ERROR,job_id)


    def _run_process_finviz_news(self, portfolio, symbol=None, d_from=None):
        """
        Entry point for Finviz sentiment analysis.
        Delegates processing to FinvizOfflineSentimentAnalyzer.
        """

        if d_from is None:
            raise ValueError("[FinvizNewsProcessor][ERROR] d_from cannot be None")

        if isinstance(d_from, str):
            try:
                d_from = datetime.strptime(d_from, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"[FinvizNewsProcessor][ERROR] Invalid date format: {d_from} (expected YYYY-MM-DD)")


        try:
            out_file = FinvizOfflineSentimentAnalyzer.process_portfolio(portfolio, symbol, d_from)
            self.logger.do_log(
                f"[REPORT] ✅ Sentiment summary created -> {out_file}",
                MessageType.INFO
            )
        except Exception as e:
            self.logger.do_log(
                f"[REPORT] ❌ Failed to process Finviz sentiment for {symbol}: {str(e)}",
                MessageType.ERROR
            )

    def _run_quarterly_income_statement(self):
        # ✅ Get securities list from your manager
        securities = self.report_securities_mgr.get_report_securities(ReportType.DOWNLOAD_QUARTERLY_INCOME_STATEMENT.value)
        self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process", MessageType.INFO)

        for i, sec in enumerate(securities):
            symbol = sec.ticker
            try:
                files = YahooIncomeStatement.download(symbol,mode="quarterly")

                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded {len(files)} quarterly Income Statements for {symbol}",
                    MessageType.INFO
                )
            except Exception as e:
                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {str(e)}",
                    MessageType.ERROR
                )

    def _run_download_last_income_statement(self,portfolio):
        # ✅ Get securities from portfolio
        securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)
        self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process", MessageType.INFO)
        ibIncomeStatementDownloader=IBIncomeStatement()
        for i, sec in enumerate(securities):
            symbol = sec.ticker
            try:
                files = ibIncomeStatementDownloader.download(symbol,portfolio=portfolio)

                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded {len(files)} yearly Income Statements for {symbol}",
                    MessageType.INFO
                )
            except Exception as e:
                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {str(e)}",
                    MessageType.ERROR
                )


    def _run_yearly_income_statement(self,portfolio):
        # ✅ Get securities from portfolio
        securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)
        self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process", MessageType.INFO)

        for i, sec in enumerate(securities):
            symbol = sec.ticker
            try:
                files = YahooIncomeStatement.download(symbol,portfolio=portfolio,mode="yearly")

                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded {len(files)} yearly Income Statements for {symbol}",
                    MessageType.INFO
                )
            except Exception as e:
                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {str(e)}",
                    MessageType.ERROR
                )

    def _download_13f_reports(self, year, quarter, rank_folder, job_id):
        """
        Download, process and persist 13F filings as a graph.
        Emits a FINAL structured completion event for MCP clients.
        """

        from datetime import datetime
        import json

        # ---------------------------------------------------------
        # 📊 Global summary
        # ---------------------------------------------------------
        summary = {
            "report": "13f_graph",
            "year": year,
            "quarter": quarter,
            "status": "started",
            "downloaded": False,
            "processed": False,
            "error": None,
            "elapsed_sec": None,
            "filings": {
                "count": 0
            }
        }

        start_time = datetime.now()

        self.logger.do_log(
            f"[13F] 🚀 Starting 13F graph download | year={year} q={quarter}",
            MessageType.INFO,
            job_id
        )

        try:
            # -----------------------------------------------------
            # ⬇️ Download filings
            # -----------------------------------------------------
            downloader = ThirteenFGraphDownloader(
                logger=self.logger,
                out_folder=rank_folder,
                job_id=job_id
            )

            raw_dir, filings = downloader.download(year, quarter)

            summary["downloaded"] = True
            summary["filings"]["count"] = len(filings)

            self.logger.do_log(
                f"[13F] ✔ Reports successfully downloaded | filings={len(filings)}",
                MessageType.INFO,
                job_id
            )

            # -----------------------------------------------------
            # 🧠 Process filings into graph
            # -----------------------------------------------------
            processor = ThirteenFGraphProcessor(
                logger=self.logger,
                job_id=job_id
            )

            edges = processor.process(raw_dir, year, quarter)

            summary["processed"] = True
            summary["edges"] = len(edges)
            summary["status"] = "completed"

            elapsed = (datetime.now() - start_time).total_seconds()
            summary["elapsed_sec"] = round(elapsed, 2)

            self.logger.do_log(
                f"[13F] ✅ Graph generation completed | edges={len(edges)} | {elapsed:.1f}s",
                MessageType.INFO,
                job_id
            )

        except Exception as e:
            summary["status"] = "failed"
            summary["error"] = str(e)

            self.logger.do_log(
                f"[13F] ❌ 13F graph generation failed | {e}",
                MessageType.ERROR,
                job_id
            )

        # ---------------------------------------------------------
        # 🧾 FINAL COMPLETION EVENT (CRITICAL)
        # ---------------------------------------------------------
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "13f_graph",
                "summary": summary
            }),
            MessageType.INFO,
            job_id
        )

    def _create_13f_graph(self, year, quarter, source, rank_folder, job_id):
        """
        Build and persist 13F graph from previously downloaded reports.
        Emits a FINAL structured completion event for MCP clients.
        """

        from datetime import datetime
        import json

        # ---------------------------------------------------------
        # 📊 Global summary
        # ---------------------------------------------------------
        summary = {
            "report": "13f_graph_creation",
            "source": source,
            "year": year,
            "quarter": quarter,
            "status": "started",
            "processed": False,
            "persisted": False,
            "error": None,
            "elapsed_sec": None,
            "edges": 0,
            "input_dir": None,
            "output_file": None,
        }

        start_time = datetime.now()

        self.logger.do_log(
            f"[13F] 🚀 Starting 13F graph creation | source={source} year={year} q={quarter}",
            MessageType.INFO,
            job_id
        )

        try:
            # -----------------------------------------------------
            # 🧠 Process XML filings into edges
            # -----------------------------------------------------
            processor = ThirteenFGraphProcessor(
                logger=self.logger,
                job_id=job_id
            )

            downloader = ThirteenFGraphDownloader(
                logger=self.logger,
                out_folder=rank_folder,
                job_id=job_id
            )

            input_dir = downloader.get_reports_dir(year, quarter, source)
            summary["input_dir"] = input_dir

            edges = processor.process(input_dir, year, quarter)

            summary["processed"] = True
            summary["edges"] = len(edges)

            self.logger.do_log(
                f"[13F] ▶ Graph processed | edges={len(edges)}",
                MessageType.INFO,
                job_id
            )

            # -----------------------------------------------------
            # 💾 Persist graph
            # -----------------------------------------------------
            graph_dir, output_file = downloader.get_graph_file(rank_folder, year, quarter)
            summary["output_file"] = output_file

            self._persist_file_graph(graph_dir, output_file, edges)
            self._persist_store_graph(output_file,year,quarter, job_id)
            summary["persisted"] = True
            summary["status"] = "completed"

            elapsed = (datetime.now() - start_time).total_seconds()
            summary["elapsed_sec"] = round(elapsed, 2)

            self.logger.do_log(
                f"[13F] ✅ Graph persisted | edges={len(edges)} | file={output_file} | {elapsed:.1f}s",
                MessageType.INFO,
                job_id
            )

        except Exception as e:
            summary["status"] = "failed"
            summary["error"] = str(e)

            self.logger.do_log(
                f"[13F] ❌ Failed to build 13F graph | {e}",
                MessageType.ERROR,
                job_id
            )

        # ---------------------------------------------------------
        # 🧾 FINAL COMPLETION EVENT (CRITICAL)
        # ---------------------------------------------------------
        self.logger.do_log(
            json.dumps({
                "event": "completed",
                "report": "13f_graph_creation",
                "summary": summary
            }),
            MessageType.INFO,
            job_id
        )

    def _run_start_mcp(self):
        """
        Starts the MCP WebSocket server.
        Minimal, blocking startup.
        """

        # Prevent double start
        if getattr(self, "_mcp_started", False):
            self.logger.do_log(
                "[MCP] Server already running – skipping",
                MessageType.WARNING
            )
            return

        self._mcp_started = True

        self.progress_bus = ProgressBus()
        self.mcp_registry = build_mcp_registry_reports(orchestrator=self)
        self.mcp_dispatcher = JsonRpcDispatcher(self.mcp_registry,self.progress_bus)

        try:
            # Log startup
            self.logger.do_log(
                f"[MCP] Starting server on {self.mcp_server}:{self.mcp_port}",
                MessageType.INFO
            )

            # Create MCP server instance (already configured elsewhere)
            server = MCPServer(
                host=self.mcp_server,
                port=self.mcp_port,
                dispatcher=self.mcp_dispatcher,  # existing dispatcher,
                bus=self.progress_bus,
                logger=self.logger
            )

            # Run async MCP server (blocks current thread)
            asyncio.run(server.start())

        except Exception as e:
            # Fatal startup error: log and propagate
            self.logger.do_log(
                f"[MCP] ❌ Fatal error while starting server: {e}",
                MessageType.ERROR
            )
            raise

    def process_run_report(self, report_key, year=None,quarter=None,portfolio=None,symbol=None,d_from=None,source=None,dest_folder=None,
                           rank_folder=None,job_id=None,query=None,tag_cfg=None):
        if report_key.lower() == ReportType.DOWNLOAD_K10.value:
            self._run_download_k10(year,portfolio,job_id)
        elif report_key.lower() == ReportType.DOWNLOAD_Q10.value:
            self._run_download_q10(year,portfolio,job_id)
        elif report_key.lower() == ReportType.DOWNLOAD_K8.value:
            self._run_download_k8(year,portfolio,job_id)
        elif report_key.lower() == ReportType.SENTIMENT_SUMMARY_REPORT_K10.value:
            self._run_sentiment_summary_report(year, SECReports.K10.value,portfolio=portfolio,dest_folder=dest_folder,rank_folder=rank_folder,job_id=job_id)
        elif report_key.lower() == ReportType.SENTIMENT_SUMMARY_REPORT_SINGLE_STOCK_K10.value:
            self._run_sentiment_single_security_report(symbol=symbol,year=year, quarter=quarter,report_type=ReportFolder.K10.value,portfolio=portfolio, job_id=job_id)
        elif report_key.lower() == ReportType.SENTIMENT_SUMMARY_REPORT_SINGLE_STOCK_Q10.value:
            self._run_sentiment_single_security_report(symbol=symbol,year=year, quarter=quarter,report_type=ReportFolder.Q10.value,portfolio=portfolio, job_id=job_id)
        elif report_key.lower() == ReportType.SENTIMENT_SUMMARY_REPORT_Q10.value:
            self._run_sentiment_summary_report(year, SECReports.Q10.value,portfolio=portfolio,dest_folder=dest_folder,rank_folder=rank_folder,job_id=job_id)
        elif report_key.lower() == ReportType.COMPETITION_SUMMARY_REPORT_Q10.value:
            self._run_competition_summary_report(year, SECReports.Q10.value,portfolio=portfolio,dest_folder=dest_folder,rank_folder=rank_folder)
        elif report_key.lower() == ReportType.COMPETITION_SUMMARY_REPORT_K10.value:
            self._run_competition_summary_report(year, SECReports.K10.value,portfolio=portfolio,dest_folder=dest_folder,rank_folder=rank_folder)
        elif report_key.lower() == ReportType.FINVIZ_NEWS_DOWNLOAD.value:
            self._run_fin_viz_news_downloader(portfolio,symbol,job_id)
        elif report_key.lower() == ReportType.FINANCIAL_RATIOS_REPORT_SA.value:
            self.financial_ratios_report_SA(symbol,job_id)
        #
        elif report_key.lower() == ReportType.PROCESS_FINVIZ_NEWS.value:
            self._run_process_finviz_news(portfolio,symbol,d_from)
        elif report_key.lower() == ReportType.DOWNLOAD_LAST_INCOME_STATEMENT.value:
            self._run_download_last_income_statement(portfolio)
        elif report_key.lower() == ReportType.DOWNLOAD_YEARLY_INCOME_STATEMENT.value:
            self._run_yearly_income_statement(portfolio)
        elif report_key.lower() == ReportType.DOWNLOAD_QUARTERLY_INCOME_STATEMENT.value:
            self._run_quarterly_income_statement()
        elif report_key.lower() == ReportType.QUERY_MATCH_REPORT_K10.value:
            self.run_query_match_report_KQ_10(year, SECReports.K10.value, portfolio=portfolio, dest_folder=dest_folder,
                                              query=query)
        elif report_key.lower() == ReportType.DOWNLOAD_SECURITIES_REPORTS_CALENDAR.value:
            self._run_download_securities_calendar(year,portfolio,job_id)
        elif report_key.lower() == ReportType.DOCUMENT_TAGGING_RANKING.value:
            self._run_document_tagging(portfolio, year,quarter, source, rank_folder, tag_cfg,job_id)
        elif report_key.lower() == ReportType.DOCUMENT_TAGGING_SINGLE_SECURITY.value:
            self._run_document_single_security(symbol=symbol,source=source,year=year,quarter=quarter,tag_cfg=tag_cfg,job_id=job_id)
        elif report_key.lower() == ReportType.COMPETITION_GRAPH.value:
            self._run_competition_graph(portfolio, year,quarter, source, rank_folder,job_id)
        elif report_key.lower() == ReportType.DOWNLOAD_13F_REPORTS.value:
            self._download_13f_reports(year, quarter, rank_folder, job_id)
        elif report_key.lower() == ReportType.CREATE_13F_GRAPH.value:
            self._create_13f_graph(year, quarter,source, rank_folder, job_id)
        #
        elif report_key.lower() == ReportType.START_MCP.value:
            self._run_start_mcp()
        else:
            self.logger.do_log(f"[REPORT] Report {report_key} not implemented.", MessageType.WARNING,job_id)
        '''
        elif report_key.lower() == ReportType.FINANCIAL_RATIOS_REPORT_K10.value:
            self._run_financial_ratios_report(year, SECReports.K10.value)
        elif report_key.lower() == ReportType.FINANCIAL_RATIOS_REPORT_Q10.value:
            self._run_financial_ratios_report(year, SECReports.Q10.value)
        '''

