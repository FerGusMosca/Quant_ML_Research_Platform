import os
import shutil
from datetime import datetime
from pathlib import Path
import asyncio

from common.dto.mcp.bootstrap_registry import build_mcp_registry
from common.dto.mcp.dispatcher import JsonRpcDispatcher
from common.dto.mcp.progress_bus import ProgressBus
from common.dto.mcp.tools import ToolRegistry
from common.dto.security_report_calendar import SecurityReportCalendar
from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from common.enums.report_type import ReportType
from common.enums.sec_reports import SECReports
from common.util.date_mgmt.date_range_handler import DateRangeHandler
from common.util.downloaders.finviz_full_news_downloader import FinVizFullNewsDownloader
from common.util.downloaders.finviz_offline_sentiment_analyzer import FinvizOfflineSentimentAnalyzer
from common.util.downloaders.ib_income_statement import IBIncomeStatement
from common.util.downloaders.k10_downloader import K10Downloader
from common.util.downloaders.q10_downloader import Q10Downloader
from common.util.downloaders.securities_calendar_downloader import SecuritiesCalendarDownloader
from common.util.downloaders.yahoo_income_statement import YahooIncomeStatement
from common.util.scrappers.securities_calendar_extractor import SecuritiesCalendarExtractor
from common.util.std_in_out.file_locators import FileLocators
from common.util.std_in_out.root_locator import RootLocator
from data_access_layer.portfolio_securities_manager import PortfolioSecuritiesManager
from data_access_layer.report_securities_manager import ReportSecuritiesManager
from data_access_layer.securities_calendar_manager import SecuritiesCalendarManager
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.tagger.transformers_topic_tagger import TransformersTopicTagger
from logic_layer.report_generators.competition_summary_report import CompetitionSummaryReport
from logic_layer.report_generators.query_match_report import QueryMatchReportK10Q10
from logic_layer.report_generators.sentiment.sentence_sentiment_summary_report import SentimentSummaryReport
from logic_layer.report_generators.sentiment.sentence_sentiment_summary_report_v2 import SentimentSummaryReportV2
from service_layer.server.mcp_server import MCPServer


class ReportsOrchestationLogic:
    def __init__(self,hist_data_conn_str,ml_reports_conn_str,mcp_server=None,mcp_port=None,p_classification_map_key=None,
                 logger=None):

        self.logger=logger

        self.report_securities_mgr = ReportSecuritiesManager(ml_reports_conn_str, logger)

        self.portfolio_securities_mgr = PortfolioSecuritiesManager(ml_reports_conn_str,logger)

        self.sec_cal_mgr =SecuritiesCalendarManager(ml_reports_conn_str)

        self.mcp_server=mcp_server
        self.mcp_port=mcp_port

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

    def _run_download_k10(self, year, portfolio):
        # parse years
        years=DateRangeHandler.handle_date_range(year,self.logger)
        single_year= len(years)==1

        for y in years:
            base_path = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{ReportFolder.K10.value}/{y}"
            self.logger.do_log(f"[REPORT] Downloading K10 to {base_path}", MessageType.INFO)

            # only remove existing dir when user asked a single year (explicit overwrite behavior)
            if 'single_year' in locals() and single_year:
                if os.path.exists(base_path):
                    shutil.rmtree(base_path)
                    self.logger.do_log(f"[REPORT] Removed existing directory {base_path}", MessageType.INFO)

            os.makedirs(base_path, exist_ok=True)

            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)
            self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process for year {y}", MessageType.INFO)

            for i, sec in enumerate(securities):
                symbol = sec.ticker
                cik = sec.cik
                try:
                    result = K10Downloader.download_k10(symbol, cik, y, base_path)
                    if result == "EXISTS":
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ⚠️ Skipped {symbol}: file already exists ({y})",
                            MessageType.INFO)
                    elif result == "NOT_FOUND":
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ❌ No 10-K available yet for {symbol} ({y})",
                            MessageType.WARNING)
                    else:
                        self.logger.do_log(f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded K10 for {symbol} ({y})",
                                           MessageType.INFO)
                except Exception as e:
                    self.logger.do_log(f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {e}",
                                       MessageType.ERROR)

    def _run_download_q10(self, year, portfolio):
        # ---------------------------------------------------------
        # 🧠 Parse year(s)
        # ---------------------------------------------------------
        years=DateRangeHandler.handle_date_range(year,self.logger)

        # ---------------------------------------------------------
        # 🚀 Process each year
        # ---------------------------------------------------------
        for y in years:
            base_path = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{ReportFolder.Q10.value}/{y}"
            self.logger.do_log(f"[REPORT] Downloading Q10 to {base_path}", MessageType.INFO)

            # ✅ Ensure directory exists (no deletion at all)
            os.makedirs(base_path, exist_ok=True)

            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)
            self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process for year {y}", MessageType.INFO)

            for i, sec in enumerate(securities):
                symbol = sec.ticker
                cik = sec.cik
                try:
                    result = Q10Downloader.download_q10s(symbol, cik, y, base_path)
                    if result == "EXISTS":
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ⚠️ Skipped {symbol}: files already exist ({y})",
                            MessageType.INFO)
                    elif result == "NOT_FOUND":
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ❌ No 10-Q available yet for {symbol} ({y})",
                            MessageType.WARNING)
                    else:
                        self.logger.do_log(
                            f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded {len(result)} Q10(s) for {symbol} ({y})",
                            MessageType.INFO)
                except Exception as e:
                    self.logger.do_log(f"[REPORT][{i + 1}/{len(securities)}] 💥 Failed for {symbol}: {e}",
                                       MessageType.ERROR)

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

    def _run_sentiment_summary_report(self, year, report_type=ReportFolder.K10.value,
                                      portfolio=None, universe=None, dest_folder=None,
                                      rank_folder=None):
        """
        Build sentiment summaries focused on management guidance/opinion.
        Extract MD&A / Outlook-like text, score sentiment, and consolidate.
        Supports both single year (e.g. 2024) and range (e.g. 2022-2025).
        """
        # Parse year or range
        years=DateRangeHandler.handle_date_range(year,self.logger)

        for y in years:
            start_time = datetime.now()
            self.logger.do_log(f"[SENT] 🚀 Starting sentiment summary ({report_type}, year={y})", MessageType.INFO)

            whitelist = self._get_universe_filers(universe) if universe else None
            gen=SentimentSummaryReportV2(
            #gen = SentimentSummaryReport(
                year=y,
                report_type=report_type,
                logger=self.logger,
                portfolio=portfolio,
                filers_whitelist=whitelist,
                universe_key=universe,
                dest_folder=dest_folder,
                rank_folder=rank_folder
            )

            try:
                gen.run()
            except Exception as e:
                self.logger.do_log(f"[SENT] ❌ Error during run() for year {y}: {e}", MessageType.ERROR)
                continue

            try:
                if report_type==ReportFolder.K10.value:
                    consolidated = gen.consolidate_year(y, report_type)
                    ranking_csv = os.path.join(os.path.dirname(consolidated), f"sentiment_summary_ranking_{y}.csv")
                    SentimentSummaryReport.rank(consolidated, ranking_csv, self.logger)
                    pass
                else:
                    for quarter in [1,2,3]:
                        consolidated=gen.consolidate_year(y,report_type,quarter)
                        '''
                        consolidated = SentimentSummaryReport.consolidate_year(
                                        y,
                                        report_type,
                                        portfolio,
                                        self.logger,
                                        dest_folder=dest_folder,
                                        rank_folder=rank_folder
                                    )
                        
                        '''
                        ranking_csv = os.path.join(os.path.dirname(consolidated),f"sentiment_summary_ranking_{y}.csv")
                        SentimentSummaryReport.rank(consolidated, ranking_csv, self.logger)

            except Exception as e:
                self.logger.do_log(f"[SENT] ⚠️ Consolidation/Ranking failed for {y}: {e}",
                                   MessageType.WARNING)
                continue

            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.do_log(
                f"[SENT] ✅ Sentiment summary completed ({report_type}, year={y}) in {elapsed:.1f}s",
                MessageType.INFO
            )

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

    def _run_document_tagging(self, portfolio, year, source, dest_folder, tag_cfg):
        """
        Runs document tagging for a portfolio/year range.
        Filters files by portfolio securities (symbol match in filename).
        """

        tagger=TransformersTopicTagger(self.logger,tag_cfg)

        # Parse year or year range
        years = DateRangeHandler.handle_date_range(year, self.logger)

        # Load portfolio securities once
        securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

        for y in years:
            start_time = datetime.now()
            self.logger.do_log(
                f"[TAGGING] 🚀 Starting Document Tagging (source={source}, dest={dest_folder}, year={y})",
                MessageType.INFO
            )

            file_folder = os.path.join(source, y)

            matched_files= FileLocators.enumerate_all_files(file_folder,self.logger,
                                                            filters= [s.symbol.lower() for s in securities if getattr(s, "symbol", None)])


            #TODo process matched_files in document tagger


            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.do_log(
                f"[TAGGING] 🏁 Completed year={y} in {elapsed:.2f}s",
                MessageType.INFO
            )

    def _run_download_securities_calendar(self, year, portfolio):
        """
        Download and persist SEC filing calendars for all securities in the given portfolio.
        """
        self.logger.do_log(f"[REPORT] Starting SEC calendar download for portfolio={portfolio}, year={year}",
                           MessageType.INFO)

        securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

        from_year, to_year = map(int, str(year).split('-')) if '-' in str(year) else (int(year), int(year))
        existing = self.sec_cal_mgr.get_calendars_by_range(from_year, to_year)
        root_dir = RootLocator.get_root(markers=["bias_mgmt_console.py", "README.md"])

        for i, sec in enumerate(securities):
            for yr in range(from_year, to_year + 1):

                try:
                    # Inside your loop for each security and year
                    k10_dir = (
                            root_dir
                            / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                            / portfolio
                            / "K10"
                            / str(yr)
                    )

                    q10_dir = (
                            root_dir
                            / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                            / portfolio
                            / "Q10"
                            / str(yr)
                    )

                    # Extract real filing dates from downloaded files
                    k10_filing_date, q10_filing_dates = SecuritiesCalendarExtractor.process_k10_q10_directories(
                        sec.ticker, k10_dir, q10_dir)

                    # Build the calendar entry using real extracted dates
                    entry = SecurityReportCalendar(
                        cik=sec.cik,
                        symbol=sec.ticker,
                        fiscal_year=yr,
                        q1=q10_filing_dates[1],
                        q2=q10_filing_dates[2],
                        q3=q10_filing_dates[3],
                        k10=k10_filing_date
                    )

                    self.sec_cal_mgr.upsert_calendar_entry(entry)

                    self.logger.do_log(f"[REPORT][{i + 1}/{len(securities)}][{yr}] ✅ {sec.ticker} saved.",
                                       MessageType.INFO)
                except Exception as e:
                    self.logger.do_log(f"[REPORT][{i + 1}/{len(securities)}][{yr}] ❌ {sec.ticker} failed: {e}",
                                       MessageType.ERROR)

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
        self.mcp_registry = build_mcp_registry(orchestrator=self)
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

    def process_run_report(self, report_key, year=None,portfolio=None,symbol=None,d_from=None,source=None,dest_folder=None,
                           rank_folder=None,job_id=None,query=None,tag_cfg=None):
        if report_key.lower() == ReportType.DOWNLOAD_K10.value:
            self._run_download_k10(year,portfolio)
        elif report_key.lower() == ReportType.DOWNLOAD_Q10.value:
            self._run_download_q10(year,portfolio)
        elif report_key.lower() == ReportType.SENTIMENT_SUMMARY_REPORT_K10.value:
            self._run_sentiment_summary_report(year, SECReports.K10.value,portfolio=portfolio,dest_folder=dest_folder,rank_folder=rank_folder)
        elif report_key.lower() == ReportType.SENTIMENT_SUMMARY_REPORT_Q10.value:
            self._run_sentiment_summary_report(year, SECReports.Q10.value,portfolio=portfolio,dest_folder=dest_folder,rank_folder=rank_folder)
        elif report_key.lower() == ReportType.COMPETITION_SUMMARY_REPORT_Q10.value:
            self._run_competition_summary_report(year, SECReports.Q10.value,portfolio=portfolio,dest_folder=dest_folder,rank_folder=rank_folder)
        elif report_key.lower() == ReportType.COMPETITION_SUMMARY_REPORT_K10.value:
            self._run_competition_summary_report(year, SECReports.K10.value,portfolio=portfolio,dest_folder=dest_folder,rank_folder=rank_folder)
        elif report_key.lower() == ReportType.FINVIZ_NEWS_DOWNLOAD.value:
            self._run_fin_viz_news_downloader(portfolio,symbol,job_id)
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
            self._run_download_securities_calendar(year,portfolio)
        elif report_key.lower() == ReportType.DOCUMENT_TAGGING_RANKING.value:
            self._run_document_tagging(portfolio,year,source,dest_folder,tag_cfg)
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

