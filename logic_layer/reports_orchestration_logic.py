import os
import shutil
from datetime import datetime

from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from common.enums.report_type import ReportType
from common.enums.sec_reports import SECReports
from common.util.downloaders.finviz_full_news_downloader import FinVizFullNewsDownloader
from common.util.downloaders.finviz_offline_sentiment_analyzer import FinvizOfflineSentimentAnalyzer
from common.util.downloaders.ib_income_statement import IBIncomeStatement
from common.util.downloaders.k10_downloader import K10Downloader
from common.util.downloaders.q10_downloader import Q10Downloader
from common.util.downloaders.yahoo_income_statement import YahooIncomeStatement
from data_access_layer.portfolio_securities_manager import PortfolioSecuritiesManager
from data_access_layer.report_securities_manager import ReportSecuritiesManager
from framework.common.logger.message_type import MessageType
from logic_layer.report_generators.competition_summary_report import CompetitionSummaryReport
from logic_layer.report_generators.sentence_sentiment_summary_report import SentimentSummaryReport


class ReportsOrchestationLogic:
    def __init__(self,hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger):

        self.logger=logger

        self.report_securities_mgr = ReportSecuritiesManager(ml_reports_conn_str, logger)

        self.portfolio_securities_mgr = PortfolioSecuritiesManager(ml_reports_conn_str,logger)

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


    def _run_download_k10(self, year,portfolio):
        base_path = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{ReportFolder.K10.value}/{year}"
        self.logger.do_log(f"[REPORT] Downloading K10 to  {base_path}", MessageType.INFO)
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
            self.logger.do_log(f"[REPORT] Removed existing directory {base_path}", MessageType.INFO)

        os.makedirs(base_path, exist_ok=True)

        # ✅ Get securities from portfolio
        securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

        self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process", MessageType.INFO)

        for i, sec in enumerate(securities):
            symbol = sec.ticker
            cik=sec.cik
            try:
                K10Downloader.download_k10(symbol,cik, year, base_path)
                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded K10 for {symbol}",
                    MessageType.INFO
                )
            except Exception as e:
                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {str(e)}",
                    MessageType.ERROR
                )

    def _run_download_q10(self, year,portfolio):
        base_path = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{ReportFolder.Q10.value}/{year}"
        self.logger.do_log(f"[REPORT] Downloading Q10 to  {base_path}", MessageType.INFO)
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
            self.logger.do_log(f"[REPORT] Removed existing directory {base_path}", MessageType.INFO)

        os.makedirs(base_path, exist_ok=True)

        # ✅ Get securities from portfolio
        securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

        self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process", MessageType.INFO)

        for i, sec in enumerate(securities):
            symbol = sec.ticker
            cik = sec.cik
            try:
                q10_files = Q10Downloader.download_q10s(symbol, cik, year, base_path)
                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded {len(q10_files)} Q10(s) for {symbol}",
                    MessageType.INFO
                )
            except Exception as e:
                self.logger.do_log(
                    f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {str(e)}",
                    MessageType.ERROR
                )


    def _get_universe_filers(self, universe_key: str):
        if not universe_key:
            return None
        dtos = self.report_securities_mgr.get_report_securities(universe_key)
        return sorted({(d.ticker or "").upper() for d in dtos if d.ticker})

    def _run_sentiment_summary_report(self, year, report_type=ReportFolder.K10.value,portfolio=None, universe=None):
        """
        Build sentiment summaries focused on management guidance/opinion.
        Extract MD&A / Outlook-like text, score sentiment, and consolidate.
        :param year: Filing year
        :param report_type: "K10" or "Q10"
        :param universe: optional universe key for subfolder
        """
        # Universe → list of tickers (or None)
        whitelist = self._get_universe_filers(universe) if universe else None

        # Instantiate processor with report_type
        gen = SentimentSummaryReport(
            year=year,
            report_type=report_type,
            logger=self.logger,
            portfolio=portfolio,
            filers_whitelist=whitelist,
            universe_key=universe
        )
        gen.run()

        # Consolidate & rank (year + report_type aware)
        consolidated = SentimentSummaryReport.consolidate_year(year, report_type,portfolio, self.logger, universe_key=universe)
        ranking_csv = os.path.join(os.path.dirname(consolidated), f"sentiment_summary_ranking_{year}.csv")
        SentimentSummaryReport.rank(consolidated, ranking_csv, self.logger)

        self.logger.do_log(
            f"[SENT] ✅ Sentiment summary completed ({report_type}, scope={universe or 'ALL'})",
            MessageType.INFO
        )

    def _run_competition_summary_report(self, year, report_type="K10",portfolio=None):
        report = CompetitionSummaryReport(
            year=year,
            logger=self.logger,
            report_type=report_type,
            portfolio=portfolio
        )
        report.run()

    def _run_fin_viz_news_downloader(self,portfolio,symbol=None):

        if portfolio!="SINGLE_STOCKS":
            # ✅ Get securities from portfolio
            securities = self.portfolio_securities_mgr.get_portfolio_securities(portfolio)

            self.logger.do_log(f"[REPORT] Found {len(securities)} securities to process", MessageType.INFO)

            for i, sec in enumerate(securities):
                symbol = sec.ticker
                try:
                    out_file = FinVizFullNewsDownloader.download(symbol,portfolio)

                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}] ✅ Downloaded news for {symbol} -> {out_file}",
                        MessageType.INFO
                    )
                except Exception as e:
                    self.logger.do_log(
                        f"[REPORT][{i + 1}/{len(securities)}] ❌ Failed for {symbol}: {str(e)}",
                        MessageType.ERROR
                    )
        else:
            try:
                FinVizFullNewsDownloader.download(symbol, portfolio)
                pass


            except Exception as e:
                self.logger.do_log(f"[REPORT] ❌ Failed for {symbol}: {str(e)}",MessageType.ERROR)


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

    def process_run_report(self, report_key, year=None,portfolio=None,symbol=None,d_from=None):
        if report_key.lower() == ReportType.DOWNLOAD_K10.value:
            self._run_download_k10(year,portfolio)
        elif report_key.lower() == ReportType.DOWNLOAD_Q10.value:
            self._run_download_q10(year,portfolio)
        elif report_key.lower() == ReportType.SENTIMENT_SUMMARY_REPORT_K10.value:
            self._run_sentiment_summary_report(year, SECReports.K10.value,portfolio=portfolio)
        elif report_key.lower() == ReportType.SENTIMENT_SUMMARY_REPORT_Q10.value:
            self._run_sentiment_summary_report(year, SECReports.Q10.value,portfolio=portfolio)
        elif report_key.lower() == ReportType.COMPETITION_SUMMARY_REPORT_Q10.value:
            self._run_competition_summary_report(year, SECReports.Q10.value,portfolio=portfolio)
        elif report_key.lower() == ReportType.COMPETITION_SUMMARY_REPORT_K10.value:
            self._run_competition_summary_report(year, SECReports.K10.value,portfolio=portfolio)
        elif report_key.lower() == ReportType.FINVIZ_NEWS_DOWNLOAD.value:
            self._run_fin_viz_news_downloader(portfolio,symbol)
        elif report_key.lower() == ReportType.PROCESS_FINVIZ_NEWS.value:
            self._run_process_finviz_news(portfolio,symbol,d_from)
        elif report_key.lower() == ReportType.DOWNLOAD_LAST_INCOME_STATEMENT.value:
            self._run_download_last_income_statement(portfolio)
        elif report_key.lower() == ReportType.DOWNLOAD_YEARLY_INCOME_STATEMENT.value:
            self._run_yearly_income_statement(portfolio)
        elif report_key.lower() == ReportType.DOWNLOAD_QUARTERLY_INCOME_STATEMENT.value:
            self._run_quarterly_income_statement()
        else:
            self.logger.do_log(f"[REPORT] Report {report_key} not implemented.", MessageType.WARNING)
        '''
        elif report_key.lower() == ReportType.FINANCIAL_RATIOS_REPORT_K10.value:
            self._run_financial_ratios_report(year, SECReports.K10.value)
        elif report_key.lower() == ReportType.FINANCIAL_RATIOS_REPORT_Q10.value:
            self._run_financial_ratios_report(year, SECReports.Q10.value)
        '''

