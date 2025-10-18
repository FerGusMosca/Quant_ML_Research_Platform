import json
import os
import shutil
import traceback
from datetime import timedelta, datetime
from decimal import Decimal

from dateutil.relativedelta import relativedelta
from decimal import Decimal, ROUND_HALF_UP



from business_entities.porft_summary import PortfSummary
from common.dto.indicator_type_dto import IndicatorTypeDTO
from common.dto.sec_security_dto import SecSecurityDTO
from common.enums.filename_prefix import FilenamePrefix
from common.enums.folders import Folders
from common.enums.grouping_criterias import GroupCriteria as gc
from common.enums.information_vendors import InformationVendors
from common.enums.intervals import Intervals
from common.enums.parameters.parameters_keys import ParametersKeys
from common.enums.report_folder import ReportFolder
from common.enums.report_type import ReportType
from common.enums.sec_reports import SECReports
from common.enums.sliding_window_strategy import SlidingWindowStrategy as sws, SlidingWindowStrategy
from common.enums.trading_algo_strategy import TradingAlgoStrategy as tas, TradingAlgoStrategy
from common.util.downloaders.FRED_downloader import FredDownloader
from common.util.downloaders.SEC_securities_downloader import SECSecuritiesDownloader
from common.util.downloaders.finviz_full_news_downloader import FinVizFullNewsDownloader
from common.util.downloaders.finviz_offline_sentiment_analyzer import FinvizOfflineSentimentAnalyzer

from common.util.downloaders.ib_income_statement import IBIncomeStatement
from common.util.downloaders.k10_downloader import K10Downloader
from common.util.downloaders.q10_downloader import Q10Downloader
from common.util.downloaders.tradingview_downloader import TradingViewDownloader
from common.util.downloaders.yahoo_income_statement import  YahooIncomeStatement
from common.util.financial_calculations.PCA_calculator import PCACalcualtor
from common.util.pandas_dataframes.dataframe_filler import DataframeFiller
from common.util.pandas_dataframes.dataframe_printer import DataframePrinter
from common.util.financial_calculations.date_handler import DateHandler
from common.util.std_in_out.etf_file_extraction_handler import ETFFileExtractionHandler
from common.util.financial_calculations.financial_calculation_helper import FinancialCalculationsHelper
from common.util.graphs.graph_builder import GraphBuilder
from common.util.graphs.image_handler import ImageHandler
from common.util.logging.light_logger import LightLogger
from common.util.financial_calculations.portfolio_summary_analyzer import PortfolioSummaryAnalyzer
from common.util.financial_calculations.random_walk_generator import RandomWalkGenerator
from common.util.financial_calculations.slope_calculator import SlopeCalculator
from data_access_layer.date_range_classification_manager import DateRangeClassificationManager
from data_access_layer.economic_series_manager import EconomicSeriesManager
from data_access_layer.portfolio_securities_manager import PortfolioSecuritiesManager
from data_access_layer.report_securities_manager import ReportSecuritiesManager
from data_access_layer.sec_securities_manager import SECSecuritiesManager
from data_access_layer.timestamp_classification_manager import TimestampClassificationManager

from framework.common.logger.message_type import MessageType
from logic_layer.ARIMA_models_analyzer import ARIMAModelsAnalyzer
from logic_layer.convolutional_neural_netowrk import ConvolutionalNeuralNetwork
from logic_layer.indicator_algos.sintethic_indicator_creator import SintheticIndicatorCreator
from logic_layer.model_creators.random_forest_model_creator import RandomForestModelCreator
from logic_layer.model_creators.xg_boost_model_creator import XGBoostModelCreator
from logic_layer.report_generators.competition_summary_report import CompetitionSummaryReport
from logic_layer.report_generators.financial_ratios_summary_report import FinancialRatiosSummaryReport
from logic_layer.report_generators.sentence_sentiment_summary_report import SentimentSummaryReport
from logic_layer.trading_algos.buy_and_hold_backtester import BuyAndHoldBacktester
from logic_layer.trading_algos.direct_slope_backtester import DirectSlopeBacktester
from logic_layer.trading_algos.inv_slope_backtester import InvSlopeBacktester
from logic_layer.trading_algos.n_flip_prediction_backtester import NFlipPredictionBacktester
from logic_layer.trading_algos.n_min_buffer_w_flip_daily_trading_backtester import NMinBufferWFlipDailyTradingBacktester
from logic_layer.trading_algos.on_off_backtester import OnOffBacktester
from logic_layer.trading_algos.only_signal_n_min_plus_mov_avg import OnlySignalNMinMovAvgBacktester
from logic_layer.trading_algos.raw_algo_daily_trading_backtester import RawAlgoDailyTradingBacktester
from logic_layer.data_set_builder import DataSetBuilder
from logic_layer.deep_neural_network import DeepNeuralNetwork
from logic_layer.indicator_based_trading_backtester import IndicatorBasedTradingBacktester
from logic_layer.ml_models_analyzer import MLModelAnalyzer
import pandas as pd

from logic_layer.neural_network_models_trainer import NeuralNetworkModelTrainer
from logic_layer.model_creators.daytrading_RNN_model_creator import DayTradingRNNModelCreator
from service_layer.bcra_service_layer import BCRAServiceLayer


class AlgosOrchestationLogic:
    def __init__(self,hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger):

        self.logger=logger

        self.data_set_builder=DataSetBuilder(hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger)

        self.date_range_classif_mgr = DateRangeClassificationManager(ml_reports_conn_str)

        self.timestamp_range_classif_mgr=TimestampClassificationManager(ml_reports_conn_str)

        self.economic_series_mgr = EconomicSeriesManager(hist_data_conn_str)

        self.sec_securities_mgr = SECSecuritiesManager(ml_reports_conn_str,logger)

        self.report_securities_mgr = ReportSecuritiesManager(ml_reports_conn_str, logger)

        self.portfolio_securities_mgr = PortfolioSecuritiesManager(ml_reports_conn_str,logger)



    def __classify_group__(self,classifications, grouping_classif_criteria):
        unique_classes = classifications.unique()

        # If there's only one unique classification in the group, return it
        if len(unique_classes) == 1:
            return unique_classes[0]

        # Apply different criteria for multiple classifications
        if gc(grouping_classif_criteria) == gc.GO_FIRST:
            # Return the first classification in the group
            return classifications.iloc[0]
        elif gc(grouping_classif_criteria) == gc.GO_HIGHEST_COUNT:
            # Return the most frequent classification
            return classifications.mode().iloc[0]
        elif gc(grouping_classif_criteria) == gc.GO_FLAT_ON_DIFF:
            # Mark the entire group as "FLAT"
            return "FLAT"
        else:
            raise ValueError("Unknown grouping_classif_criteria")

    def __group_dataframe__(self, training_series_df, grouping_unit, variables_csv, grouping_classif_criteria=None,
                            classif_key=None):
        """
        Groups the DataFrame by a specified time unit, applies OHLC (open, high, low, close) calculations
        on the specified variables, and optionally adds a classification column based on a criterion.

        :param training_series_df: The input DataFrame containing time series data.
        :param grouping_unit: The time unit to group by (e.g., '10min', '30min').
        :param variables_csv: Comma-separated string of variable names (e.g., 'SPY,VIX').
        :param grouping_classif_criteria: Criteria for adding a classification column (optional).
        :param classif_key: Column to be classified based on the grouping criteria (optional).
        :return: DataFrame with grouped OHLC data for the specified variables.
        """
        # Convert the 'date' column to datetime and set it as the DataFrame index
        training_series_df['date'] = pd.to_datetime(training_series_df['date'])
        training_series_df = training_series_df.set_index('date')

        # List to store grouped data for each variable
        grouped_data = []

        # Iterate over the variables (symbols) listed in variables_csv
        for var in variables_csv.split(','):
            # Ensure the column for the variable exists in the DataFrame
            if var not in training_series_df.columns:
                continue

            # Resample the data based on the grouping unit and calculate OHLC (open, high, low, close)
            resampled_df = training_series_df[[var]].resample(f'{grouping_unit}min').agg({
                var: ['first', 'max', 'min', 'last']
            })

            # Rename the columns to reflect 'open', 'high', 'low', 'close' for the current variable
            resampled_df.columns = [f'open_{var}', f'high_{var}', f'low_{var}', f'close_{var}']

            # Append the resampled and renamed DataFrame to the grouped data list
            grouped_data.append(resampled_df)

        # Concatenate all grouped DataFrames along the columns
        final_grouped_df = pd.concat(grouped_data, axis=1)

        # Optionally add a classification column based on the specified criteria
        if grouping_classif_criteria is not None and classif_key is not None:
            # Resample the classification key and apply the classification criteria
            classification_df = training_series_df.resample(f'{grouping_unit}min')[classif_key].apply(
                lambda x: self.__classify_group__(x, grouping_classif_criteria)
            )
            # Add the classification data to the final grouped DataFrame
            final_grouped_df[classif_key] = classification_df

        # Reset the index to bring 'date' back as a column
        final_grouped_df = final_grouped_df.reset_index()

        # Drop any rows with NaN values that might have been generated during resampling
        final_grouped_df = final_grouped_df.dropna()

        return final_grouped_df

    def __log_day_trading_results__(self,day,daily_net_profit,total_positions,trading_summary_df):
        self.logger.do_log(
            f"Results for day {day}: Net_Profit=${daily_net_profit:.2f} (total positions={total_positions})",
            MessageType.INFO)
        self.logger.do_log("---Summarizing trades---", MessageType.INFO)
        for index, row in trading_summary_df[trading_summary_df['total_net_profit'].notnull()].iterrows():
            self.logger.do_log(
                f" Pos: {row['side']} --> open_time={row['open']} close_time={row['close']} open_price=${row['price_open']:.2f} close_price=${row['price_close']:.2f} --> net_profit=${row['total_net_profit']}",
                MessageType.INFO)
        self.logger.do_log("--------------------", MessageType.INFO)

    def __log_scalping_trading_results__(self, d_from, d_to, portf_init_size, summary_dto):

        max_cum_drawdown_percentage = summary_dto.max_cum_drawdown * 100
        portf_final_size = summary_dto.daily_net_profit + portf_init_size

        # --- CAGR ---
        days_diff = (d_to - d_from).days
        years = days_diff / 365.0
        cagr = (portf_final_size / portf_init_size) ** (1 / years) - 1
        cagr_pct = cagr * 100.0

        self.logger.do_log(
            f"Results for algo {summary_dto.algo}: Net_Profit=${summary_dto.daily_net_profit:,.2f} "
            f"(total positions={summary_dto.total_positions} Max Drawdown ={max_cum_drawdown_percentage:.2f}% "
            f"| CAGR={cagr_pct:.2f}%)",
            MessageType.INFO
        )

        self.logger.do_log("---Summarizing trades---", MessageType.INFO)
        for index, row in summary_dto.trading_summary_df[
            summary_dto.trading_summary_df['total_net_profit'].notnull()].iterrows():
            self.logger.do_log(
                f" Pos: {row['side']} --> open_time={row['open']} close_time={row['close']} "
                f"open_price=${row['price_open']:,.2f} close_price=${row['price_close']:,.2f} --> "
                f"net_profit=${row['total_net_profit']:,.2f} ==> Final Portfolio = ${row['end_portfolio']:,.2f}"
                f" ( Pos. Profit=${row['pct_profit']}  Max. Cum. Drawdown=${row['max_drawdown']})",
                MessageType.INFO
            )

        self.logger.do_log("--------------------", MessageType.INFO)

    def __sliding_window__(self,df, timesteps):
        """
        Genera ventanas deslizantes de tamaño `timesteps + 1` en un DataFrame.

        Args:
        - df: DataFrame de entrada ordenado por tiempo.
        - timesteps: Número de pasos de tiempo a considerar (ventana de tamaño timesteps + 1).

        Yields:
        - Un DataFrame para cada ventana deslizante.
        """
        for start in range(len(df) - timesteps):
            yield df.iloc[start: start + timesteps + 1]

    # def __backtest_scalping_strategy__(self,rnn_predictions_df,portf_size, trade_comm,trading_algo,n_algo_params=[]):
    #
    #     #print(f"{rnn_predictions_df.head()}")
    #
    #     if tas(trading_algo)==tas.RAW_ALGO:
    #         daily_trading_backtester = RawAlgoDailyTradingBacktester()
    #         daily_net_profit, total_positions, max_daily_cum_drawdown, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(
    #                                                                                         rnn_predictions_df, portf_size, trade_comm,n_algo_params)
    #
    #         return daily_net_profit, total_positions, max_daily_cum_drawdown, trading_summary_df
    #     elif tas(trading_algo)==tas.N_MIN_BUFFER_W_FLIP:
    #         raise Exception(f'Not implemented algo')
    #     elif tas(trading_algo)==tas.ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG:
    #         raise Exception(f'Not implemented algo')
    #     else:
    #         raise Exception(f"NOT RECOGNIZED trading algo {trading_algo}")
    def __backtest_daily_strategy__(self,rnn_predictions_df,portf_summary):


        #print(f"{rnn_predictions_df.head()}")

        if tas(portf_summary.trading_algo)==tas.RAW_ALGO:
            daily_trading_backtester = RawAlgoDailyTradingBacktester()
            summary_dto, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(rnn_predictions_df, portf_summary.portf_pos_size, portf_summary.trade_comm,portf_summary.n_algo_params)

            return summary_dto, trading_summary_df
        elif tas(portf_summary.trading_algo)==tas.N_MIN_BUFFER_W_FLIP:


            daily_trading_backtester = NMinBufferWFlipDailyTradingBacktester()
            summary_dto, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(rnn_predictions_df, portf_summary.portf_pos_size, portf_summary.trade_comm,portf_summary.n_algo_params)

            return summary_dto, trading_summary_df
        elif tas(portf_summary.trading_algo)==tas.ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG:
            daily_trading_backtester = OnlySignalNMinMovAvgBacktester()
            summary_dto, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(rnn_predictions_df, portf_summary.portf_pos_size, portf_summary.trade_comm,portf_summary.n_algo_params)

            return summary_dto, trading_summary_df

        #InvSlopeBacktester

        else:
            raise Exception(f"NOT RECOGNIZED trading algo {portf_summary.trading_algo}")

    def __wrap_positions_in_summary__(self, algo_name, portf_positions, n_algo_param_dict,
                                      eval_d_from, eval_d_to):
        summary = PortfSummary(
            symbol=portf_positions[0].symbol if portf_positions else "UNKNOWN",
            p_portf_position_size=n_algo_param_dict["init_portf_size"],
            p_trade_comm=n_algo_param_dict["trade_comm"],
            p_trading_algo=algo_name,
            p_algo_params=n_algo_param_dict,
            p_period=DateHandler.get_period_label_from_dates(eval_d_from, eval_d_to),
            p_year=eval_d_from.year
        )

        summary.portf_pos_summary = portf_positions

        # Collect all detailed MTMs (date + MTM) from each portfolio position
        all_MTMs = []
        for pos in portf_positions:
            all_MTMs.extend(pos.detailed_MTMs)

        # Sort MTMs by date
        all_MTMs.sort(key=lambda x: x.date)

        if not all_MTMs:
            summary.daily_profits = []
            summary.max_cum_drawdowns.append(0.0)
            summary.portf_init_MTM = getattr(self, "last_known_portf_value", n_algo_param_dict["init_portf_size"])
            summary.portf_final_MTM = summary.portf_init_MTM
            summary.total_net_profit = 0.0
            summary.total_net_profit_str = "0.0 $"
            LightLogger.do_log(f"[INFO] No positions opened in period {summary.period}. Portfolio unchanged.")
            summary.update_max_drawdown()
            return summary

        summary.daily_profits = [m.MTM for m in all_MTMs]
        summary.max_cum_drawdowns.append(FinancialCalculationsHelper.max_drawdown_on_MTM(summary.daily_profits))
        summary.max_drawdown=FinancialCalculationsHelper.max_drawdown_on_MTM(summary.daily_profits)

        summary.portf_init_MTM = all_MTMs[0].MTM
        summary.portf_final_MTM = all_MTMs[-1].MTM
        summary.total_net_profit = summary.portf_final_MTM - summary.portf_init_MTM
        summary.total_net_profit_str = f"{round(summary.total_net_profit, 2)} $"

        self.last_known_portf_value = summary.portf_final_MTM
        total_profit = summary.portf_final_MTM - summary.portf_init_MTM
        profit_pct = (total_profit / summary.portf_init_MTM) * 100
        drawdown_pct = summary.max_drawdown * 100

        # --- CAGR ---
        days_diff = (eval_d_to - eval_d_from).days
        years = days_diff / 365.0
        cagr = (summary.portf_final_MTM / summary.portf_init_MTM) ** (1 / years) - 1
        cagr_pct = cagr * 100.0

        LightLogger.do_log(
            f"[SUMMARY] Portfolio Positions Breakdown: "
            f"Init={summary.portf_init_MTM:.2f} | "
            f"Final={summary.portf_final_MTM:.2f} | "
            f"Profit={profit_pct:.2f}% | "
            f"CAGR={cagr_pct:.2f}% | "
            f"Max. Drawdown={drawdown_pct:.2f}% | "
        )

        for idx, pos in enumerate(portf_positions):
            pos_MTM=FinancialCalculationsHelper.max_drawdown_on_MTM(pos.daily_MTMs)*100
            LightLogger.do_log(
                f"  Position #{idx + 1}: {pos.side} | open={pos.price_open} | close={pos.price_close} | "
                f"open_date={pos.date_open.date()} | close_date={pos.date_close.date()} | "
                f"profit={pos.calculate_th_nom_profit():.2f} | pct={pos.calculate_pct_profit():.2f}% | Drawdown={pos_MTM:.2f}%"
            )

        summary.update_max_drawdown()

        return summary

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

    def _run_competition_summary_report(self, year, report_type="K10",portfolio=None):
        report = CompetitionSummaryReport(
            year=year,
            logger=self.logger,
            report_type=report_type,
            portfolio=portfolio
        )
        report.run()

    def _run_download_k10(self, year,portfolio):
        base_path = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/{ReportFolder.K10.value}/{year}"
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

    #_run_process_finviz_news

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

    def __backtest_strategy__(self, series_df,indicator,portf_size,
                                    n_algo_params,portf_summary,
                                    etf_comp_dto_arr=None):

        if tas(portf_summary.trading_algo) == tas.RAW_INV_SLOPE:
            backtester=InvSlopeBacktester()
            return backtester.backtest(series_df, indicator, portf_size, n_algo_params,
                                       etf_comp_dto_arr=etf_comp_dto_arr)

        elif tas(portf_summary.trading_algo) == tas.RAW_DIRECT_SLOPE:
            backtester=DirectSlopeBacktester()
            return backtester.backtest(series_df, indicator, portf_size, n_algo_params,
                                       etf_comp_dto_arr=etf_comp_dto_arr)

        elif tas(portf_summary.trading_algo) == tas.RAW_ON_OFF_VALUE:
            backtester=OnOffBacktester()
            return backtester.backtest(series_df, indicator, portf_size, n_algo_params,
                                       etf_comp_dto_arr=etf_comp_dto_arr)

        elif tas(portf_summary.trading_algo) == tas.BUY_AND_HOLD:
            backtester=BuyAndHoldBacktester()
            return backtester.backtest(series_df, indicator, portf_size, n_algo_params,
                                       etf_comp_dto_arr=etf_comp_dto_arr)


        else:
            raise Exception(f"NOT RECOGNIZED trading algo {portf_summary.trading_algo}")

    # Returns prices of Symbol, starting on day, for N timesteps of X interval
    # Ex: If inteval is Day and start day 10/10/2023  and timesteps is 10
    # it returns daily prices from 10/10/2023 to 10/20/2023
    def __build_symbol_series__(self,symbol,day,timesteps,interval):
        start_day_timestamp = day
        start_period = day + pd.offsets.BDay(-1 * (timesteps + 2))  # we go back <timesteps> business days in time
        start_period_all_ind = start_period - timedelta(days=60)
        end_period = start_day_timestamp + timedelta(hours=23, minutes=59, seconds=59)

        symbol_int_series_df = self.data_set_builder.build_interval_series(symbol, start_period_all_ind,
                                                                           end_period, interval=interval,
                                                                           output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        return symbol_int_series_df,start_period_all_ind,start_period,end_period


    def __get_business_days_in_range__(self,d_from,d_to):
        # Generate a date range between d_from and d_to
        all_days = pd.date_range(start=d_from, end=d_to)

        # Filter out weekends (Saturday = 5, Sunday = 6)
        business_days = [day for day in all_days if day.weekday() < 5]

        return  business_days

    def __eval_df_grouping__(self,test_series_df,grouping_unit,variables_csv):

        if grouping_unit is not None:
            test_series_df = self.__group_dataframe__(test_series_df, grouping_unit, variables_csv)
            self.logger.do_log(test_series_df.head(),MessageType.INFO)

        return test_series_df

    def __backtest_scalping__(self,algo,symbol,rnn_predictions_df,symbol_int_series_df):

        max_cum_drawdowns = []
        daily_profits = []
        total_net_profit = 0
        accum_positions = 0

        # TODO Run backtests and evaluate performance
        max_daily_drawdown = 0
        max_total_drawdown = 0

        ml_analyzer = MLModelAnalyzer(self.logger)

        #n-We pre-process the predictions --> chg action->Prediction
        rnn_predictions_df.rename(columns={'action': 'Prediction'}, inplace=True)
        predictions_dic = {}
        predictions_dic[algo] = rnn_predictions_df

        #n-We add the colum <symbol> with the price to use : the 'open' price
        symbol_df = symbol_int_series_df  # TODO see how to best apply only open numbers
        symbol_df[symbol] = symbol_df['open'] #

        #n-We run the backtest
        #TODO - Properly implmeent the backetsting algo
        raise Exception("Backtesting algo not implemented")

        self.logger.do_log(f"---Summarizing PORTFOLIO PERFORMANCE---", MessageType.INFO)
        self.logger.do_log(f" Total Net_Profit=${total_net_profit:.2f} Accum. Positions={accum_positions} Max. Daily Drawdown=${max_daily_drawdown:.2f} Max. Period Drawdown=${max_total_drawdown:.2f}",MessageType.INFO)

        return None

    def process_test_scalping_XGBoost(self, symbol, series_csv, model_to_use, d_from, d_to, n_algo_param_dict):
        """
        Evaluate an XGBoost model over a single period, run backtest, and return summary.
        """

        self.logger.do_log(
            f"Initializing XGBoost backtest for symbol {symbol} from {d_from} to {d_to}",
            MessageType.INFO
        )

        init_portf_size = float(n_algo_param_dict["init_portf_size"])
        pos_regime_filters_csv = n_algo_param_dict.get("pos_regime_filters_csv", None)
        neg_regime_filters_csv = n_algo_param_dict.get("neg_regime_filters_csv", None)
        draw_predictions = n_algo_param_dict.get("draw_predictions", False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_dict_arr = []

        symbol_prices_df = self.data_set_builder.build_interval_series(
            symbol, d_from, d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "open", "high", "low", "close"]
        )

        features_df = self.data_set_builder.build_daily_series_classification(
            series_csv, d_from, d_to, add_classif_col=False,fill_to_max=True
        )
        features_df = DataframeFiller.fill_missing_values(features_df)

        pos_regime_df = self.data_set_builder.build_interval_series(
            pos_regime_filters_csv, d_from, d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "open", "high", "low", "close"]
        ).dropna() if pos_regime_filters_csv else pd.DataFrame()

        neg_regime_df = self.data_set_builder.build_interval_series(
            neg_regime_filters_csv, d_from, d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "open", "high", "low", "close"]
        ).dropna() if neg_regime_filters_csv else pd.DataFrame()

        reg_df = pos_regime_df if not pos_regime_df.empty else neg_regime_df if not neg_regime_df.empty else None
        pos_df = True if not pos_regime_df.empty else False if not neg_regime_df.empty or reg_df is None else None

        ml_analyzer = MLModelAnalyzer(self.logger)

        preds_df, test_features_df = ml_analyzer.evaluate_trading_performance_last_model_XGBoost(
            symbol_df=symbol_prices_df, symbol=symbol, features_df=features_df, model_filename=model_to_use,
            bias=n_algo_param_dict.get("bias", "LONG"), last_trading_dict=None, n_algo_param_dict=n_algo_param_dict,
            draw_statistics=draw_predictions)


        backtester = NFlipPredictionBacktester()
        portf_pos_dict = backtester.backtest(
            symbol=symbol,
            symbol_prices_df=preds_df,
            predictions_dic={"DAILY_XGB": preds_df},
            last_trading_dict=None,
            n_algo_param_dict=n_algo_param_dict,
            init_last_portf_size_dict={"DAILY_XGB": init_portf_size},
            regime_df=reg_df,
            pos_regime=pos_df
        )

        summary = self.__wrap_positions_in_summary__("DAILY_XGB", portf_pos_dict["DAILY_XGB"],
                                                     n_algo_param_dict, d_from, d_to)
        summary.period = f"{d_from.strftime('%b')}-{d_to.strftime('%b')}"

        summary_dict_arr.append({"XGBOOST": summary})

        if draw_predictions:
            symbol_prices_df = symbol_prices_df[symbol_prices_df['date'] >= d_from]
            GraphBuilder.plot_prices_with_trades(symbol_prices_df, summary_dict_arr, "XGBOOST")

        return {"DAILY_XGB": summary}



    def process_test_scalping_RF(self, symbol, series_csv, model_to_use, d_from, d_to, n_algo_param_dict):
        """
        Evaluate a Random Forest model over a single period, run backtest, and return summary.
        """

        self.logger.do_log(
            f"Initializing RF backtest for symbol {symbol} from {d_from} to {d_to}",
            MessageType.INFO
        )

        # Parameters
        init_portf_size = float(n_algo_param_dict["init_portf_size"])
        pos_regime_filters_csv = n_algo_param_dict.get("pos_regime_filters_csv", None)
        neg_regime_filters_csv = n_algo_param_dict.get("neg_regime_filters_csv", None)
        draw_predictions = n_algo_param_dict.get("draw_predictions", False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_dict_arr=[]

        # Load price series of target symbol
        symbol_prices_df = self.data_set_builder.build_interval_series(
            symbol, d_from, d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "open", "high", "low", "close"]
        )

        # Load feature variables for classification
        series_df = self.data_set_builder.build_daily_series_classification(
            series_csv, d_from, d_to, add_classif_col=False
        )
        series_df = DataframeFiller.fill_missing_values(series_df)

        # Optional: Load regime filter series
        pos_regime_df = self.data_set_builder.build_interval_series(
            pos_regime_filters_csv, d_from, d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "open", "high", "low", "close"]
        ).dropna() if pos_regime_filters_csv else pd.DataFrame()

        neg_regime_df = self.data_set_builder.build_interval_series(
            neg_regime_filters_csv, d_from, d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "open", "high", "low", "close"]
        ).dropna() if neg_regime_filters_csv else pd.DataFrame()

        reg_df = pos_regime_df if not pos_regime_df.empty else neg_regime_df if not neg_regime_df.empty else None
        pos_df = True if not pos_regime_df.empty else False if not neg_regime_df.empty or reg_df is None else None

        # Load model and label encoder
        ml_analyzer = MLModelAnalyzer(self.logger)
        label_encoder = ml_analyzer.extract_label_encoder_from_model(model_to_use)

        # Run prediction
        result_df, test_series_df = ml_analyzer.evaluate_trading_performance_last_model_RF(
            symbol_df=symbol_prices_df,
            symbol=symbol,
            series_df=series_df,
            model_filename=model_to_use,
            bias=n_algo_param_dict.get("bias", "LONG"),
            label_encoder=label_encoder,
            last_trading_dict=None,
            n_algo_param_dict=n_algo_param_dict,
            draw_statistics=draw_predictions
        )

        # Backtest predictions
        backtester = NFlipPredictionBacktester()
        portf_pos_dict = backtester.backtest(
            symbol=symbol,
            symbol_prices_df=result_df,
            predictions_dic={"DAILY_RF": result_df},
            last_trading_dict=None,
            n_algo_param_dict=n_algo_param_dict,
            init_last_portf_size_dict={"DAILY_RF": init_portf_size},
            regime_df=reg_df,
            pos_regime=pos_df
        )

        # Generate portfolio summary
        summary = self.__wrap_positions_in_summary__("DAILY_RF", portf_pos_dict["DAILY_RF"],
                                                     n_algo_param_dict, d_from, d_to)
        summary.period = f"{d_from.strftime('%b')}-{d_to.strftime('%b')}"

        summary_dict_arr.append({"RANDOM_FOREST": summary})

        # PortfolioSummaryAnalyzer.convert_summary_dict_arr_to_dataframe(
        #     summary_dict_arr, symbol, series_csv,
        #     strategy_key="RANDOM_FOREST",
        #     init_portf=init_portf_size,
        #     timestamp=timestamp
        # )

        if draw_predictions:
            symbol_prices_df = symbol_prices_df[symbol_prices_df['date'] >= d_from]
            GraphBuilder.plot_prices_with_trades(symbol_prices_df, summary_dict_arr, "RANDOM_FOREST")

        return {"DAILY_RF": summary}

    def train_algos(self,series_csv,d_from,d_to,p_classif_key,algos_arr=None):

        try:
            series_df= self.data_set_builder.build_daily_series_classification(series_csv, d_from, d_to)
            mlAnalyzer=MLModelAnalyzer(self.logger)
            model_base_name,variables_def =MLModelAnalyzer.__build_model_base_name__(series_csv,d_from,d_to,p_classif_key)
            comp_df= mlAnalyzer.fit_and_evaluate(series_df, DataSetBuilder._CLASSIFICATION_COL,model_base_name,variables_def,
                                                 algos_arr=algos_arr)
            return comp_df

        except Exception as e:
            msg="CRITICAL ERROR processing model @train_algos:{}".format(str(e))
            self.logger.do_log(msg,MessageType.ERROR)
            raise Exception(msg)


    def backtest_neural_network_algo(self,symbol, variables_csv,d_from,d_to,model_to_use):
        try:

            #symbol_df = self.data_set_builder.build_series(symbol, d_from, d_to)
            test_series_df = self.data_set_builder.build_daily_series_classification(variables_csv, d_from, d_to)
            nn_trainer = NeuralNetworkModelTrainer(self.logger)

            nn_trainer.run_predictions(test_series_df,DataSetBuilder._CLASSIFICATION_COL,model_to_use)

            return None

        except Exception as e:
            msg = "CRITICAL ERROR processing model @backtest_neural_network_algo:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def train_neural_network(self,symbol, variables_csv,d_from,d_to,depth,learning_rate,epochs,model_output):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(variables_csv, d_from, d_to)
            nn_trainer = NeuralNetworkModelTrainer(self.logger)
            nn_trainer.train_neural_network(series_df,variables_csv,DataSetBuilder._CLASSIFICATION_COL,depth,learning_rate,epochs,model_output)
            return None

        except Exception as e:
            msg = "CRITICAL ERROR processing model @train_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def sliding_train_and_evaluate_ml_performance(self, symbol, series_csv, d_from, d_to, bias, last_trading_dict=None,
                                                  n_algo_param_dict={}):
        """
        Train and evaluate ML performance using a sliding window approach.
        Asynchronously invoke calculate_portfolio_metrics after each window to update results incrementally.

        Args:
            symbol (str): Symbol for trading.
            series_csv (str): Path to the CSV file with series data.
            d_from (datetime): Start date.
            d_to (datetime): End date.
            bias (float): Bias parameter.
            last_trading_dict (dict, optional): Last trading dictionary.
            n_algo_param_dict (dict): Parameters for the algorithm.

        Returns:
            list: List of summary dictionaries for each window.
        """
        summary_dict_arr = []
        summary_dict = None
        sliding_window_years = int(n_algo_param_dict["sliding_window_years"])
        sliding_window_months = int(n_algo_param_dict["sliding_window_months"])
        classif_key = n_algo_param_dict["classif_key"]
        init_portf_size = n_algo_param_dict["init_portf_size"]

        bias_df=None
        if bias is None or bias !=SlidingWindowStrategy.NONE.value:
            bias_df = self.data_set_builder.get_classification_df(bias,d_from,d_to)

        # Validate inputs
        if d_from > d_to:
            raise ValueError("d_from must be earlier than d_to.")
        if sliding_window_years < 0 or sliding_window_months <= 0:
            raise ValueError("sliding_window_years must be non-negative and sliding_window_months must be positive.")

        # Generate a single timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize the list to store the windows
        windows = []

        algos_arr=n_algo_param_dict["algos"].split(",") if n_algo_param_dict["algos"] is not None else None

        # Start with the first window
        curr_d_from = d_from
        # Calculate the end of the first window by adding the window duration
        curr_d_to = curr_d_from + relativedelta(years=sliding_window_years) - relativedelta(days=1)

        # Ensure the first window's end does not exceed d_to
        if curr_d_to > d_to:
            curr_d_to = d_to

        # Loop until the window's start date exceeds d_to
        while curr_d_from <= d_to:
            # Add the current window to the list
            windows.append((curr_d_from, curr_d_to))

            # 1- We train the algos in the train window
            self.train_algos(series_csv, curr_d_from, curr_d_to, classif_key,algos_arr=algos_arr)

            eval_d_from = curr_d_to + relativedelta(days=1)
            eval_d_to = eval_d_from + relativedelta(months=sliding_window_months) - relativedelta(days=1)

            if bias_df is not None:
                bias = self.data_set_builder.get_classification_in_classif_df(bias_df,eval_d_from)
                bias= bias  if bias is not None else SlidingWindowStrategy.NONE.value

            summary_dict = self.evaluate_trading_performance(
                symbol, series_csv, eval_d_from, eval_d_to,
                last_trading_dict=summary_dict,
                bias=bias,
                n_algo_param_dict=n_algo_param_dict,
                algos_arr=algos_arr
            )
            summary_dict_arr.append(summary_dict)

            # Asynchronously invoke calculate_portfolio_metrics after each window
            PortfolioSummaryAnalyzer.calculate_portfolio_metrics(summary_dict_arr, init_portf_size, eval_d_from, eval_d_to, timestamp)

            # Slide the window forward by sliding_window_months
            curr_d_from = curr_d_from + relativedelta(months=sliding_window_months)
            # Recalculate the end date for the new window
            curr_d_to = curr_d_from + relativedelta(years=sliding_window_years) - relativedelta(days=1)

            # Ensure the end date does not exceed d_to
            if curr_d_to > d_to:
                curr_d_to = d_to

            # Break if the window's end date is not advancing (i.e., we're at the end)
            if curr_d_to <= windows[-1][1]:
                break

        return summary_dict_arr

    def sliding_train_and_evaluate_random_forest_performance(self, symbol, series_csv, d_from, d_to,
                                                             last_trading_dict=None, n_algo_param_dict=None):
        """
        Train and evaluate a Random Forest model using a sliding window approach.
        """
        summary_dict_arr = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        sliding_window_years = int(n_algo_param_dict["sliding_window_years"])
        sliding_window_months = int(n_algo_param_dict["sliding_window_months"])
        init_portf_size = float(n_algo_param_dict["init_portf_size"])
        classif_key = n_algo_param_dict["classif_key"]
        pos_regime_filters_csv=n_algo_param_dict["pos_regime_filters_csv"]
        neg_regime_filters_csv = n_algo_param_dict["neg_regime_filters_csv"]
        draw_predictions = n_algo_param_dict["draw_predictions"]

        idx_loop=0
        init_last_portf_size_dict=None
        class_weight = n_algo_param_dict["class_weight"]

        curr_d_from = d_from
        curr_d_to = curr_d_from + relativedelta(years=sliding_window_years) - relativedelta(days=1)
        draw_d_from=curr_d_to
        if curr_d_to > d_to:
            curr_d_to = d_to

        os.makedirs(Folders.OUTPUT_RF_FOLDER.value, exist_ok=True)

        mlAnalyzer = MLModelAnalyzer(self.logger)

        symbol_prices_df = self.data_set_builder.build_interval_series(
            symbol, d_from, d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "open", "high", "low", "close"]
        )

        while curr_d_from <= d_to:

            if curr_d_from > d_to:
                self.logger.do_log(
                    f"Stopping loop: curr_d_from {curr_d_from.strftime('%Y-%m-%d')} > d_to {d_to.strftime('%Y-%m-%d')}",
                    MessageType.INFO)
                break

            model_filename = f"{Folders.OUTPUT_RF_FOLDER.value}/{FilenamePrefix.RF_MODEL.value }{symbol}_{curr_d_from.strftime('%Y%m%d')}_{curr_d_to.strftime('%Y%m%d')}.pkl"

            self.logger.do_log(f"Training RF from {curr_d_from} to {curr_d_to} → {model_filename}", MessageType.INFO)

            label_encoder= self.process_train_RF(symbol=symbol, series_csv=series_csv,
                                                 d_from=curr_d_from.strftime('%m/%d/%Y'),
                                                 d_to=curr_d_to.strftime('%m/%d/%Y'), model_output=model_filename,
                                                 classif_key=classif_key, n_estimators=200, max_depth=5,
                                                 min_samples_split=10, criterion="gini",
                                                 interval=DataSetBuilder._1_DAY_INTERVAL, make_stationary=True,
                                                 class_weight=class_weight)

            eval_d_from = curr_d_to + relativedelta(days=1)
            eval_d_to = eval_d_from + relativedelta(months=sliding_window_months) - relativedelta(days=1)
            if eval_d_to > d_to:
                eval_d_to = d_to

            self.logger.do_log(f"Evaluating RF from {eval_d_from} to {eval_d_to}", MessageType.INFO)

            n_algo_param_dict["series_csv"] = series_csv

            pos_regime_df = self.data_set_builder.build_interval_series(
                pos_regime_filters_csv, eval_d_from, eval_d_to,
                interval=DataSetBuilder._1_DAY_INTERVAL,
                output_col=["symbol", "date", "open", "high", "low", "close"]
            ).dropna()

            neg_regime_df = self.data_set_builder.build_interval_series(
                neg_regime_filters_csv, eval_d_from, eval_d_to,
                interval=DataSetBuilder._1_DAY_INTERVAL,
                output_col=["symbol", "date", "open", "high", "low", "close"]
            ).dropna()

            if eval_d_from > d_to:
                self.logger.do_log(f"Skipping out-of-bound eval window: {eval_d_from} to {eval_d_to}", MessageType.INFO)
                break

            series_df = self.data_set_builder.build_daily_series_classification(series_csv, eval_d_from, eval_d_to,
                                                                                add_classif_col=False)
            series_df = DataframeFiller.fill_missing_values(series_df)

            result_df, test_series_df = mlAnalyzer.evaluate_trading_performance_last_model_RF(
                symbol_df=symbol_prices_df,
                symbol=symbol,
                series_df=series_df,
                label_encoder=label_encoder,
                model_filename=model_filename,
                bias=None,
                last_trading_dict=last_trading_dict,
                n_algo_param_dict=n_algo_param_dict,
                draw_statistics=draw_predictions
            )

            # If not the first iteration, carry forward the final MTM from the previous summary
            if idx_loop > 0:
                prev_summary = summary_dict_arr[-1]["SLIDING_RF"]  # last summary from previous window
                init_last_portf_size_dict={}
                init_last_portf_size_dict["SLIDING_RF"] = prev_summary.portf_final_MTM


            # Convert predictions to portfolio positions using backtester
            reg_df= pos_regime_df if not pos_regime_df.empty else neg_regime_df if not neg_regime_df.empty else None
            pos_df = True if not pos_regime_df.empty else False if not neg_regime_df.empty or reg_df is None else None

            backtester = NFlipPredictionBacktester()
            portf_pos_dict = backtester.backtest(
                                                    symbol=symbol,
                                                    symbol_prices_df=result_df,
                                                    predictions_dic={"SLIDING_RF": result_df},
                                                    last_trading_dict=last_trading_dict,
                                                    n_algo_param_dict=n_algo_param_dict,
                                                    init_last_portf_size_dict=init_last_portf_size_dict,
                                                    regime_df=reg_df,
                                                    pos_regime=pos_df
                                                )
            idx_loop+=1
            # Wrap results into summary
            summary = self.__wrap_positions_in_summary__("SLIDING_RF", portf_pos_dict["SLIDING_RF"],
                                                         n_algo_param_dict, eval_d_from,eval_d_to)
            summary.period = f"{eval_d_from.strftime('%b')}-{eval_d_to.strftime('%b')}"
            summary_dict_arr.append({"SLIDING_RF": summary})

            PortfolioSummaryAnalyzer.convert_summary_dict_arr_to_dataframe(
                summary_dict_arr,symbol, series_csv,
                strategy_key="SLIDING_RF",
                init_portf=init_portf_size,
                timestamp=timestamp
            )

            curr_d_from = curr_d_from + relativedelta(months=sliding_window_months)
            curr_d_to = curr_d_from + relativedelta(years=sliding_window_years) - relativedelta(days=1)
            if curr_d_to > d_to:
                curr_d_to = d_to
            if curr_d_to <= curr_d_from:
                break

            if draw_predictions:
                symbol_prices_df = symbol_prices_df[symbol_prices_df['date'] >= draw_d_from]
                GraphBuilder.plot_prices_with_trades(symbol_prices_df,summary_dict_arr,"SLIDING_RF")


        return summary_dict_arr

    def evaluate_trading_performance(self,symbol,series_csv,d_from,d_to,bias,last_trading_dict=None,
                                     n_algo_param_dict={},algos_arr=None):

        try:
            symbol_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to,add_classif_col=False)
            series_df = self.data_set_builder.build_daily_series_classification(series_csv, d_from, d_to,add_classif_col=False)

            # Given different frequencies of the data, we will the missing <interval> values with the last available data
            series_df = DataframeFiller.fill_missing_values(series_df)  # We fill missing values with the last one

            mlAnalyzer = MLModelAnalyzer(self.logger)
            portf_pos_dict = mlAnalyzer.evaluate_trading_performance_last_model(symbol_df,symbol,
                                                                                series_df, bias,
                                                                                last_trading_dict,
                                                                                n_algo_param_dict,
                                                                                algos_arr=algos_arr)

            backtester=IndicatorBasedTradingBacktester()

            summary_dict={}
            for algo in portf_pos_dict.keys():
                port_positions_arr=portf_pos_dict[algo]
                if len(port_positions_arr)>0:
                    summary= backtester.calculate_portfolio_performance_summary_extended(symbol,port_positions_arr,ref_date=d_from)
                    summary_dict[algo]=summary
                else:
                    summary=PortfSummary(symbol=symbol,p_portf_position_size=n_algo_param_dict["init_portf_size"],
                                         p_trade_comm=0,p_trading_algo=algo,
                                         p_algo_params=n_algo_param_dict["init_portf_size"],
                                         p_period=DateHandler.get_two_month_period_from_date(d_from),
                                         p_year=d_from.year)
                    summary_dict[algo] = summary

            return summary_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @evaluate_trading_performance:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def run_predictions_last_model(self,series_csv,d_from,d_to):

        try:
            series_df = self.data_set_builder.build_daily_series_classification(series_csv, d_from, d_to,
                                                                                add_classif_col=False)
            mlAnalyzer = MLModelAnalyzer(self.logger)
            pred_dict = mlAnalyzer.run_predictions_last_model(series_df)
            return pred_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @run_predictions_last_model:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def build_ARIMA(self,symbol, d_from, d_to, period=None):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to,
                                                                                add_classif_col=False)
            arima_Analyzer = ARIMAModelsAnalyzer(self.logger)
            dickey_fuller_test_dict=arima_Analyzer.build_ARIMA_model(series_df,symbol,period,True)
            return dickey_fuller_test_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @build_ARIMA:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def eval_singe_indicator_algo(self,symbol,indicator, inv, d_from, d_to):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to,
                                                                                add_classif_col=False)

            indic_classif_list = self.date_range_classif_mgr.get_date_range_classification_values(indicator,d_from,d_to)
            indic_classif_df = pd.DataFrame([vars(classif) for classif in indic_classif_list])

            backtester = IndicatorBasedTradingBacktester()
            return backtester.backtest_indicator_based_strategy(symbol,series_df,indic_classif_df,inv)


        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR processing model @eval_singe_indicator_algo:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def eval_ml_biased_algo(self,symbol, indicator,seriesCSV,d_from,d_to,inverted):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(seriesCSV, d_from, d_to,
                                                                                add_classif_col=False)

            indic_classif_list = self.date_range_classif_mgr.get_date_range_classification_values(indicator, d_from,
                                                                                                  d_to)
            indic_classif_df = pd.DataFrame([vars(classif) for classif in indic_classif_list])

            mlAnalyzer = MLModelAnalyzer(self.logger)

            pred_dict = mlAnalyzer.run_predictions_last_model(series_df)

            backtester = IndicatorBasedTradingBacktester()

            return backtester.backtest_ML_indicator_biased_strategy(symbol,series_df,indic_classif_df,inverted,pred_dict)

        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR processing model @eval_ml_biased_algo:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def predict_ARIMA(self,symbol, p,d,q,d_from,d_to, steps,period=None):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to,
                                                                                add_classif_col=False)
            arima_Analyzer = ARIMAModelsAnalyzer(self.logger)
            preds=arima_Analyzer.build_and__predict_ARIMA_model(series_df,symbol,p,d,q,period,steps)
            return preds

        except Exception as e:
            msg = "CRITICAL ERROR processing model @predict_ARIMA:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def train_convolutional_neural_network(self,train_true_path,train_false_path,test_true_path,test_false_path,true_label,arch_file,padding,stride,iterations):
        try:
            LightLogger.do_log("Extracting images from true and false paths")

            handler = ImageHandler()

            LightLogger.do_log("Fetching images from true path {} and false path {} ".format(train_true_path, train_false_path))

            train_x, train_y, image_idx_train = handler.create_non_vect_sets(train_true_path, train_false_path, true_label, ".jpg")

            test_x, test_y, image_idx_test = handler.create_non_vect_sets(test_true_path, test_false_path, true_label, ".jpg")

            cnn= ConvolutionalNeuralNetwork()

            cnn.train_model(train_x, train_y,test_x, test_y,arch_file,padding,stride,iterations)


            #TODO finish everything
            #raise ("NOT FINISHED")

        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR @train_convolutional_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def train_deep_neural_network(self,true_path,false_path,true_label,learning_rate=0.075,num_iterations=2500,
                                  arch_file=None, activ_file=None,output_file=None,step_size=200,lambd=0,use_He_init=False):


        try:

            LightLogger.do_log("Extracting images from true and false paths")

            handler = ImageHandler()

            offset=0
            end=False

            neural_network = DeepNeuralNetwork()

            parameters=None
            activations = neural_network.build_activations(activ_file)
            index=0

            while not end  : #we prepare the batches

                LightLogger.do_log("Fetching images for step {}".format(index))

                train_x,train_y,image_idx=handler.create_sets(true_path,false_path,true_label,".jpg",offset,step_size)

                if(len(train_x)< (step_size*2) ): #we have less than en enough rows for the available step --> we got to the end
                    end=True #this is the last run

                if len(train_x)>0:

                    LightLogger.do_log("Extracted {} train examples".format(len(image_idx)))

                    layers_dims=neural_network.build_layers_dims(len(train_x),arch_file) # len(train_x)--> ints in flattened vectors--> ex: 122880

                    LightLogger.do_log("Training Network with Learning Rate={} and num_iterations={}".format(learning_rate,num_iterations))
                    parameters, costs =neural_network.L_layer_model_train(train_x,train_y,layers_dims,activations,learning_rate=learning_rate,
                                                                          num_iterations=num_iterations,print_cost=True,parameters=parameters,loop=index,
                                                                          lambd=lambd,use_He_init=use_He_init)

                offset+=1
                index+=1


            #We persist everything

            if output_file is not None:
                neural_network.persist_parameters(parameters,activations,output_file)#parameters es el modelo!
                LightLogger.do_log("Model successfully persisted at {}".format(output_file))

                #test retreive parameters
                paramters2= neural_network.retrieve_parameters(output_file)
                LightLogger.do_log("Successfully retreived {} for testing".format(output_file))



        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR @train_deep_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def test_deep_neural_network_model(self,true_path,false_path,true_label,output_file):
        try:
            LightLogger.do_log("Extracting model from file {}".format(output_file))

            neural_network = DeepNeuralNetwork()

            parameters = neural_network.retrieve_parameters(output_file)
            LightLogger.do_log("Successfully retrieved {} for testing".format(output_file))

            LightLogger.do_log("Extracting tests sets from true path={} and false paths={}".format(true_path,false_path))
            handler = ImageHandler()
            test_x, test_y, image_idx = handler.create_sets(true_path, false_path, true_label, ".jpg")

            accuracy=neural_network.L_layer_model_test(test_x,test_y,image_idx,parameters,parameters["activations"])

            LightLogger.do_log("Found an accuracy of {} for {} test instances".format(accuracy,test_x.shape[1]))

            return  accuracy


        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR @test_deep_neural_network_model:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def process_test_scalping_LSTM(self,symbol,variables_csv, model_to_use, d_from,d_to,timesteps,portf_size, trade_comm,
                                trading_algo,interval=None,grouping_unit=None,n_algo_params=[],
                                   make_stationary=True,classif_threshold=0.5):
        try:

            self.logger.do_log(f"Initializing backest for symbol {symbol} from {d_from} to {d_to} (porft_size={portf_size} comm={trade_comm} )", MessageType.INFO)

            rnn_model_processer = DayTradingRNNModelCreator()
            rnn_predictions_df=None
            states = None
            for day in self.__get_business_days_in_range__(d_from,d_to):
                self.logger.do_log(f"Processing day {day}  )",MessageType.INFO)
                #1- We get the date range of prices to use to predict day <day> AND the symbol prices of that period
                symbol_int_series_df,start_period_all_ind,start_period,end_period= self.__build_symbol_series__(symbol,day,timesteps,interval)

                if symbol_int_series_df is None:
                    self.logger.do_log(f"Skipping day {day} because missing values (probable holiday!)",MessageType.WARNING)
                    continue

                #2- We get the variables (features) of that period
                variables_int_series_df = self.data_set_builder.build_interval_series(variables_csv,
                                                                                      start_period_all_ind, end_period,
                                                                                      interval=interval,
                                                                                      output_col=["symbol", "date",
                                                                                                  "open",
                                                                                                  "high", "low",
                                                                                                  "close"])
                #3- We merge everything in one single dataframe
                test_series_df = self.data_set_builder.merge_series(symbol_int_series_df, variables_int_series_df,
                                                                    "symbol", "date", symbol)

                #4- Given different frequencies of the data, we will the missing <interval> values with the last available data
                test_series_df = DataframeFiller.fill_missing_values(test_series_df)#We fill missing values with the last one

                #5- We evaluate if grouping must be applied
                test_series_df=self.__eval_df_grouping__(test_series_df,grouping_unit,variables_csv)

                #we filter the unecessary in all record to fetch all the indicators
                test_series_df = test_series_df[test_series_df['date'] >= start_period ]

                if(test_series_df["trading_symbol"].isna().any()):
                    continue # must be a holiday

                rnn_predictions_df_today,states = rnn_model_processer.test_LSTM_scalping(symbol, test_series_df,
                                                                                         model_to_use, timesteps,
                                                                                         prev_states=states,
                                                                                         make_stationary=make_stationary,
                                                                                         variables_csv=variables_csv,
                                                                                         threshold=classif_threshold)
                if rnn_predictions_df is None:
                    rnn_predictions_df = pd.DataFrame(columns=rnn_predictions_df_today.columns).astype(rnn_predictions_df_today.dtypes)
                rnn_predictions_df_today = rnn_predictions_df_today[rnn_predictions_df_today['date'] == day]
                rnn_predictions_df = pd.concat([rnn_predictions_df, rnn_predictions_df_today], ignore_index=True)

            #6- We have the predictions --> Lest run the backtest
            self.__backtest_scalping__("LSTM_SCALPING",symbol,rnn_predictions_df,symbol_int_series_df)


        except Exception as e:
            msg = "CRITICAL ERROR processing model @process_test_daily_LSTM:{}".format(str(e))
            traceback.print_exc()
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def __process_LSTM_day_single_run__(self,rnn_model_processer,model_to_use,test_series_df,timesteps,day,symbol,
                                        variables_csv=None):

        rnn_predictions_today_df, states = rnn_model_processer.test_LSTM_daily(symbol=symbol,
                                                                               test_series_df=test_series_df,
                                                                               model_to_use=model_to_use,
                                                                               timesteps=timesteps,
                                                                               price_to_use="close",
                                                                               variables_csv=variables_csv)
        self.logger.do_log(f"Predicting ALL TRADES for day {day} and symbol {symbol}",MessageType.INFO)
        return rnn_predictions_today_df


    def __process_LSTM_day_as_sliding_window__(self,rnn_model_processer,model_to_use,test_series_df,timesteps,day,symbol):
        rnn_predictions_today_df = None
        #preloaded_model = rnn_model_processer.preload_model(model_to_use=model_to_use)
        states=None
        for i, window in enumerate(self.__sliding_window__(test_series_df, timesteps)):
            test_series_curr_window_df = window.copy()
            min_timestamp = test_series_curr_window_df["date"].min()
            max_timestamp = test_series_curr_window_df["date"].max()
            self.logger.do_log(f"Processing Window from {min_timestamp} to {max_timestamp} for day {day} for symbol {symbol}",MessageType.INFO)

            if rnn_predictions_today_df is None:  # we initialize the summarization df
                rnn_predictions_today_df = pd.DataFrame(columns=test_series_curr_window_df.columns).astype(
                    test_series_curr_window_df.dtypes)

            rnn_predictions_curr_window_df,states = rnn_model_processer.test_LSTM_daily(symbol,
                                                                                           test_series_curr_window_df,
                                                                                           model_to_use=model_to_use,
                                                                                           timesteps=timesteps,
                                                                                           price_to_use="close",
                                                                                           prev_states=states)

            pred_action = rnn_predictions_curr_window_df["action"].iloc[0]
            curr_mkt_price = rnn_predictions_curr_window_df["trading_symbol_price"].iloc[0]
            end_of_timestamp = rnn_predictions_curr_window_df["date"].iloc[0]
            self.logger.do_log(
                f"Predicting at {pred_action} at end of {end_of_timestamp} at current mkt price={curr_mkt_price} for symbol {symbol}",
                MessageType.INFO)

            rnn_predictions_today_df = pd.concat([rnn_predictions_today_df, rnn_predictions_curr_window_df],ignore_index=True)
        return rnn_predictions_today_df


    #
    def __process_LSTM_day_as_fill_fake_data__(self,rnn_model_processer,model_to_use,test_series_df,timesteps,
                                               day,symbol,variables_csv):
        rnn_predictions_today_df = None
        #preloaded_model = rnn_model_processer.preload_model(model_to_use=model_to_use)
        states=None
        for i, window in enumerate(self.__sliding_window__(test_series_df, timesteps)):
            test_series_curr_window_df = window.copy()
            min_window_timestamp = test_series_curr_window_df["date"].min()
            max_window_timestamp = test_series_curr_window_df["date"].max()
            max_df_timestamp=test_series_df["date"].max()

            curr_min_df= test_series_df[test_series_df['date'] <= max_window_timestamp]
            self.logger.do_log(f"Processing Window from {min_window_timestamp} to {max_window_timestamp} for day {day} for symbol {symbol}", MessageType.INFO)

            # We fill with random values
            curr_min_df = RandomWalkGenerator.__fill_with_random_walk_values__(symbol, curr_min_df,
                                                                               max_window_timestamp,
                                                                               max_df_timestamp,
                                                                               variables_csv)
            if rnn_predictions_today_df is None:  # we initialize the summarization df
                rnn_predictions_today_df =  pd.DataFrame(columns=['trading_symbol', 'date', 'formatted_date', 'action'])

            rnn_predictions_curr_window_df,_ = rnn_model_processer.test_LSTM_daily(symbol, curr_min_df,
                                                                                      model_to_use=model_to_use,
                                                                                      timesteps=timesteps,
                                                                                      price_to_use="close",
                                                                                      prev_states=states,
                                                                                      variables_csv=variables_csv)

            pred_action = rnn_predictions_curr_window_df["action"].iloc[-1]
            curr_mkt_price = rnn_predictions_curr_window_df["trading_symbol_price"].iloc[-1]
            end_of_timestamp = rnn_predictions_curr_window_df["date"].iloc[-1]
            self.logger.do_log(f"{datetime.now()}-Predicting at {pred_action} at end of {end_of_timestamp} at current mkt price={curr_mkt_price} for symbol {symbol}",MessageType.INFO)

            rnn_predictions_today_df = pd.concat([rnn_predictions_today_df, rnn_predictions_curr_window_df.iloc[-1].to_frame().T],ignore_index=True)
        return rnn_predictions_today_df

    def __process_LSTM_day_as_cum_sliding_window__(self, rnn_model_processer, model_to_use, test_series_df,
                                                   timesteps, day, symbol, variables_csv,
                                                   warmup_minutes=60):
        rnn_predictions_today_df = None
        warmup_seconds = warmup_minutes * 60
        states = None

        for i, window in enumerate(self.__sliding_window__(test_series_df, timesteps)):
            test_series_curr_window_df = window.copy()
            min_timestamp = test_series_curr_window_df["date"].min()
            max_timestamp = test_series_curr_window_df["date"].max()

            # Skip predictions if not enough time has passed (warm-up period)
            if (max_timestamp - min_timestamp).total_seconds() < warmup_seconds:
                self.logger.do_log(f"Skipping prediction from {min_timestamp} to {max_timestamp} (warm-up)",
                                   MessageType.INFO)
                continue

            curr_min_df = test_series_df[test_series_df['date'] <= max_timestamp].tail(timesteps)
            self.logger.do_log(
                f"Processing Window from {min_timestamp} to {max_timestamp} for day {day} for symbol {symbol}",
                MessageType.INFO)

            if rnn_predictions_today_df is None:
                rnn_predictions_today_df = pd.DataFrame(columns=['trading_symbol', 'date', 'formatted_date', 'action'])

            rnn_predictions_curr_window_df = rnn_model_processer.test_cumulative_window_LSTM(
                symbol=symbol,
                test_series_df=curr_min_df,
                model_to_use=model_to_use,
                timesteps=timesteps,
                price_to_use="close",
                variables_csv=variables_csv
            )

            pred_action = rnn_predictions_curr_window_df["action"].iloc[-1]
            curr_mkt_price = rnn_predictions_curr_window_df["trading_symbol_price"].iloc[-1]
            end_of_timestamp = rnn_predictions_curr_window_df["date"].iloc[-1]

            self.logger.do_log(
                f"Predicting {pred_action} at end of {end_of_timestamp} at current mkt price={curr_mkt_price} for symbol {symbol}",
                MessageType.INFO
            )

            rnn_predictions_today_df = pd.concat(
                [rnn_predictions_today_df, rnn_predictions_curr_window_df.iloc[-1].to_frame().T],
                ignore_index=True
            )

        return rnn_predictions_today_df

    def __run_LSTM_daily_backtest__(self,day,rnn_predictions_today_df,portf_summary):
        self.logger.do_log(f"Backtesting all day {day} for symbol {portf_summary.symbol}", MessageType.INFO)
        summary_dto, trading_summary_df = self.__backtest_daily_strategy__(
                                                                                        rnn_predictions_today_df, portf_summary)

        #portf_summary.max_cum_drawdowns.append(max_daily_cum_drawdown)
        portf_summary.daily_profits.append(summary_dto.daily_net_profit)
        portf_summary.accum_positions += summary_dto.total_positions
        self.__log_day_trading_results__(day, summary_dto.daily_net_profit, summary_dto.total_positions, trading_summary_df)
        self.logger.do_log(f"Moving to the next business day from {day}", MessageType.INFO)
        return  summary_dto.daily_net_profit

    def process_test_daily_LSTM(self,symbol,variables_csv, model_to_use, d_from,d_to,timesteps,portf_size, trade_comm,
                                trading_algo,interval=None,grouping_unit=None,n_algo_params=[],
                                use_sliding_window=None,make_stationary=True):
        try:

            self.logger.do_log(f"Initializing backest for symbol {symbol} from {d_from} to {d_to} (porft_size={portf_size} comm={trade_comm} )", MessageType.INFO)
            # Generate a date range between d_from and d_to
            all_days = pd.date_range(start=d_from, end=d_to)

            # Filter out weekends (Saturday = 5, Sunday = 6)
            business_days = [day for day in all_days if day.weekday() < 5]
            rnn_model_processer = DayTradingRNNModelCreator()

            portf_summary = PortfSummary(symbol,portf_size,p_trade_comm= trade_comm,
                                         p_trading_algo= trading_algo,p_algo_params=n_algo_params)
            for day in business_days:
                self.logger.do_log(f"Processing day {day}  )",MessageType.INFO)

                start_timestamp=day if (d_from.hour == 0 and d_from.minute == 0 and d_to.second == 0) else d_from
                end_timestamp= (start_timestamp + timedelta(hours=23, minutes=59, seconds=59)) if (d_to.hour == 0 and d_to.minute == 0 and d_to.second == 0) else d_to

                symbol_int_series_df = self.data_set_builder.build_interval_series(portf_summary.symbol, start_timestamp,
                                                                                   end_timestamp, interval=interval,
                                                                                   output_col=["symbol", "date", "open",
                                                                                               "high", "low", "close"])

                if symbol_int_series_df is None or symbol_int_series_df.shape[0] <= 1:
                    self.logger.do_log(f"Skipping day {day} because missing values (probable holiday!)",MessageType.WARNING)
                    continue


                variables_int_series_df = self.data_set_builder.build_interval_series(variables_csv, start_timestamp,
                                                                                      end_timestamp, interval=interval,
                                                                                      output_col=["symbol", "date",
                                                                                                  "open",
                                                                                                  "high", "low",
                                                                                                  "close"])

                test_series_df = self.data_set_builder.merge_series(symbol_int_series_df, variables_int_series_df,
                                                                    "symbol", "date", portf_summary.symbol)

                if grouping_unit is not None:
                    test_series_df = self.__group_dataframe__(test_series_df, grouping_unit,variables_csv)
                    print((test_series_df.head()))

                #preprocess before the predictions
                test_series_df = test_series_df.dropna(subset=[portf_summary.symbol])

                if sws(use_sliding_window)==sws.CUT_INPUT_DF :#slower, but we only pass every <timestemps> records to make sure of the accuracy of the prediction
                    #rnn_predictions_today_df=self.__process_LSTM_day_as_sliding_window__(rnn_model_processer,model_to_use,test_series_df,timesteps,day,symbol)
                    rnn_predictions_today_df = self.__process_LSTM_day_as_cum_sliding_window__(rnn_model_processer,model_to_use, test_series_df,timesteps, day, portf_summary.symbol,variables_csv=variables_csv)
                elif sws(use_sliding_window)==sws.NONE:#faster, but we pass the dataframe with ALL the daily records, which might create some look ahead bias
                    rnn_predictions_today_df=self.__process_LSTM_day_single_run__(rnn_model_processer,model_to_use,test_series_df,timesteps,day,portf_summary.symbol,variables_csv=variables_csv)
                elif sws(use_sliding_window)==sws.GET_FAKE_DATA:
                    rnn_predictions_today_df = self.__process_LSTM_day_as_fill_fake_data__(rnn_model_processer,model_to_use,test_series_df,timesteps, day, portf_summary.symbol,variables_csv= variables_csv)
                else:
                    raise  Exception(f"Could not find use_sliding_window param {use_sliding_window}")

                portf_summary.total_net_profit+=self.__run_LSTM_daily_backtest__(day, rnn_predictions_today_df,
                                                                                portf_summary)

                new_position_summary=portf_summary.calculate_last_portf_position_summary(day)#We append the day summary

                self.logger.do_log(f"PERFORMANCE--> day {new_position_summary.day}: Profit:{new_position_summary.day_nom_profit:2f} ( drawdown={new_position_summary.day_drawdown:2f})", MessageType.INFO)



            portf_summary.update_max_drawdown()
            self.logger.do_log(f"---Re Displaying Daily PERFORMANCE---", MessageType.INFO)
            for pos_summary in portf_summary.portf_pos_summary:
                self.logger.do_log(f" day {pos_summary.day}: Profit:{pos_summary.day_nom_profit:2f} ( drawdown={pos_summary.day_drawdown:2f})", MessageType.INFO)
            self.logger.do_log(f"---------------------------------------", MessageType.INFO)
            self.logger.do_log(f"---Summarizing PORTFOLIO PERFORMANCE---", MessageType.INFO)
            self.logger.do_log(
                f" Total Net_Profit=${portf_summary.total_net_profit:.2f} Accum. Positions={portf_summary.accum_positions} Max. Daily Drawdown=${portf_summary.max_daily_drawdown:.2f} Max. Period Drawdown=${portf_summary.max_drawdown:.2f}",
                MessageType.INFO)




        except Exception as e:
            msg = "CRITICAL ERROR processing model @process_test_daily_LSTM:{}".format(str(e))
            traceback.print_exc()
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def process_train_XGBoost(self, symbol, series_csv, d_from, d_to, model_output, classif_key,
                              n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
                              interval=DataSetBuilder._1_MIN_INTERVAL,
                              grouping_unit=None, grouping_classif_criteria=None,
                              group_as_mov_avg=False, grouping_mov_avg_unit=20,
                              make_stationary=False, class_weight=None):
        try:
            # Retrieve classification ranges based on interval
            range_clasifs = None
            if interval == DataSetBuilder._1_MIN_INTERVAL:
                range_clasifs = self.timestamp_range_classif_mgr.get_timestamp_range_classification_values(
                    classif_key, d_from, d_to)
            elif interval == DataSetBuilder._1_DAY_INTERVAL:
                range_clasifs = self.date_range_classif_mgr.get_date_range_classification_values(
                    classif_key, d_from, d_to)
            else:
                raise Exception(f'Invalid interval at process_train_XGBoost: {interval}')

            # Build main symbol time series
            symbol_min_series_df = self.data_set_builder.build_interval_series(
                symbol, d_from, d_to, interval=interval,
                output_col=["symbol", "date", "open", "high", "low", "close"]
            )

            # Add classification labels to symbol series
            symbol_min_series_df = self.data_set_builder.build_minute_series_classification(
                range_clasifs,
                symbol_min_series_df,
                classif_col_name=classif_key,
                not_found_clasif="FLAT"
            )

            # Load feature variables
            variables_min_series_df = self.data_set_builder.build_interval_series(
                series_csv, d_from, d_to,
                interval=interval,
                output_col=["symbol", "date", "open", "high", "low", "close"]
            )

            # Merge features and labels into final dataset
            training_series_df = self.data_set_builder.merge_series(
                symbol_min_series_df, variables_min_series_df, "symbol", "date", symbol
            )

            # Optional: compute moving averages
            if group_as_mov_avg:
                training_series_df = self.data_set_builder.group_as_mov_avgs(
                    training_series_df, series_csv, grouping_mov_avg_unit
                )

            # Optional: apply grouping logic (e.g., weekly bars)
            if grouping_unit is not None:
                training_series_df = self.__group_dataframe__(
                    training_series_df, grouping_unit,
                    series_csv, grouping_classif_criteria, classif_key
                )

            # Logging for inspection
            DataframePrinter.print_dataframe_head_values_w_time(
                variables_min_series_df, "symbol", series_csv, 10, "date", "10:30:00"
            )
            DataframePrinter.print_data_farme_head(training_series_df, 10)

            # Train XGBoost
            xgb_model_trainer = XGBoostModelCreator()
            label_encoder = xgb_model_trainer.train_xgboost_daily(
                training_series_df=training_series_df,
                model_output=model_output,
                symbol=symbol,
                classif_key=classif_key,
                series_csv=series_csv,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                make_stationary=make_stationary,
                class_weight=class_weight
            )

            return label_encoder

        except Exception as e:
            msg = f"CRITICAL ERROR processing model @train_xgboost: {str(e)}"
            traceback.print_exc()
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def process_train_RF(self, symbol, series_csv, d_from, d_to, model_output, classif_key,
                         n_estimators, max_depth, min_samples_split, criterion,
                         interval=DataSetBuilder._1_MIN_INTERVAL,
                         grouping_unit=None, grouping_classif_criteria=None,
                         group_as_mov_avg=False, grouping_mov_avg_unit=20,
                         make_stationary=False,class_weight=None):
        try:
            # Retrieve classification ranges based on interval
            range_clasifs = None
            if interval == DataSetBuilder._1_MIN_INTERVAL:
                range_clasifs = self.timestamp_range_classif_mgr.get_timestamp_range_classification_values(
                    classif_key, d_from, d_to)
            elif interval == DataSetBuilder._1_DAY_INTERVAL:
                range_clasifs = self.date_range_classif_mgr.get_date_range_classification_values(
                    classif_key, d_from, d_to)
            else:
                raise Exception(f'Invalid interval at process_train_RF: {interval}')

            # Build main symbol time series
            symbol_min_series_df = self.data_set_builder.build_interval_series(
                symbol, d_from, d_to, interval=interval,
                output_col=["symbol", "date", "open", "high", "low", "close"]
            )

            # Add classification labels to symbol series
            symbol_min_series_df = self.data_set_builder.build_minute_series_classification(
                range_clasifs,
                symbol_min_series_df,
                classif_col_name=classif_key,
                not_found_clasif="FLAT"
            )

            # Load feature variables
            variables_min_series_df = self.data_set_builder.build_interval_series(
                series_csv, d_from, d_to,
                interval=interval,
                output_col=["symbol", "date", "open", "high", "low", "close"]
            )

            # Merge features and labels into final dataset
            training_series_df = self.data_set_builder.merge_series(
                symbol_min_series_df, variables_min_series_df, "symbol", "date", symbol
            )

            # Optional: compute moving averages
            if group_as_mov_avg:
                training_series_df = self.data_set_builder.group_as_mov_avgs(
                    training_series_df, series_csv, grouping_mov_avg_unit
                )

            # Optional: apply grouping logic (e.g., weekly bars)
            if grouping_unit is not None:
                training_series_df = self.__group_dataframe__(
                    training_series_df, grouping_unit,
                    series_csv, grouping_classif_criteria, classif_key
                )

            # Logging for inspection
            DataframePrinter.print_dataframe_head_values_w_time(
                variables_min_series_df, "symbol", series_csv, 10, "date", "10:30:00"
            )
            DataframePrinter.print_data_farme_head(training_series_df, 10)

            # Train Random Forest
            rf_model_trainer = RandomForestModelCreator()
            label_encoder= rf_model_trainer.train_random_forest_daily(training_series_df=training_series_df,
                                                                      model_output=model_output, symbol=symbol,
                                                                      classif_key=classif_key, series_csv=series_csv,
                                                                      n_estimators=n_estimators, max_depth=max_depth,
                                                                      min_samples_split=min_samples_split,
                                                                      criterion=criterion,
                                                                      make_stationary=make_stationary,
                                                                      class_weight=class_weight)

            return label_encoder

        except Exception as e:
            msg = f"CRITICAL ERROR processing model @train_random_forest: {str(e)}"
            traceback.print_exc()
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def process_train_LSTM(self,symbol,variables_csv,d_from,d_to,model_output,classif_key,
                           epochs,timestamps,n_neurons,learning_rate,reg_rate, dropout_rate,
                           interval=DataSetBuilder._1_MIN_INTERVAL,clipping_rate=None,threshold_stop=None,grouping_unit=None,
                           grouping_classif_criteria=None,
                           group_as_mov_avg=False,grouping_mov_avg_unit=20,
                           batch_size=1,inner_activation=None, make_stationary=False):
        try:

            range_clasifs=None
            if interval==DataSetBuilder._1_MIN_INTERVAL:
                range_clasifs=self.timestamp_range_classif_mgr.get_timestamp_range_classification_values(classif_key, d_from, d_to)
            elif interval==DataSetBuilder._1_DAY_INTERVAL:
                range_clasifs = self.date_range_classif_mgr.get_date_range_classification_values(
                    classif_key, d_from, d_to)
            else:
                raise Exception(f'Invalid interval at process_train_LSTM:{interval}')


            symbol_min_series_df= self.data_set_builder.build_interval_series(symbol, d_from, d_to, interval=interval,
                                                                              output_col=["symbol", "date", "open",
                                                                                          "high", "low", "close"])
            symbol_min_series_df = self.data_set_builder.build_minute_series_classification(range_clasifs,
                                                                                            symbol_min_series_df,
                                                                                            classif_col_name=classif_key,
                                                                                            not_found_clasif="FLAT")

            variables_min_series_df= self.data_set_builder.build_interval_series(variables_csv, d_from, d_to,
                                                                                 interval=interval,
                                                                                 output_col=["symbol", "date", "open",
                                                                                             "high", "low", "close"])

            training_series_df= self.data_set_builder.merge_series(symbol_min_series_df, variables_min_series_df,
                                                                   "symbol", "date", symbol)

            if group_as_mov_avg:
                training_series_df=self.data_set_builder.group_as_mov_avgs(training_series_df,variables_csv,grouping_mov_avg_unit)


            DataframePrinter.print_dataframe_head_values_w_time(variables_min_series_df, "symbol", variables_csv, 10,"date","10:30:00")
            if grouping_unit is not None:

                training_series_df= self.__group_dataframe__(training_series_df, grouping_unit,
                                                             variables_csv,grouping_classif_criteria, classif_key)
                DataframePrinter.print_data_farme_head(training_series_df,10)



            rnn_model_trainer= DayTradingRNNModelCreator()

            if interval==DataSetBuilder._1_DAY_INTERVAL:
                rnn_model_trainer.train_LSTM_scalping(training_series_df, model_output, symbol_min_series_df, classif_key,
                                                      epochs, timestamps, n_neurons, learning_rate, reg_rate, dropout_rate,
                                                      variables_csv, clipping_rate, threshold_stop,
                                                      make_stationary=make_stationary, inner_activation=inner_activation,
                                                      batch_size=batch_size)
            elif interval==DataSetBuilder._1_MIN_INTERVAL:
                rnn_model_trainer.train_LSTM_daily_stateful(training_series_df= training_series_df,
                                                           model_output= model_output,
                                                           symbol=symbol,
                                                           variables_csv=variables_csv,
                                                           classif_key=classif_key,
                                                           epochs= epochs,
                                                           timestamps= timestamps,
                                                           n_neurons= n_neurons,
                                                           learning_rate= learning_rate,
                                                           reg_rate= reg_rate,
                                                           dropout_rate= dropout_rate,
                                                           clipping_rate= clipping_rate,
                                                           inner_activation=inner_activation,
                                                           batch_size=batch_size)

            #pd.set_option('display.max_columns', None)
            #print(training_series_df.head())
            #pd.reset_option('display.max_columns')

            return None

        except Exception as e:
            msg = "CRITICAL ERROR processing model @train_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)



    def detect_and_save_regime_switch(self, variables, d_from, d_to, regime_filter, regime_candle, regime_window,
                                      slope_threshold=0.3, abs_value_threshold=10):
        """
        Detects regime switch conditions on the given variables and saves them as an economic candle.
        """
        df = self.data_set_builder.build_daily_series_classification(",".join(variables), d_from, d_to,
                                                                     add_classif_col=False)

        for var in variables:
            self.logger.print(f"🔍 Evaluating regime filter '{regime_filter}' on variable: {var}", MessageType.INFO)

            values = df[var].tolist()
            dates = df["date"].tolist()


            regime_signals = SlopeCalculator.classify_slope_regime(
                values=values,
                dates=dates,
                regime_filter=regime_filter,
                window=regime_window,
                slope_threshold=slope_threshold,
                abs_value_threshold=abs_value_threshold
            )

            for date, signal in zip(dates, regime_signals):
                self.data_set_builder.economic_series_mgr.persist_economic_series(regime_candle, date,
                                                                                  Intervals.DAY.value, signal)

        self.logger.print(f"✅ Finished saving regime signal '{regime_candle}'", MessageType.INFO)

    def process_daily_candles_graph(self,symbol, date, interval,mov_avg_unit):
        start_of_day = datetime(date.year, date.month, date.day)
        end_of_day = start_of_day + timedelta(hours=23, minutes=59, seconds=59)

        prices_df= self.data_set_builder.build_interval_series(symbol, start_of_day, end_of_day,
                                                               interval=DataSetBuilder._1_MIN_INTERVAL)

        GraphBuilder.build_candles_graph(prices_df,mov_avg_unit =mov_avg_unit)

        return  None


    def process_indicator_candles_graph(self,symbol, d_from,d_to, interval,mov_avg_unit):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = datetime(d_to.year, d_to.month, d_to.day) + timedelta(hours=23, minutes=59, seconds=59)

        prices_df= self.data_set_builder.build_interval_series(symbol, start_of_day, end_of_day,
                                                               interval=interval)

        GraphBuilder.build_candles_graph(prices_df,mov_avg_unit =mov_avg_unit)

        return  None


    def process_backtest_slope_model(self,symbol,model_candle,d_from,d_to,portf_init_size,trading_algo,
                                     n_algo_params):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = d_to + timedelta(hours=23, minutes=59, seconds=59)

        #1- The trading symbol DF
        symbol_series_df = self.data_set_builder.build_interval_series(symbol, start_of_day, end_of_day,
                                                                       interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                       output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        #2- The explanatory variables DF
        variables_series_df = self.data_set_builder.build_interval_series(model_candle, d_from, d_to,
                                                                          interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                          output_col=["symbol", "date", "open",
                                                                                          "high", "low", "close"])

        #3- We merge them
        training_series_df = self.data_set_builder.merge_series(symbol_series_df, variables_series_df, "symbol", "date",
                                                                symbol)

        #4- We drop weekends and holidays
        training_series_df = training_series_df.dropna(subset=[f"close_{symbol}"])

        portf_summary = PortfSummary(symbol, portf_init_size, p_trade_comm=0,
                                     p_trading_algo=trading_algo, p_algo_params=n_algo_params)

        summary_dto,portf_positions= self.__backtest_strategy__(training_series_df, model_candle, portf_init_size, n_algo_params,
                                                                portf_summary=portf_summary)


        self.__log_scalping_trading_results__(d_from, d_to, portf_init_size, summary_dto)
        return summary_dto


    def process_ARIMA_predictions(self, series_key,d_from,d_to,algo_params):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = d_to + timedelta(hours=23, minutes=59, seconds=59)

        # 1- The trading symbols DF
        series_df = self.data_set_builder.build_interval_series(series_key, start_of_day, end_of_day,
                                                                           interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                           output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        #2 - Predict ARIMA
        arima_Analyzer = ARIMAModelsAnalyzer(self.logger)
        p=algo_params["p"]
        d=algo_params["d"]
        q=algo_params["q"]
        s = algo_params["s"]
        period=algo_params["period"]
        step=algo_params["step"]

        preds=None
        if s is  None: #ARIMA
            preds = arima_Analyzer.build_and__predict_ARIMA_model(series_df,f"close",
                                                                 p, d, q, period, step)
        else:
            preds = arima_Analyzer.build_and_predict_SARIMA_model(series_df, f"close",
                                                                  p, d, q,p,d,q,s,period, step)

        return preds

    def process_download_financial_data(self, symbol: str, d_from: str, d_to: str, algo_params: dict):
        vendor = algo_params.get("vendor", "").upper()
        vendor_params = algo_params.get("vendor_params", {})

        if vendor == InformationVendors.TRADINGVIEW.value:
            downloader = TradingViewDownloader(vendor_params)
            df = downloader.download(symbol, from_date=d_from, to_date=d_to)
            interval_enum=downloader.get_interval_enum_translation()

            for index, row in df.iterrows():
                date = row.name if "date" not in row else row["date"]  # por si es DataFrame con índice datetime
                value = row["value"] if "value" in row else row["close"]
                self.economic_series_mgr.persist_economic_series(symbol, date, interval_enum.value, value)

        elif vendor == InformationVendors.FRED.value:
            downloader = FredDownloader(vendor_params)
            df = downloader.download(symbol, from_date=d_from, to_date=d_to)
            for index, row in df.iterrows():
                date = row["date"]
                value = row["value"]
                try:

                    print(f"[DEBUG] Persisting FRED row: symbol={symbol}, date={date}, value={value} ({type(value)})")
                    if pd.isna(value):
                        print(f"[WARNING] Skipping NaN value for symbol={symbol} at date={date}")
                        continue
                    value = float(value)
                    self.economic_series_mgr.persist_economic_series(symbol, date, Intervals.DAY.value, value)

                except Exception as e:
                    print(f"[ERROR] Failed to persist row for symbol={symbol}, date={date}, value={value}")
                    print(traceback.format_exc())

        else:
            raise Exception(f"❌ Unknown data vendor '{vendor}'. Supported: TRADINGVIEW, FRED")

       #TODO --> Convertir a EconomicValues y persistir

    def process_download_bcra_interest_rates(self, d_from, d_to, algo_params):
        """
        Executes the BCRA Interest Rates retrieval process.
        This method is part of the business logic layer.
        """

        # Instantiate the BCRA Service Layer (no API key required)
        bcra_service = BCRAServiceLayer()

        # Fetch the data from the public API
        rates = bcra_service.get_interest_rates(d_from, d_to)

        # Handle empty results
        if not rates:
            print("⚠️ No rates were returned from BCRA API.")
            return

        # Display results (later this can be persisted or visualized)
        print("📊 BCRA Interest Rates Retrieved:")
        for r in rates:
            print(f"  {r.name:<35} {r.date} → {r.value}")

    def process_create_lightweight_indicator(self, csv_indicators, d_from, d_to,output_symbol, benchmark=None, plot_result=True):
        print(f"🧩 Building DataFrame with indicators: {csv_indicators}")

        indicators_series_df = self.data_set_builder.build_interval_series(
            series_csv=csv_indicators,
            d_from=d_from,
            d_to=d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "close"]
        )

        if indicators_series_df.empty:
            raise Exception("❌ No data returned for selected indicators")

        # Pivot to wide format
        pivot_df=self.data_set_builder.pivot_and_merge_indicators(indicators_series_df)
        pivot_df = DataframeFiller.fill_missing_values(pivot_df)
        pivot_df.dropna(inplace=True)

        print(f"📊 Running PCA on {pivot_df.shape[1]} variables and {pivot_df.shape[0]} rows")
        pivot_df=PCACalcualtor.calculate_PCA(pivot_df,output_symbol)

        # Persist each row
        for _, row in pivot_df.iterrows():
            date = row["date"]
            value = row[output_symbol]
            self.economic_series_mgr.persist_economic_series(
                output_symbol,
                date if isinstance(date, datetime) else pd.to_datetime(date),
                Intervals.DAY.value,
                value
            )

        print(f"✅ Persisted {len(pivot_df)} values for {output_symbol}")

        benchmark_df=None
        if benchmark:
            benchmark_df = self.data_set_builder.build_interval_series(
                series_csv=benchmark,
                d_from=d_from,
                d_to=d_to,
                interval=DataSetBuilder._1_DAY_INTERVAL,
                output_col=["symbol", "date", "close"]
            )

        # Plot result (optional) --> Take to graph
        if plot_result:
            GraphBuilder.plot_2_series_overlapped(pivot_df,benchmark_df,benchmark,output_symbol)

    def process_download_financial_data_bulk(self, symbol: str, d_from: str, d_to: str, algo_params: dict):
        self.logger.print(f"[Orch] Bulk dispatching '{symbol}' to standard download logic", MessageType.INFO)
        self.process_download_financial_data(symbol, d_from, d_to, algo_params)

    def process_create_spread_varaible(self, diff_indicators, d_from, d_to,output_symbol):
        print(f"🧩 Building DataFrame with indicators: {diff_indicators}")

        variables_csv= ",".join(diff_indicators.split("-"))

        if d_to is None:
            d_to = datetime.today().date() + timedelta(days=1)

        indicators_series_df = self.data_set_builder.build_interval_series(
            series_csv=variables_csv,
            d_from=d_from,
            d_to=d_to,
            interval=DataSetBuilder._1_DAY_INTERVAL,
            output_col=["symbol", "date", "close"]
        )

        if indicators_series_df.empty:
            raise Exception("❌ No data returned for selected indicators")

        # Pivot to wide format
        pivot_df=self.data_set_builder.pivot_and_merge_indicators(indicators_series_df)
        pivot_df = DataframeFiller.fill_missing_values(pivot_df)
        pivot_df.dropna(inplace=True)

        pivot_df[output_symbol] = pivot_df[f"close_{diff_indicators.split('-')[0]}"] - pivot_df[f"close_{diff_indicators.split('-')[1]}"]

        # Persist each row
        for _, row in pivot_df.iterrows():
            date = row["date"]
            value = row[output_symbol]
            self.economic_series_mgr.persist_economic_series(
                output_symbol,
                date if isinstance(date, datetime) else pd.to_datetime(date),
                Intervals.DAY.value,
                value
            )

        print(f"✅ Persisted {len(pivot_df)} values for {output_symbol}")



    def process_create_sinthetic_indicator_logic(self,comp_path,model_candle,d_from,d_to,algo_params):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = d_to + timedelta(hours=23, minutes=59, seconds=59)

        #0- We extract the ETF composition info
        indicators_csv = self.data_set_builder.extract_series_csv_from_etf_file(comp_path, 0)
        indicator_types = self.data_set_builder.extract_series_csv_from_etf_file(comp_path, 1)

        #1- The trading symbols DF
        indicators_series_df = self.data_set_builder.build_interval_series(indicators_csv, start_of_day, end_of_day,
                                                                        interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                        output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        #2- We have one dataframe per indicator
        indicators_series_df=self.data_set_builder.privot_and_merge_dataframes(indicators_series_df)
        indicators_series_df["indicator"]=model_candle

        #3- Dto with indicator-type arrays
        indicator_type_arr= IndicatorTypeDTO.load_indicator_type_data(indicators_csv,indicator_types)

        #4- Build and create the indicator
        ind_creator = SintheticIndicatorCreator(self.logger)
        indicators_series_df=ind_creator.build_sinthetic_indicator(indicators_series_df,indicator_type_arr,algo_params)

        #4- We persist the newly created indicator
        self.data_set_builder.persist_sinthetic_indicator(indicators_series_df)

        self.logger.do_log(f"Successfully created {len(indicators_series_df)} records for indicator {model_candle}",MessageType.INFO)

        return  indicators_series_df


    def process_backtest_slope_model_on_custom_etf(self,etf_path,model_candle,d_from,d_to,portf_init_size,trading_algo,
                                     n_algo_params):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = d_to + timedelta(hours=23, minutes=59, seconds=59)

        #0- We extract the ETF composition info
        symbols_csv=self.data_set_builder.extract_series_csv_from_etf_file(etf_path,1)
        etf_comp_dto_arr = ETFFileExtractionHandler.__extract_etf_composition__(etf_path, 1, 0)

        #1- The trading symbols DF
        symbols_series_df = self.data_set_builder.build_interval_series(symbols_csv, start_of_day, end_of_day,
                                                                        interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                        output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        #2- We have one datafraem per symbol in the ETF
        symbols_series_df_arr=self.data_set_builder.split_dataframe_by_symbol(symbols_series_df,"symbol")


        #3- The explanatory variables DF
        variables_series_df = self.data_set_builder.build_interval_series(model_candle, d_from, d_to,
                                                                          interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                          output_col=["symbol", "date", "open",
                                                                                          "high", "low", "close"])
        variables_series_df= self.data_set_builder.shift_dates(variables_series_df, n_algo_params)

        #4- We merge them
        training_series_df_arr={}
        for symbol in  symbols_series_df_arr:
            symbols_series_df= symbols_series_df_arr[symbol]
            training_series_df = self.data_set_builder.merge_series(symbols_series_df,
                                                                    variables_series_df,
                                                                    "symbol","date",
                                                                    symbol)
            training_series_df_arr[symbol]=training_series_df


        #5-We merge all the training_series_df
        merged_training_series_df=None
        for symbol in   training_series_df_arr:
            training_series_df=training_series_df_arr[symbol]
            if merged_training_series_df is None:
                merged_training_series_df=training_series_df
            else:
                merged_training_series_df=self.data_set_builder.merge_dataframes(training_series_df,merged_training_series_df,
                                                                          "date")
        merged_training_series_df = merged_training_series_df.drop(columns=['trading_symbol'], errors='ignore')


        #6- We run the backtest
        portf_summary = PortfSummary(symbols_csv, portf_init_size, p_trade_comm=0, p_trading_algo=trading_algo, p_algo_params=n_algo_params)
        summ_dto,portf_positions= self.__backtest_strategy__(merged_training_series_df, model_candle, portf_init_size, n_algo_params,
                                                             portf_summary=portf_summary, etf_comp_dto_arr=etf_comp_dto_arr)


        self.__log_scalping_trading_results__(d_from, d_to, portf_init_size, summ_dto)
        return summ_dto,portf_positions


    def model_custom_etf(self,weights_csv,symbols_csv,start_of_day, end_of_day):

        # 0- We extract the ETF composition info
        etf_comp_dto_arr=ETFFileExtractionHandler.__extract_etf_composition_from_csv__(weights_csv, symbols_csv)

        # 1- The trading symbols DF
        symbols_series_df = self.data_set_builder.build_interval_series(symbols_csv, start_of_day, end_of_day,
                                                                        interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                        output_col=["symbol", "date", "open",
                                                                                    "high", "low", "close"])

        # 2- We have one datafraem per symbol in the ETF
        symbols_series_df_arr = self.data_set_builder.split_dataframe_by_symbol(symbols_series_df, "symbol")


        # 3-We merge all the training_series_df
        merged_training_series_df=self.data_set_builder.merge_multiple_series(symbols_series_df_arr.values(),date_col="date")

        #6- We run the backtest
        portf_size=ParametersKeys.STANDARD_PORTFOLIO.value
        n_algo_param_dict={}
        n_algo_param_dict[ParametersKeys.TRADE_COMM_NOM_KEY.value] = 0  # No comissions
        portf_summary = PortfSummary(symbols_csv, portf_size, p_trade_comm=0,
                                     p_trading_algo=TradingAlgoStrategy.BUY_AND_HOLD.name, p_algo_params={})
        summary_dto,portf_pos= self.__backtest_strategy__(merged_training_series_df, TradingAlgoStrategy.BUY_AND_HOLD.name,
                                             portf_size,n_algo_params= n_algo_param_dict,
                                             portf_summary=portf_summary,
                                             etf_comp_dto_arr=etf_comp_dto_arr)


        #4- We bould the custom ETF series
        if len(portf_pos)<=0:
            raise Exception("CRITICAL ERROR building MTMs portfolio! ")
        return portf_pos[0].detailed_MTMS

    def process_download_sec_securities(self):
        try:
            # Download all securities from SEC API (or local JSON)
            json_data = SECSecuritiesDownloader.download_security_list_from_edgar()

            dtos = []
            for item in json_data:
                dto = SecSecurityDTO(
                    cik=item.get("cik_str"),
                    ticker=item.get("ticker"),
                    name=item.get("title"),
                    exchange=item.get("exchange"),
                    category=item.get("category"),
                    sic=item.get("sic"),
                    entityType=item.get("entityType")
                )
                dtos.append(dto)

            self.sec_securities_mgr.persist_bulk(dtos)

            self.logger.do_log(f"process_download_sec_securities: ✅ Persisted {len(dtos)} SEC securities",
                               MessageType.INFO)

        except Exception as e:
            print(traceback.format_exc())
            self.logger.do_log(f"process_download_sec_securities: ❌ Error persisting SEC securities - {str(e)}",
                               MessageType.ERROR)

    def persist_custom_etf_series(self, symbol: str, base: float, detailed_mtms: list, interval: str):
        """
        Persist a custom ETF time series into the EconomicSeries database.
        - Each MTM value is divided by 'base' and rounded to 2 decimals.
        - Uses EconomicSeriesManager to call stored procedure PersistCandle.
        """
        if not detailed_mtms:
            self.logger.do_log("persist_custom_etf_series: empty detailed_mtms, nothing to persist.",
                               MessageType.WARNING)
            return

        if base <= 0:
            raise ValueError("Base must be > 0")


        q = Decimal("0.01")
        base_dec = Decimal(str(base))

        inserted = 0
        for dp in detailed_mtms:
            try:
                raw = Decimal(str(dp.MTM))
                scaled = (raw / base_dec).quantize(q, rounding=ROUND_HALF_UP)
                self.economic_series_mgr.persist_economic_series(
                    symbol=symbol,
                    date=dp.date,
                    interval=interval,
                    value=float(scaled)
                )
                inserted += 1
            except Exception as ex:
                self.logger.do_log(f"persist_custom_etf_series: skipped {dp.date} -> {ex}", MessageType.WARNING)

        self.logger.do_log(f"persist_custom_etf_series: persisted {inserted} candles for {symbol}", MessageType.INFO)

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









