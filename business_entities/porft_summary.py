from business_entities.portf_pnl_summary import PortfPnLSummary
from common.util.financial_calculations.financial_calculation_helper import FinancialCalculationsHelper


class PortfSummary:

    def __init__(self,symbol,p_portf_position_size,p_trade_comm=0,p_trading_algo=None,p_algo_params=[],
                 p_period=None,p_year=None):
        self.trading_algo=p_trading_algo
        self.n_algo_params=p_algo_params
        self.symbol=symbol

        self.total_net_profit=0
        self.total_net_profit_str="-"

        self.portf_pos_summary=[]
        self.max_cum_drawdowns=[]
        self.daily_profits=[]

        self.portf_pos_size=p_portf_position_size
        self.trade_comm=p_trade_comm

        self.max_drawdown=0
        self.max_daily_drawdown=0
        self.accum_positions=0

        self.max_drawdown_on_MTM_str= "-"
        self.max_drawdown_on_MTM=0

        self.portf_init_MTM=0
        self.portf_final_MTM=0

        self.profit_pct=None
        self.cagr_pct=None
        self.drawdown_pct=None

        self.period=p_period
        self.year=p_year


    def append_position_summary(self,pos_summary):
        self.portf_pos_summary.append(pos_summary)

    def calculate_th_nom_profit(self):

        accum=0
        for portf_pos_summary  in self.portf_pos_summary:
            accum+=portf_pos_summary.th_nom_profit


        return accum

    def update_max_drawdown(self):
        if isinstance(self.max_cum_drawdowns, list) and len(self.max_cum_drawdowns) > 0:
            self.max_daily_drawdown = min(self.max_cum_drawdowns)
        else:
            self.max_daily_drawdown = 0

        self.max_drawdown = FinancialCalculationsHelper.calculate_max_total_drawdown(self.daily_profits)

    def calculate_last_portf_position_summary(self,day):
        position_summary= PortfPnLSummary(day,self.daily_profits[-1],
                                          self.max_cum_drawdowns[-1] if len(self.max_cum_drawdowns)>0 else 0)
        self.portf_pos_summary.append(position_summary)
        return  position_summary

    def calculate_profit_stats(self,eval_d_from,eval_d_to):
        total_profit = self.portf_final_MTM - self.portf_init_MTM
        self.profit_pct = (total_profit / self.portf_init_MTM) * 100
        self.drawdown_pct = self.max_drawdown * 100
        self.calculate_cagr(eval_d_from,eval_d_to)

    def calculate_cagr(self,eval_d_from,eval_d_to):
        days_diff = (eval_d_to - eval_d_from).days
        years = days_diff / 365.0
        cagr = (self.portf_final_MTM / self.portf_init_MTM) ** (1 / years) - 1
        self.cagr_pct = cagr * 100.0