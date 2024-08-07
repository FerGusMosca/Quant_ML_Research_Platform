class EconomicValue:

    def __init__(self,p_symbol, p_interval,p_date,p_open=None,p_high=None,p_low=None,p_close=None,p_trade=None,
                 p_cash_volume=None,p_nominal_volume=None):
        self.symbol=p_symbol
        self.interval=p_interval
        self.date=p_date
        self.open=p_open
        self.high=p_high
        self.low=p_low
        self.close=p_close
        self.trade=p_trade
        self.cash_volume=p_cash_volume
        self.nominal_volume=p_nominal_volume