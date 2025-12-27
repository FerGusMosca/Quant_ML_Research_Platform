class SecSecurityDTO:
    """
    Data Transfer Object for SEC Securities
    """
    def __init__(self, cik, ticker,symbol, name, exchange, category, sic, entityType):
        self.cik = cik
        self.ticker = ticker
        self.symbol=symbol
        self.name = name
        self.exchange = exchange
        self.category = category
        self.sic = sic
        self.entityType = entityType
