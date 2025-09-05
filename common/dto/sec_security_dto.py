class SecSecurityDTO:
    """
    Data Transfer Object for SEC Securities
    """
    def __init__(self, cik, ticker, name, exchange, category, sic, entityType):
        self.cik = cik
        self.ticker = ticker
        self.name = name
        self.exchange = exchange
        self.category = category
        self.sic = sic
        self.entityType = entityType
