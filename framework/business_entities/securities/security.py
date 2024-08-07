from sources.framework.common.enums.SecurityType import SecurityType
from sources.framework.business_entities.market_data.market_data import *
import json

class Security:
    def __init__(self, Symbol=None, Exchange=None,Currency=None, SecType=None):

        #region Attributes
        self.Symbol=Symbol
        self.SecurityDesc = Symbol
        self.Exchange= Exchange
        self.Currency=Currency
        self.SecurityType=SecType
        self.MarketData = MarketData()
        self.MarketDataArr={}

        #endregion

    @staticmethod
    def GetSecurityType(secType):
        if secType=="CS":
            return SecurityType.CS
        else:
            raise Exception("Not implemented security type conversion for CS {}".format(secType))


    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,sort_keys=True, indent=4)