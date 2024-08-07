from sources.framework.business_entities.market_data.market_data import *
from sources.framework.common.enums.Actions import *
from sources.framework.common.enums.fields.market_data_field import *
from sources.framework.common.wrappers.wrapper import *

class MarketDataWrapper(Wrapper):
    def __init__(self,pMarketData):
        self.MarketData=pMarketData

        # region Public Methods

    def GetAction(self):
        return Actions.MARKET_DATA

    def GetField(self, field):

        if field == None:
            return None

        if self.MarketData == None:
            return None

        if field == MarketDataField.Symbol:
            return self.MarketData.Security.Symbol
        elif field == MarketDataField.SecurityType:
            return self.MarketData.Security.SecurityType
        elif field == MarketDataField.Currency:
            return self.MarketData.Currency
        elif field == MarketDataField.MDMkt:
            return self.MarketData.MDMkt
        elif field == MarketDataField.OpeningPrice:
            return self.MarketData.OpeningPrice
        elif field == MarketDataField.ClosingPrice:
            return self.MarketData.ClosingPrice
        elif field == MarketDataField.TradingSessionHighPrice:
            return self.MarketData.TradingSessionHighPrice
        elif field == MarketDataField.TradingSessionLowPrice:
            return self.MarketData.TradingSessionLowPrice
        elif field == MarketDataField.TradeVolume:
            return self.MarketData.TradeVolume
        elif field == MarketDataField.OpenInterest:
            return self.MarketData.OpenInterest
        elif field == MarketDataField.SettlType:
            return self.MarketData.SettlType
        elif field == MarketDataField.CompositeUnderlyingPrice:
            return self.MarketData.CompositeUnderlyingPrice
        elif field == MarketDataField.MidPrice:
            return self.MarketData.MidPrice
        elif field == MarketDataField.SessionHighBid:
            return self.MarketData.SessionHighBid
        elif field == MarketDataField.SessionLowOffer:
            return self.MarketData.SessionLowOffer
        elif field == MarketDataField.EarlyPrices:
            return self.MarketData.EarlyPrices
        elif field == MarketDataField.Trade:
            return self.MarketData.Trade
        elif field == MarketDataField.MDTradeSize:
            return self.MarketData.MDTradeSize
        elif field == MarketDataField.BestBidPrice:
            return self.MarketData.BestBidPrice
        elif field == MarketDataField.BestAskPrice:
            return self.MarketData.BestAskPrice
        elif field == MarketDataField.BestBidSize:
            return self.MarketData.BestBidSize
        elif field == MarketDataField.BestBidCashSize:
            return self.MarketData.BestBidCashSize
        elif field == MarketDataField.BestAskSize:
            return self.MarketData.BestAskSize
        elif field == MarketDataField.BestAskCashSize:
            return self.MarketData.BestAskCashSize
        elif field == MarketDataField.BestBidExch:
            return self.MarketData.BestBidExch
        elif field == MarketDataField.BestAskExch:
            return self.MarketData.BestAskExch
        elif field == MarketDataField.MDEntryDate:
            return self.MarketData.MDEntryDate
        elif field == MarketDataField.MDEntryTime:
            return self.MarketData.MDEntryTime
        elif field == MarketDataField.MDLocalEntryDate:
            return self.MarketData.MDLocalEntryDate
        elif field == MarketDataField.MDLocalEntryDate:
            return self.MarketData.MDLocalEntryDate
        elif field == MarketDataField.LastTradeDateTime:
            return self.MarketData.LastTradeDateTime
        elif field == MarketDataField.Change:
            return self.MarketData.Change
        elif field == MarketDataField.StdDev:
            return self.MarketData.StdDev
        else:
            return None

    # endregion
