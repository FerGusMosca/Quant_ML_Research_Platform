from sources.framework.business_entities.securities.security import Security
from sources.framework.common.enums.SettlType import SettlType
from sources.framework.common.enums.Side import Side
from sources.framework.common.enums.QuantityType import QuantityType
from sources.framework.common.enums.PriceType import PriceType
from sources.framework.common.enums.OrdStatus import OrdStatus
import datetime

class Order:

    def __init__(self, ClOrdId=None,Security=None,
                 SettlType=None,Side=None,Exchange=None,OrdType=None,QuantityType=None,OrderQty=None,
                 PriceType=None,Price=None,StopPx=None,Currency=None,
                 TimeInForce=None,Account=None,OrdStatus=None,Broker=None,Strategy=None,MarketArrivalTime=None):

        #region Attributes
        self.ClOrdId=ClOrdId
        self.OrigClOrdId=ClOrdId
        self.OrderId=""
        self.Security=Security
        self.SettlType=SettlType
        self.Side=Side
        self.Exchange=Exchange
        self.OrdType=OrdType
        self.QuantityType=QuantityType
        self.OrderQty=OrderQty
        self.PriceType=PriceType
        self.Price=Price
        self.StopPx = StopPx
        self.Currency=Currency
        self.TimeInForce=TimeInForce
        self.Account=Account
        self.OrdStatus=OrdStatus
        self.RejReason=None
        self.Broker = Broker
        self.Strategy = Strategy
        self.MarketArrivalTime=MarketArrivalTime

        #endregion

    def IsOpenOrder(self):
        return ( self.OrdStatus==OrdStatus.New or self.OrdStatus==OrdStatus.PartiallyFilled
               or self.OrdStatus==OrdStatus.PendingNew or self.OrdStatus==OrdStatus.Replaced
               or self.OrdStatus==OrdStatus.PendingReplace or self.OrdStatus==OrdStatus.AcceptedForBidding)