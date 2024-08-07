import uuid

from sources.framework.common.enums.PositionsStatus import PositionStatus
from sources.framework.common.enums.ExecType import *
from sources.framework.common.enums.Side import *
from sources.framework.common.enums.PriceType import PriceType
from sources.framework.common.enums.QuantityType import QuantityType
from sources.framework.common.enums.OrdType import *


_New = "New"
_DoneForDay = "DoneForDay"
_Canceled = "Canceled"
_Replaced = "Replaced"
_PendingCancel = "PendingCancel"
_Stopped = "Stopped"
_Rejected = "Rejected"
_Suspended = "Suspended"
_PendingNew = "PendingNew"
_Calculated = "Calculated"
_Expired = "Expired"
_PendingReplace = "PendingReplace"
_PartiallyFilled = "PartiallyFilled"
_Filled = "Filled"
_Unknown = "Unknown"

_Side_Buy="Buy"
_Side_Sell="Sell"
_Side_Sell_Short="SellShort"
_Side_Buy_To_Close="BuyToClose"
_Side_Uknown="Unknown"

class Position:

    _ORDER_ID_SEPARATOR="__"

    def __init__(self, PosId=None, Security=None,Side=None,PriceType=None,Qty=None,QuantityType=None,Account=None,
                Broker=None,Strategy=None,OrderType=None, OrderPrice = None, CumQty=None, LeavesQty = None,
                 AvgPx=None, LastQty=None,LastPx=None,LastMkt=None):

        #region Attributes
        self.PosId=PosId
        self.Security=Security
        self.PosStatus=PositionStatus.New
        self.Side=Side
        self.Qty=Qty
        self.QuantityType=QuantityType
        self.Exchange=None
        self.PriceType=PriceType
        self.CumQty = CumQty
        self.LeavesQty = LeavesQty
        self.AvgPx = AvgPx
        self.LastQty = LastQty
        self.LastPx = LastPx
        self.LastMkt = LastMkt

        self.CashQty=None
        self.Percent =None

        self.ExecutionReports=[]
        self.Orders=[]

        self.PositionRejectReason = None
        self.PositionRejectText = None
        self.Account = Account

        self.Broker=Broker
        self.Strategy=Strategy
        self.OrderType=OrderType
        self.OrderPrice = OrderPrice

        self.ArrivalPrice= None

        self.StopLoss=None
        self.TakeProfit=None
        self.CloseEndOfDay =None

        #endregion


        #region Public Methods

    def SetPositionStatusFromExecution(self,execReport):

        execType=execReport.ExecType

        if execType==ExecType.New:
            self.PosStatus=PositionStatus.New
        elif execType==ExecType.DoneForDay:
            self.PosStatus=PositionStatus.DoneForDay
        elif execType==ExecType.Canceled:
            self.PosStatus=PositionStatus.Canceled
        elif execType==ExecType.Replaced:
            self.PosStatus=PositionStatus.Replaced
        elif execType==ExecType.PendingCancel:
            self.PosStatus=PositionStatus.PendingCancel
        elif execType==ExecType.Stopped:
            self.PosStatus=PositionStatus.Stopped
        elif execType==ExecType.Rejected:
            self.PosStatus=PositionStatus.Rejected
        elif execType==ExecType.Suspended:
            self.PosStatus=PositionStatus.Suspended
        elif execType==ExecType.PendingNew:
            self.PosStatus=PositionStatus.PendingNew
        elif execType==ExecType.Calculated:
            self.PosStatus=PositionStatus.Calculated
        elif execType==ExecType.Expired:
            self.PosStatus=PositionStatus.Expired
        elif execType==ExecType.PendingReplace:
            self.PosStatus=PositionStatus.PendingReplace
        elif execType==ExecType.Trade:
            self.PosStatus=PositionStatus.Filled if execReport.LeavesQty==0 else PositionStatus.PartiallyFilled
        else:
            raise Exception("Unknown exec type {}".format(execType))



        #endregion

    def LongPositionOpened(self):
        return self.Side==Side.Buy

    def ShortPositionOpened(self):
        return self.Side==Side.SellShort or self.Side==Side.Sell

    def IsOpenPosition(self):
        return self.PosStatus == PositionStatus.New or self.PosStatus == PositionStatus.PendingNew \
               or self.PosStatus == PositionStatus.PartiallyFilled or self.PosStatus == PositionStatus.Replaced \
               or self.PosStatus == PositionStatus.PendingCancel or self.PosStatus == PositionStatus.Calculated \
               or self.PosStatus == PositionStatus.PendingReplace or self.PosStatus == PositionStatus.AcceptedForBidding \
               or self.PosStatus == PositionStatus.PendingCancel

    def IsFinishedPosition(self):

        if (self.PosStatus == PositionStatus.Filled
            or self.PosStatus == PositionStatus.DoneForDay
            or self.PosStatus == PositionStatus.Stopped
            or self.PosStatus == PositionStatus.Rejected
            or self.PosStatus == PositionStatus.Suspended
            or self.PosStatus == PositionStatus.Expired
            or self.PosStatus == PositionStatus.Canceled):
            return True
        else:
            return  False

    def IsRejectedPosition(self):
        return self.PosStatus==PositionStatus.Rejected

    def IsTradedPosition(self):

        if (self.PosStatus == PositionStatus.Filled
            or self.PosStatus == PositionStatus.PartiallyFilled):
            return True
        else:
            return  False

    def ValidateNewPosition(self):
        if(self.OrderType is not None):

            if  self.OrderType==OrdType.Market and self.OrderPrice is not None:
                raise Exception("You cannot specify a limit price for a market position. Symbol {}".format(self.Security.Symbol))

    def GetLastExecutionReport(self):

        if len(self.ExecutionReports)>0:
            #execList = sorted(self.ExecutionReports, key=lambda execRep: execRep.TransactTime, reverse=True)
            #return execList[0]
            return  self.ExecutionReports[-1]
        else:
            return None

    def GetLastOrder(self):

        if len(self.Orders)>0:
            #ordList = sorted(self.Orders, key=lambda order: order.ClOrdId, reverse=True)
            #return ordList[0]
            return  self.Orders[-1]#last one
        else:
            return None

    def GetFirstTradeExecutionReport(self):

        if len(self.ExecutionReports)>0:
            execList = sorted(self.ExecutionReports, key=lambda execRep: execRep.TransactTime is not None, reverse=False)
            if len(execList)>0:
                return execList[0]
            else:
                return None
        else:
            return None

    def AppendOrder(self,order):
        ordersInMem = list(filter(lambda x: x.ClOrdId == order.ClOrdId, self.Orders))

        if len(ordersInMem)>0:
            ordersInMem[0]=order
        else:
            self.Orders.append(order)

    @staticmethod
    def CountOpenPositions(self, positions):
        i=0

        for pos in positions:
            if not pos.IsFinishedPosition():
                i+=1

        return i


    def GetStrStatus(self):

        if self.PosStatus is None:
            return _Unknown
        elif self.PosStatus==PositionStatus.New:
            return _New
        elif self.PosStatus==PositionStatus.DoneForDay:
            return _DoneForDay
        elif self.PosStatus==PositionStatus.Canceled:
            return _Canceled
        elif self.PosStatus==PositionStatus.Replaced:
            return _Replaced
        elif self.PosStatus==PositionStatus.PendingCancel:
            return _PendingCancel
        elif self.PosStatus==PositionStatus.Stopped:
            return _Stopped
        elif self.PosStatus==PositionStatus.Rejected:
            return _Rejected
        elif self.PosStatus==PositionStatus.Suspended:
            return _Suspended
        elif self.PosStatus==PositionStatus.Calculated:
            return _Calculated
        elif self.PosStatus==PositionStatus.Expired:
            return _Expired
        elif self.PosStatus==PositionStatus.PendingReplace:
            return _PendingReplace
        elif self.PosStatus==PositionStatus.PartiallyFilled:
            return _PartiallyFilled
        elif self.PosStatus==PositionStatus.Filled:
            return _Filled
        elif self.PosStatus==PositionStatus.Unknown:
            return _Unknown
        else:
            return _Unknown

    @staticmethod
    def FromStrStatus(strStatus):

        if strStatus == _Unknown :
            return PositionStatus.Unknown
        elif strStatus==_New:
            return  PositionStatus.New
        elif strStatus==_DoneForDay:
            return PositionStatus.DoneForDay
        elif strStatus==_Canceled:
            return  PositionStatus.Canceled
        elif strStatus== _Replaced:
            return PositionStatus.Replaced
        elif strStatus== _PendingCancel:
            return PositionStatus.PendingCancel
        elif strStatus== _Stopped:
            return PositionStatus.Stopped
        elif strStatus==_Rejected:
            return PositionStatus.Rejected
        elif strStatus== _Suspended:
            return PositionStatus.Suspended
        elif strStatus== _Calculated:
            return PositionStatus.Calculated
        elif strStatus== _Expired:
            return PositionStatus.Expired
        elif strStatus== _PendingReplace:
            return PositionStatus.PendingReplace
        elif strStatus== _PartiallyFilled:
            return PositionStatus.PartiallyFilled
        elif strStatus== _Filled:
            return PositionStatus.Filled
        elif strStatus== _Unknown:
            return PositionStatus.Unknown
        else:
            return _Unknown

    def GetStrSide(self):

        if self.Side is None:
            return _Side_Uknown
        elif self.Side==Side.Buy:
            return _Side_Buy
        elif self.Side == Side.Sell:
            return _Side_Sell
        elif self.Side == Side.SellShort:
            return _Side_Sell_Short
        elif self.Side == Side.BuyToClose:
            return _Side_Buy_To_Close
        else:
            return _Side_Uknown

    @staticmethod
    def FromStrSide(strSide):

        if strSide ==_Side_Uknown:
            return Side.Unknown
        elif strSide==_Side_Buy:
            return Side.Buy
        elif strSide == _Side_Sell:
            return Side.Sell
        elif strSide == _Side_Sell_Short:
            return Side.SellShort
        elif strSide == _Side_Buy_To_Close:
            return Side.BuyToClose
        else:
            return Side.Unknown


    def GetMarketArrivalTime(self):

        try:
            if self.GetLastOrder() is not None:
                if self.GetLastOrder().MarketArrivalTime is not None:
                    return self.GetLastOrder().MarketArrivalTime
                else:
                    return "1900-01-01"
            else:
                return "1900-01-01"
        except Exception as e:
            return "1900-01-01"


    def LoadNextClOrdId(self):
        ordCount = len(self.Orders)
        return  "{}{}{}".format(uuid.uuid4(),Position._ORDER_ID_SEPARATOR,str(ordCount))#first one is 0

    def LoadUpdateOrdNewClOrdId(self):

        lastOrder = self.GetLastOrder()
        posIdArr=  lastOrder.ClOrdId.split(Position._ORDER_ID_SEPARATOR)

        if len(posIdArr)==2:
            ordCount = len(self.Orders)
            return "{}{}{}".format(posIdArr[0], Position._ORDER_ID_SEPARATOR, str(ordCount))#Next index just adds one
        else:
            raise Exception("CRITICAL ERROR Generating New Order Id : Could not recognize the following ClOrdId as a valid ClOrdId:{}".format(lastOrder.ClOrdId))