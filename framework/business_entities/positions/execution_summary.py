from sources.framework.business_entities.positions.position import Position
from sources.framework.business_entities.orders.execution_report import *
import datetime
from sources.framework.common.enums.Side import *

_TRADE_ID_PREFIX="trd_"
_MAIN_SUMMARY="MAIN_SUMM"
_INNER_SUMMARY_2="SUMM_2"
_INNER_SUMMARY_3="SUMM_3"

class ExecutionSummary:
    def __init__(self, Date, Position):
        self.Date = Date
        self.AvgPx = None
        self.CumQty = 0
        self.LeavesQty = Position.Qty
        self.InitialPrice = Position.OrderPrice
        self.Commission = None
        self.Text = None
        self.Position = Position
        self.LastUpdateTime =datetime.datetime.now()
        self.Timestamp = datetime.datetime.now()
        self.LastTradeTime = None
        self.CreateTime= datetime.datetime.now()
        self.InnerSummaries={}
        self.SummaryHierarchy=None

    #region Public Methods

    def UpdateStatus(self, execReport,marketDataToUse=None):

        self.CumQty = execReport.CumQty
        self.AvgPx = execReport.AvgPx if marketDataToUse is None else marketDataToUse.Trade
        self.Commission = execReport.Commission
        self.Text = execReport.Text if execReport.Text is not None and execReport.Text!="" else self.Text

        self.Position.LeavesQty = execReport.LeavesQty
        self.Position.CumQty=execReport.CumQty
        self.Position.AvgPx=execReport.AvgPx

        self.Position.SetPositionStatusFromExecution(execReport)
        self.Position.ExecutionReports.append(execReport)
        self.LeavesQty = execReport.LeavesQty if self.Position.IsOpenPosition() else 0

        self.LastUpdateTime =  datetime.datetime.now()
        if(execReport.LastFillTime is not None):
            self.Timestamp = execReport.LastFillTime

        if execReport.ArrivalPrice is not None:
            self.Position.ArrivalPrice=execReport.ArrivalPrice

        self.LastTradeTime=execReport.LastFillTime

        if execReport.Order is not None:
            self.Position.AppendOrder(execReport.Order)

    def GetTradedSummary(self):
        if self.CumQty>0 and self.AvgPx is not None:
            return self.CumQty*self.AvgPx
        else:
            return 0

    def GetNetShares(self):
        return self.CumQty if self.CumQty is not None else 0

    def SharesAcquired(self):
        return self.Position.Side==Side.Buy or self.Position.Side==Side.BuyToClose

    def GetTradeId(self):
        if(self.Position is None):
            raise Exception("Could not save execution summary without position")

        orderId=None

        if (self.Position.IsRejectedPosition()):
            orderId= "R" + str( self.CreateTime.timestamp()) if self.CreateTime is not None else "R?"
        elif(self.Position.GetLastOrder()is None):
            orderId="nO" + str( self.CreateTime.timestamp()) if self.CreateTime is not None else "nO?"
        else:
            orderId=self.Position.GetLastOrder().OrderId

        if(self.SummaryHierarchy is not None):
            orderId="{}_{}".format(str(orderId),self.SummaryHierarchy)

        return "{}_{}_{}".format(_TRADE_ID_PREFIX,self.Position.Security.Symbol,orderId)


    def DoInnerTradesExist(self):
        return self.IsFirstInnerTradeOpen() or self.IsSecondInnerTradeOpen()

    def IsFirstInnerTradeOpen(self):
        return _INNER_SUMMARY_2 in self.InnerSummaries

    def IsSecondInnerTradeOpen(self):
        return _INNER_SUMMARY_3 in self.InnerSummaries

    def GetFirstInnerSummary(self):
        return self.InnerSummaries[_INNER_SUMMARY_2]

    def GetSecondInnerSummary(self):
        return self.InnerSummaries[_INNER_SUMMARY_3]

    def AppendFirstInnerSummary(self,summary):
        summary.SummaryHierarchy=_INNER_SUMMARY_2
        self.InnerSummaries[_INNER_SUMMARY_2]=summary

    def AppendSecondInnerSummary(self,summary):
        summary.SummaryHierarchy = _INNER_SUMMARY_3
        self.InnerSummaries[_INNER_SUMMARY_3]=summary

    #endregion

    #region Static Attributes

    @staticmethod
    def _MAIN_SUMMARY():
        return _MAIN_SUMMARY

    @staticmethod
    def _INNER_SUMMARY_2():
        return _INNER_SUMMARY_2

    @staticmethod
    def _INNER_SUMMARY_3():
        return _INNER_SUMMARY_3


    #endregion
