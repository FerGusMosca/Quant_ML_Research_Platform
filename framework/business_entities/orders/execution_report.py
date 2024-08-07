from sources.framework.business_entities.securities.security import Security
from sources.framework.common.enums.ExecType import ExecType
from sources.framework.common.enums.OrdStatus import OrdStatus
from sources.framework.common.enums.OrdRejReason import OrdRejReason


class ExecutionReport:
    def __init__(self, TransactTime=None, LastFillTime = None, ExecType=None,ExecId =None , OrdStatus=None, Order=None, LastQty=None, LastPx=None,
                 LastMkt=None, LeavesQty=None, CumQty=None, AvgPx=None, Commission=None, Text=None, ArrivalPrice=None):
        # region Attributes
        self.TransactTime = TransactTime
        self.LastFillTime=LastFillTime
        self.ExecId = ExecId
        self.ExecType = ExecType
        self.OrdStatus = OrdStatus
        self.Order = Order
        self.LastQty = LastQty
        self.LastPx = LastPx
        self.LastMkt = LastMkt
        self.LeavesQty = LeavesQty
        self.CumQty = CumQty
        self.AvgPx = AvgPx
        self.Commission = Commission
        self.Text = Text
        self.ArrivalPrice=ArrivalPrice

        # endregion

    def IsDoneExecutionReport(self):
        return OrdStatus==OrdStatus.Filled or OrdStatus==OrdStatus.DoneForDay or OrdStatus==OrdStatus.Canceled \
               or OrdStatus==OrdStatus.Stopped or OrdStatus==OrdStatus.Rejected or OrdStatus==OrdStatus.Rejected \
               or OrdStatus==OrdStatus.Suspended or OrdStatus==OrdStatus.Calculated or OrdStatus==OrdStatus.Expired
