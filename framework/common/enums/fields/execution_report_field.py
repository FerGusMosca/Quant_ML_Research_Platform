from enum import Enum


class ExecutionReportField(Enum):
    OrderID = 1
    ClOrdID = 2
    ExecType = 3
    OrdStatus = 4
    OrdRejReason = 5
    Symbol = 6
    OrderQty = 7
    CashOrderQty = 8
    OrdType = 9
    Price = 10
    StopPx = 11
    Currency = 12
    ExpireDate = 13
    LeavesQty = 14
    CumQty = 15
    AvgPx = 16
    Commission = 17
    MinQty = 18
    Text = 19
    TransactTime = 20
    LastQty = 21
    LastPx = 22
    LastMkt = 23
    ExecID = 24
    Side = 25
    QuantityType = 26
    PriceType = 27
    OrigClOrdID = 28
    Account = 29
    ExecInst = 30
    TimeInForce = 31
    PosId = 32
    Order = 33
    ArrivalPrice = 34
    MarketArrivalTime=35
    LastFillTime = 36
