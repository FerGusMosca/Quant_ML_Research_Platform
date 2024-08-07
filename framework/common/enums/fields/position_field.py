from enum import Enum
class PositionField(Enum):
    Symbol=1
    Exchange=2
    QuantityType=3
    PriceType=4
    Qty=5
    CashQty=6
    Percent=7
    ExecutionReports=8
    Orders=9
    Side=10
    PosId=11
    PositionRejectReason=12
    PositionRejectText=13
    PosStatus=14
    Security=15
    Currency=16
    SecurityType=17
    Account=18
    Broker=19
    Strategy=20
    OrderType=21
    OrderPrice=22
    StopLoss = 23
    TakeProfit = 24
    CloseEndOfDay = 25

    TradingMode= 99
    Depurate = 100
    CleanStopLoss=101
    CleanTakeProfit=102
    CleanEndOfDay = 103