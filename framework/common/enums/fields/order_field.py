from enum import Enum
class OrderField(Enum):
    SettlType=1
    SettlDate=2
    Symbol=3
    SecurityType=4
    Currency=5
    Exchange=6
    ClOrdID=7
    OrdType=8
    PriceType=9
    Price=10
    StopPx=11
    ExpireDate=12
    ExpireTime=13
    Side=14
    OrderQty=15
    CashOrderQty=16
    OrderPercent=17
    TimeInForce=18
    MinQty=19
    OrigClOrdID=20
    Account=21
    Broker=22
    Strategy=23
    OrderId=24