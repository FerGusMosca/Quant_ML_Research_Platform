from enum import Enum
class OrderCancelRejectField(Enum):
    ClOrdID=1
    OrigClOrdID=2
    OrderID=3
    OrdStatus=4
    CxlRejResponseTo=5
    CxlRejReason=6
    Text=8
