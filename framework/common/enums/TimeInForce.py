from enum import Enum
class TimeInForce(Enum):
    Day = '0'
    GoodTillCancel = '1'
    AtTheOpening = '2'
    ImmediateOrCancel = '3'
    FillOrFill = '4'
    GoodTillCrossing = '5'
    GoodTillDate = '6'
    AtTheClose = '7'
