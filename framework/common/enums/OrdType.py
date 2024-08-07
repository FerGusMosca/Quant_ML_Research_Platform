from enum import Enum
class OrdType(Enum):
    Market = '1'
    Limit = '2'
    Stop = '3'
    StopLimit = '4'
    MarketOnClose = '5'
    LimitOnClose = 'B'
