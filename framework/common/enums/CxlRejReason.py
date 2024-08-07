from enum import Enum


class CxlRejReason(Enum):
    TooLateToCancel = 0
    UnknownOrder = 1
    BrokerExchangeOption = 2
    OrderAlreadyPendingCancelOrPendingReplace = 3
    UnableProcessOrderMassCancelRequest = 4
    OrigOrdModTimeDidNotMatchLastTransactTime = 5
    DuplicateCLOrdId = 6
    Other = 7
    InvalidPriceIncrement = 8
    PriceExceedsCurrentPrice = 9
    PriceExceedsCurrentPriceBand = 10


