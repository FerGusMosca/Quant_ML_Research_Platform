from enum import Enum
class PriceType(Enum):
    Percentage = 1,
    FixedAmount = 3,
    Yield = 9,
    Discount = 4,
    Premium = 5
