from enum import Enum
class SecurityType(Enum):
    CS="CS" # common
    FUT="FUT" # Future
    OPT="OPT" # Options
    IND="IND" # INDEX
    CASH="CASH" # Cash
    TBOND="TBOND" # Treasury bond
    TB="TBOND" # Treasury Bill
    OTH="OTH" # Other
