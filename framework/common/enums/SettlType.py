from enum import Enum
class SettlType(Enum):
    Regular = '0'
    Cash = '1'
    NextDay = '2'
    Tplus2 = '3'
    Tplus3 = '4'
    Tplus4 = '5'
    Future = '6'
    Tplus5 = '9'
