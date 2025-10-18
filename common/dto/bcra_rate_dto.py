from dataclasses import dataclass
from datetime import date

@dataclass
class BCRARateDTO:
    name: str
    date: date
    value: float
