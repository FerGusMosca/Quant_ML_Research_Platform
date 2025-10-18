from dataclasses import dataclass
from datetime import date

@dataclass
class BYMARateDTO:
    """
    Data Transfer Object representing a single BYMA interest rate entry.
    """
    name: str       # Descriptive name of the rate (e.g. "Cauciones a 1 día")
    date: date      # Date of the observation
    value: float    # Rate value (percentage)
