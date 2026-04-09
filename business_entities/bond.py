"""
bond.py
=======
Business entities for sovereign bond detail and coupon schedule.
Goes in business_entities/bond.py
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BondDetail:
    security_id:   int
    symbol:        str
    security_type: str
    description:   str
    maturity_date: str           # 'YYYY-MM-DD'
    currency:      str
    is_active:     int
    law:           str           # 'NY' | 'Local'
    par_symbol:    Optional[str] = None  # counterpart bond, e.g. 'AL30' ↔ 'GD30'
    issuer:        Optional[str] = None  # e.g. 'YPF', 'Banco Galicia' — used by ONs


@dataclass
class BondCoupon:
    id:              Optional[int]
    security_id:     int
    symbol:          str
    payment_date:    str         # 'YYYY-MM-DD'
    amount:          float       # per 100 VN face value
    is_paid:         bool
    days_to_payment: Optional[int]  # negative if past