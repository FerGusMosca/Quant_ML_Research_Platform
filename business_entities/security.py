"""
security.py
===========
Business entity representing a financial security (LECAP, BONCAP, SOVEREIGN, etc.).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Security:
    symbol:          str
    security_type:   str            # 'LECAP', 'BONCAP', 'SOVEREIGN', …
    description:     str
    maturity_date:   str            # ISO format 'YYYY-MM-DD'
    final_payment:   float          # pago final per 100 VN
    currency:        str = 'ARS'

    # Read-only fields populated by GetSecurities SP (not sent on writes)
    id:               Optional[int] = None
    is_active:        Optional[int] = None
    is_expired:       Optional[int] = None   # computed by SP: 1 if maturity_date < today
    days_to_maturity: Optional[int] = None   # computed by SP (negative if expired)
    created_at:       Optional[str] = None
    updated_at:       Optional[str] = None