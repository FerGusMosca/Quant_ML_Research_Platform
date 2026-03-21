# instruction_manager.py
import pyodbc
from enum import Enum


class InstructionType(str, Enum):
    SYNC_POS = "SYNC_POS"


class SecType(str, Enum):
    OTH = "OTH"


class SyncBroker(str, Enum):
    """Brokers that support the SYNC_POS instruction."""
    IB_PROD = "IB_PROD"
    IB_DEV  = "IB_DEV"

    @classmethod
    def is_supported(cls, broker: str) -> bool:
        return broker in {b.value for b in cls}


class InstructionManager:
    """
    Data Access Layer for AutPortfolio.dbo.instructions.
    Uses the same connection string as IBPortfolioManager
    (points to AutPortfolio DB).
    """

    def __init__(self, connection_string: str):
        self._cs = connection_string

    def _connect(self) -> pyodbc.Connection:
        return pyodbc.connect(self._cs)

    def create_sync_instruction(self, account_id: int) -> int:
        """
        Inserts a SYNC_POS instruction for the given account.
        Returns the new instruction id.
        Calls: dbo.create_sync_instruction
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC dbo.create_sync_instruction @account_id = ?",
                account_id
            )
            row = cursor.fetchone()
            conn.commit()
            return int(row.id)