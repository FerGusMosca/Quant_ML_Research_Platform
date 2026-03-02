class PortfolioHolding:
    """
    Business Entity for a single portfolio holding row.

    Populated by IBPortfolioManager.fetch_ib_account_holdings()
    via dbo.get_portfolio_holdings.
    """

    def __init__(
        self,
        symbol:          str,
        name:            str | None,
        qty:             float,
        purchase_price:  float | None,
        purchase_amount: float | None,
    ):
        self.symbol          = symbol
        self.name            = name or symbol
        self.qty             = qty
        self.purchase_price  = purchase_price
        self.purchase_amount = purchase_amount

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "name":            self.name,
            "qty":             self.qty,
            "purchase_price":  self.purchase_price,
            "purchase_amount": self.purchase_amount,
        }

    def __repr__(self) -> str:
        return (
            f"PortfolioHolding(symbol={self.symbol!r}, qty={self.qty}, "
            f"purchase_amount={self.purchase_amount})"
        )