class Account:
    def __init__(self, account_number: str, account_name: str, broker: str, id: int | None = None):
        self.id = id
        self.account_number = account_number
        self.account_name = account_name
        self.broker = broker