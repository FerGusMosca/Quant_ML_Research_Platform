class AccountData:
    """
    Business Entity for a single key/value entry in dbo.account_data.

    One account can have N AccountData entries — any key string is valid.
    Examples:
        AccountData(account_id=1, data_key="legacy_id",   data_value="12345")
        AccountData(account_id=1, data_key="legacy_name", data_value="FERN_M")
        AccountData(account_id=2, data_key="fix_sender_comp_id", data_value="CLIENT1")
        AccountData(account_id=2, data_key="fix_target_comp_id", data_value="BROKER")
    """

    def __init__(
        self,
        account_id:   int,
        data_key:     str,
        data_value:   str,
        data_id:      int | None = None,
        # Denormalised fields — populated by JOINed SPs
        account_number: str | None = None,
        account_name:   str | None = None,
        broker:         str | None = None,
    ):
        self.data_id        = data_id        # PK of account_data row (None if not yet persisted)
        self.account_id     = account_id
        self.data_key       = data_key
        self.data_value     = data_value
        self.account_number = account_number
        self.account_name   = account_name
        self.broker         = broker

    def __repr__(self) -> str:
        return (
            f"AccountData(account_id={self.account_id!r}, "
            f"data_key={self.data_key!r}, data_value={self.data_value!r})"
        )