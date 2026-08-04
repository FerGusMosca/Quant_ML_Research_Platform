class SecSecurityMetadataDTO:
    """
    Data Transfer Object for the metadata returned by the SEC submissions
    endpoint, already classified by SICSectorClassifier.

    It does not replace SecSecurityDTO: that one is the security insert
    (cik/ticker/name), this one is the later enrichment.
    """

    def __init__(self, cik, symbol=None, sic=None, sic_description=None,
                 exchange=None, entity_type=None, fiscal_year_end=None,
                 state_of_incorporation=None, sector_code=None, sector_name=None,
                 industry_code=None, industry_name=None):
        self.cik = cik
        self.symbol = symbol
        self.sic = sic
        self.sic_description = sic_description
        self.exchange = exchange
        self.entity_type = entity_type
        self.fiscal_year_end = fiscal_year_end
        self.state_of_incorporation = state_of_incorporation
        self.sector_code = sector_code
        self.sector_name = sector_name
        self.industry_code = industry_code
        self.industry_name = industry_name

    def __str__(self):
        return (f"sic={self.sic} ({self.sic_description}) exch={self.exchange} "
                f"type={self.entity_type} -> {self.sector_code}/{self.industry_code}")
