from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *
from sources.framework.common.enums.TimeUnit import *
from sources.framework.common.enums.fields.historical_prices_request_field import *

class HistoricalPricesRequestWrapper(Wrapper):

    def __init__(self,pSecurity,pTime, pTimeUnit, pSubscriptionRequestType):
        self.Time=pTime
        self.TimeUnit = pTimeUnit
        self.Security=pSecurity
        self.SubscriptionRequestType = pSubscriptionRequestType

    # region Public Methods

    def GetAction(self):
        return Actions.HISTORICAL_PRICES_REQUEST

    def GetField(self, field):

        if field == HistoricalPricesRequestField.Security:
            return self.Security
        elif field == HistoricalPricesRequestField.TimeUnit:
            return self.TimeUnit
        elif field == HistoricalPricesRequestField.Time:
            return self.Time
        elif field == HistoricalPricesRequestField.SubscriptionRequestType:
            return self.SubscriptionRequestType
        else:
            return None

    # endregion
