from sources.framework.common.enums.Actions import *
from sources.framework.common.enums.fields.market_data_request_field import *
from sources.framework.business_entities.securities.security import *
from sources.framework.common.enums.SubscriptionRequestType import *
from sources.framework.common.wrappers.wrapper import *

class MarketDataRequestWrapper(Wrapper):
    def __init__(self, pSecurity,pSubscriptionRequestType):
        self.Security = pSecurity
        self.SubscriptionRequestType=pSubscriptionRequestType

    def GetAction(self):
        return Actions.MARKET_DATA_REQUEST

    def GetField(self, field):
        if field == None:
            return None

        if self.Security == None:
            return None

        if field == MarketDataRequestField.Symbol:
            return self.Security.Symbol
        elif field == MarketDataRequestField.SecurityType:
            return self.Security.SecurityType
        elif field == MarketDataRequestField.Currency:
            return self.Security.Currency
        elif field == MarketDataRequestField.SubscriptionRequestType:
            return self.SubscriptionRequestType
        elif field == MarketDataRequestField.MDReqId:
            return 0
        else:
            return None
