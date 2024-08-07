from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *
from sources.framework.common.enums.TimeUnit import *
from sources.framework.common.enums.fields.candle_bar_request_field import *

class CandleBarRequestWrapper(Wrapper):

    def __init__(self,pSecurity,pTime, pTimeUnit, pSubscriptionRequestType):
        self.Time=pTime
        self.TimeUnit = pTimeUnit
        self.Security=pSecurity
        self.SubscriptionRequestType = pSubscriptionRequestType

    # region Public Methods

    def GetAction(self):
        return Actions.CANDLE_BAR_REQUEST

    def GetField(self, field):

        if field == CandleBarRequestField.Security:
            return self.Security
        elif field == CandleBarRequestField.TimeUnit:
            return self.TimeUnit
        elif field == CandleBarRequestField.Time:
            return self.Time
        elif field == CandleBarRequestField.SubscriptionRequestType:
            return self.SubscriptionRequestType
        else:
            return None

    # endregion
