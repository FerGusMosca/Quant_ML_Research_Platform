from sources.framework.common.enums.SubscriptionRequestType import *
from sources.framework.common.enums.TimeUnit import *
class CandleBarRequest:
    def __init__(self):

        self.Security = None
        self.SubscriptionRequestType = None
        self.TimeUnit = None
        self.Time = None

