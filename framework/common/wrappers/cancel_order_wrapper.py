from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *
from sources.framework.common.enums.fields.order_field import *

class CancelOrderWrapper(Wrapper):
    def __init__(self,clOrdId,orderId):
        self.ClOrdId = clOrdId
        self.OrderId=orderId

        # region Public Methods

    def GetAction(self):
        return Actions.CANCEL_ORDER

    def GetField(self, field):

        if field == OrderField.ClOrdID:
            return self.ClOrdId
        elif field == OrderField.OrderId:
            return self.OrderId
        else:
            return None

    # endregion
