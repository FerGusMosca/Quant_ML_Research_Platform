from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *
from sources.framework.common.enums.fields.order_list_field import *
from sources.framework.business_entities.orders.order import *

class OrderListWrapper(Wrapper):
    def __init__(self,pOrders):
        self.Orders = pOrders

        # region Public Methods

    def GetAction(self):
        return Actions.ORDER_LIST

    def GetField(self, field):
        if field == None:
            return None


        if field == OrderListField.Orders:
            return self.Orders
        else:
            return None

    # endregion