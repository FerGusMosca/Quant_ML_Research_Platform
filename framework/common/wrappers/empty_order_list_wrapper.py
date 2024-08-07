from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *
from sources.framework.common.enums.fields.order_list_field import *

class EmtpyOrderListWrapper(Wrapper):
    def __init__(self):
        self.Orders = [] #OrderWrapper list

        # region Public Methods

    def GetAction(self):
        """

        Returns:

        """
        return Actions.ORDER_LIST

    def GetField(self, field):
        """

        Args:
            field ():

        Returns:

        """
        if field == None:
            return None


        if field == OrderListField.Orders:
            return self.Orders
        else:
            return None

    # endregion