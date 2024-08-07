from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *
from sources.framework.common.enums.fields.portfolio_positions_trade_list_request_field import *

class PortfolioPositionTradeListRequestWrapper(Wrapper):
    def __init__(self, portfPosId):
        self.PortfPosId =portfPosId

    # region Public Methods

    def GetAction(self):
        return Actions.PORTFOLIO_POSITION_TRADE_LIST_REQUEST

    def GetField(self, field):
        if field == PortfolioPositionTradeListRequestFields.PositionId:
            return self.PortfPosId
        else:
            return None

    # endregion