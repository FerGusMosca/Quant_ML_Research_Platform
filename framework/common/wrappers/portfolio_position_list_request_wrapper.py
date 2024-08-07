from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *

class PortfolioPositionListRequestWrapper(Wrapper):
    def __init__(self):
        pass

    # region Public Methods

    def GetAction(self):
        return Actions.PORTFOLIO_POSITIONS_REQUEST

    def GetField(self, field):
            return None

    # endregion