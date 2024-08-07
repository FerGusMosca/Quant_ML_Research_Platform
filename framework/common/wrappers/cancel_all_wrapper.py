from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *

class CancelAllWrapper(Wrapper):
    def __init__(self):
        pass

        # region Public Methods

    def GetAction(self):
        """

        Returns:

        """
        return Actions.CANCEL_ALL_POSITIONS

    def GetField(self, field):
        """

        Args:
            field ():

        Returns:

        """
        return None

    # endregion
