from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *
from sources.framework.common.enums.fields.error_field import *

class ErrorWrapper(Wrapper):
    def __init__(self,pException):
        self.Exception=pException

        # region Public Methods

    def GetAction(self):
        return Actions.ERROR

    def GetField(self, field):


        if field == ErrorField.Exception:
            return self.Exception
        elif field == ErrorField.ErrorMessage:
            return str(self.Exception)
        else:
            return None

    # endregion
