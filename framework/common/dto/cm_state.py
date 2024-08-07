class CMState:

    def __init__(self):
        self.Success = True
        self.Exception = None

    @staticmethod
    def BuildSuccess(self):
        """ Build a CMState object with success content.

        Returns:
            CMState object. Return CMState object with success info.
        """
        state = CMState()
        return state

    @staticmethod
    def BuildFailure(self, Exception=None, errorMsg=None):
        """ Build a CMState object with failure content.

        Args:
            Exception (:obj:`Wrapper`): Exception to be attached.
            errorMsg (String): Error message to be attached.

        Returns:
            CMState object. Return CMState object with exception message.
        """
        state = CMState()
        state.Success = False

        if Exception is not None:
            state.Exception = Exception
        elif errorMsg is not None:
            state.Exception = Exception(errorMsg)
        return state
