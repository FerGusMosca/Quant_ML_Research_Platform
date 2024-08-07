from abc import ABC, abstractmethod


class ICommunicationModule(ABC):

    @abstractmethod
    def ProcessOutgoing(self, wrapper):
        """ Receives a response from another module that is invoked.

        Args:
            wrapper (:obj:`Wrapper`): Generic wrapper to communicate strategy with other modules.

        Returns:
            CMState object. The return value. BuildSuccess for success.
        """
        print("ProcessOutgoing abstractmethod")

    @abstractmethod
    def ProcessMessage(self, wrapper):
        """ Receives messages from “invoking” modules.

         Args:
             wrapper (:obj:`Wrapper`): Generic wrapper to communicate strategy with other modules.

         Returns:
             CMState object. The return value. BuildSuccess for success, BuildFailure otherwise.
         """
        print("ProcessMessage abstractmethod")

    @abstractmethod
    def Initialize(self, pInvokingModule, pConfigFile):
        """  initialize everything

        Args:
            pInvokingModule (ProcessOutgoing method): Invoking module.
            pConfigFile (Ini file): Configuration file path.

        Returns:
            CMState object. The return value. BuildSuccess for success, BuildFailure otherwise.
        """
        print("Initialize abstractmethod")