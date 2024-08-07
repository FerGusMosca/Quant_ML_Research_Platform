from abc import ABC, abstractmethod


class Wrapper(ABC):

    @abstractmethod
    def GetField(self, OrderField):
        """

        Args:
            OrderField ():
        """
        pass

    @abstractmethod
    def GetAction(self):
        """

        """
        pass
