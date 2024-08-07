from enum import Enum


class CxlRejResponseTo(Enum):
    OrderCancelRequest = 'F'
    OrderCancelReplaceRequest = 'G'

