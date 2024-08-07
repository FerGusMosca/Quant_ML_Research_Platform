from enum import Enum
class SubscriptionRequestType(Enum):
    Snapshot = 0
    SnapshotAndUpdates = 1
    Unsuscribe = 2
