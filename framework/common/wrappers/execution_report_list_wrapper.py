from sources.framework.common.enums.Actions import *
from sources.framework.common.wrappers.wrapper import *
from sources.framework.common.enums.fields.execution_report_list_field import *
from sources.framework.business_entities.orders.order import *

class ExecutionReportListWrapper(Wrapper):
    def __init__(self,pExecutionReportWrappers):
        self.ExecutionReportWrappers = pExecutionReportWrappers

        # region Public Methods

    def GetAction(self):
        return Actions.EXECUTION_REPORT_LIST

    def GetField(self, field):

        if field == None:
            return None


        if field == ExecutionReportListField.ExecutionReports:
            return self.ExecutionReportWrappers
        else:
            return None

    # endregion