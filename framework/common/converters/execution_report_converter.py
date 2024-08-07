from sources.framework.business_entities.orders.execution_report import *
from sources.framework.business_entities.securities.security import *
from sources.framework.common.enums.fields.execution_report_field import *


class ExecutionReportConverter:

    @staticmethod
    def ValidateExecutionReport(wrapper):
        """ Validate execution report from wrapper object.

        Args:
            wrapper (:obj:`Wrapper`): Generic wrapper to communicate strategy with other modules.
        """
        if wrapper.GetField(ExecutionReportField.ExecType) is None:
            raise Exception("Missing parameter {} for execution report".format(ExecutionReportField.ExecType))

        if wrapper.GetField(ExecutionReportField.OrdStatus) is None:
            raise Exception("Missing parameter {} for execution report".format(ExecutionReportField.OrdStatus))

        if wrapper.GetField(ExecutionReportField.LeavesQty) is None:
            raise Exception("Missing parameter {} for execution report".format(ExecutionReportField.LeavesQty))

        if wrapper.GetField(ExecutionReportField.CumQty) is None:
            raise Exception("Missing parameter {} for execution report".format(ExecutionReportField.CumQty))

        if wrapper.GetField(ExecutionReportField.OrderID) is None:
            raise Exception("Missing parameter {} for execution report".format(ExecutionReportField.OrderID))

        if wrapper.GetField(ExecutionReportField.Symbol) is None:
            raise Exception("Missing parameter {} for execution report".format(ExecutionReportField.Symbol))

    @staticmethod
    def ConvertExecutionReport(wrapper):
        """ Convert wrapper object to execution report.

        Args:
            wrapper (:obj:`Wrapper`): Generic wrapper to communicate strategy with other modules.

        Returns:
            ExecutionReport object. Return an Execution report from Order Routers
        """
        ExecutionReportConverter.ValidateExecutionReport(wrapper)

        execReport = ExecutionReport()

        execReport.Security = Security()
        execReport.Security.Symbol = wrapper.GetField(ExecutionReportField.Symbol)

        execReport.TransactTime = wrapper.GetField(ExecutionReportField.TransactTime)
        execReport.ExecType = wrapper.GetField(ExecutionReportField.ExecType)
        execReport.ExecId = wrapper.GetField(ExecutionReportField.ExecID)
        execReport.OrdStatus = wrapper.GetField(ExecutionReportField.OrdStatus)
        execReport.OrdRejReason = wrapper.GetField(ExecutionReportField.OrdRejReason)
        execReport.LeavesQty = wrapper.GetField(ExecutionReportField.LeavesQty)
        execReport.CumQty = wrapper.GetField(ExecutionReportField.CumQty)
        execReport.AvgPx = wrapper.GetField(ExecutionReportField.AvgPx)
        execReport.Commission = wrapper.GetField(ExecutionReportField.Commission)
        execReport.Text = wrapper.GetField(ExecutionReportField.Text)
        execReport.TransactTime = wrapper.GetField(ExecutionReportField.TransactTime)
        execReport.LastQty = wrapper.GetField(ExecutionReportField.LastQty)
        execReport.TransactTime = wrapper.GetField(ExecutionReportField.TransactTime)
        execReport.LastPx = wrapper.GetField(ExecutionReportField.LastPx)
        execReport.LastMkt = wrapper.GetField(ExecutionReportField.LastMkt)

        execReport.OrderId = wrapper.GetField(ExecutionReportField.OrderID)
        execReport.ClOrdId = wrapper.GetField(ExecutionReportField.ClOrdID)
        execReport.OrderQty = wrapper.GetField(ExecutionReportField.OrderQty)
        execReport.CashOrderQty = wrapper.GetField(ExecutionReportField.CashOrderQty)
        execReport.OrdType = wrapper.GetField(ExecutionReportField.OrdType)
        execReport.Price = wrapper.GetField(ExecutionReportField.Price)
        execReport.StopPx = wrapper.GetField(ExecutionReportField.StopPx)
        execReport.Currency = wrapper.GetField(ExecutionReportField.Currency)
        execReport.MinQty = wrapper.GetField(ExecutionReportField.MinQty)
        execReport.Side = wrapper.GetField(ExecutionReportField.Side)
        # execReport.QuantityType = wrapper.GetField(ExecutionReportField.QuantityType)
        execReport.PriceType = wrapper.GetField(ExecutionReportField.PriceType)
        execReport.ArrivalPrice = wrapper.GetField(ExecutionReportField.ArrivalPrice)
        execReport.LastFillTime = wrapper.GetField(ExecutionReportField.LastFillTime)

        execReport.Order = wrapper.GetField(ExecutionReportField.Order)

        return execReport
