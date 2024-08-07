from sources.framework.common.logger.message_type import *
import datetime
class LogHelper:

    @staticmethod
    def LogPositionUpdate(logger, status, summary, execReport):
        logger.DoLog(
            "{}: Arrival={} TradeId={} PosId={} Symbol={} Side={} Final Status={} OrdQty={} CumQty={} LvsQty={} "
            "AvgPx={} Text={} "
                .format(status,
                        datetime.datetime.now(),
                        summary.GetTradeId(),
                        summary.Position.PosId,
                        summary.Position.Security.Symbol,
                        summary.Position.Side,
                        summary.Position.PosStatus,
                        summary.Position.Qty if summary.Position.Qty is not None else summary.Position.CashQty,
                        summary.CumQty,
                        summary.LeavesQty,
                        summary.AvgPx,
                        execReport.Text), MessageType.INFO)

    @staticmethod
    def LogPublishMarketDataOnSecurity(sender,logger,symbol,md):
        logger.DoLog("At {}: {}-Publishing market data for symbol {}: MDEntryDate={} Timestamp={} Last={} Bid={} Ask={} Mid={}".format(
            sender,
            datetime.datetime.now(),
            symbol,
            md.MDEntryDate if md.MDEntryDate is not None else "?",
            md.Timestamp if md.Timestamp is not None else "?",
            md.Trade if md.Trade is not None else "-",
            md.BestBidPrice if md.BestBidPrice is not None else "-",
            md.BestAskPrice if md.BestAskPrice is not None else "-",
            md.MidPrice if md.MidPrice is not None else "-"),
            MessageType.DEBUG)

    @staticmethod
    def LogPublishCandleBarOnSecurity(sender,logger,symbol,cb):
        logger.DoLog("At {}: {}-Publishing candle bar  for symbol {}: Time={} DateTime={} Timestamp={} Open={} High={} Low={} Last={}".format(
            sender,
            datetime.datetime.now(),
            symbol,
            cb.Time if cb.Time is not None else "-",
            cb.DateTime if cb.DateTime is not None else "-",
            cb.Timestamp if cb.Timestamp is not None else "-",
            cb.Open if cb.Open is not None else "-",
            cb.High if cb.High is not None else "-",
            cb.Low if cb.Low is not None else "-",
            cb.Close if cb.Close is not None else "-" ),
            MessageType.DEBUG)


    @staticmethod
    def LogNewOrder(logger,newOrder):
        logger.DoLog("Placing Order On Market: ClOrdId {} Symbol {} secType {} orderType {} totalQty {} "
                    .format(newOrder.ClOrdId, newOrder.Security.Symbol,
                            newOrder.Security.SecurityType,
                            newOrder.OrdType, newOrder.OrderQty), MessageType.INFO)