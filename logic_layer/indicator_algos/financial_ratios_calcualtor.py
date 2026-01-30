from framework.common.logger.message_type import MessageType


class FinancialRatiosCalculator:
    """
    Pure computation layer for financial ratios.
    No I/O, no vendor logic, fully deterministic.
    """

    @staticmethod
    def compute(data, logger, job_id=None):
        """
        Compute a minimal set of financial ratios.
        All divisions are guarded to avoid runtime errors.
        """

        logger.do_log(
            "[RATIOS] Computing financial ratios",
            MessageType.INFO,
            job_id
        )

        if not data:
            logger.do_log(
                "[RATIOS] Empty fundamentals data received",
                MessageType.WARNING,
                job_id
            )
            return {}

        def safe_div(num, den, name):
            if num is None or den in (None, 0):
                logger.do_log(
                    f"[RATIOS] Cannot compute {name} | num={num}, den={den}",
                    MessageType.WARNING,
                    job_id
                )
                return None
            return num / den

        ratios = {
            "gpa": safe_div(
                data.get("gross_profit"),
                data.get("total_assets"),
                "GPA (Gross Profit / Assets)"
            ),
            "net_margin": safe_div(
                data.get("net_income"),
                data.get("revenue"),
                "Net Margin (Net Income / Revenue)"
            ),
            "debt_to_equity": safe_div(
                data.get("total_debt"),
                data.get("equity"),
                "Debt to Equity"
            ),
            "pe": data.get("pe"),
        }

        logger.do_log(
            f"[RATIOS] Computation completed | ratios={ratios}",
            MessageType.INFO,
            job_id
        )

        return ratios
