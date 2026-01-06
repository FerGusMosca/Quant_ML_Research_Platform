from framework.common.logger.message_type import MessageType


class DateRangeHandler:
    @staticmethod
    def handle_date_range(year,logger):
        if "-" in str(year):
            try:
                start_year, end_year = map(int, str(year).split("-"))
                years = list(range(start_year, end_year + 1))
                logger.do_log(f"[SENT] 📆 Detected year range {start_year}-{end_year}", MessageType.INFO)
                return years
            except Exception as e:
                logger.do_log(f"[SENT] ❌ Invalid year format '{year}' Error: {e}", MessageType.ERROR)
                raise
        else:
            years = [int(year)]

        return years