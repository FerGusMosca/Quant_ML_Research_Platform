from datetime import datetime

import pandas as pd

from common.enums.months import Months
from datetime import date, datetime

class DateHandler:

    def std_date(d) -> date:
        """
        Standardize any date/time-like object to a datetime.date.

        - If it's a datetime or pandas.Timestamp -> returns d.date().
        - If it's already a date -> returns it unchanged.
        - If it has a .date() method -> calls it.
        - Otherwise returns the object as-is (fallback).
        """
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if hasattr(d, "date"):  # e.g. pandas.Timestamp
            return d.date()
        return d  # fallback

    @staticmethod
    def norm_date(df, col="date"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None).dt.normalize()
        return df

    @staticmethod
    def get_two_month_period_from_date(date):
        """
        Return a two-month period label (e.g., 'Jan-Feb', 'Mar-Apr') based on the month of the given date.
        """
        month = date.month

        if 1 <= month <= 2:
            start, end = Months.JAN, Months.FEB
        elif 3 <= month <= 4:
            start, end = Months.MAR, Months.APR
        elif 5 <= month <= 6:
            start, end = Months.MAY, Months.JUN
        elif 7 <= month <= 8:
            start, end = Months.JUL, Months.AUG
        elif 9 <= month <= 10:
            start, end = Months.SEP, Months.OCT
        else:
            start, end = Months.NOV, Months.DEC

        return f"{start.label}-{end.label}"

    @staticmethod
    def get_period_label_from_dates(start_date: datetime, end_date: datetime) -> str:
        """
        Return a string like 'Jan-Feb' or 'Jan-Dec' from two dates.
        If dates span different years, include the year in the label.

        Args:
            start_date (datetime): Period start date
            end_date (datetime): Period end date

        Returns:
            str: Period label
        """
        start_label = Months.label_from_number(start_date.month)
        end_label = Months.label_from_number(end_date.month)

        if start_date.year != end_date.year:
            return f"{start_label} {start_date.year} - {end_label} {end_date.year}"

        if start_label == end_label:
            return f"{start_label} {start_date.year}"
        return f"{start_label}-{end_label} {start_date.year}"

    @staticmethod
    def evaluate_consecutive_days(day_1, day_2):

        if abs((day_2 - day_1).days) == 1:
            return True
        else:
            if (abs((day_2 - day_1).days) ==3) and day_1.weekday()==4 and day_2.weekday()==0 : #Friday To Monday
                return  True
            else:
                return  False


    @staticmethod
    def convert_str_date(date,format):
        try:
            converted_date = datetime.strptime(date, format)
            return  converted_date
        except Exception as e:
            raise Exception("Could not convert date {} to datetime w/format {}".format(date,format))