from datetime import datetime


class DateHandler:


    @staticmethod
    def convert_str_date(date,format):
        try:
            converted_date = datetime.strptime(date, format)
            return  converted_date
        except Exception as e:
            raise Exception("Could not convert date {} to datetime w/format {}".format(date,format))