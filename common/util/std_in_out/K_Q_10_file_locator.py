from common.enums.sec_reports import SECReports


class KQ10FileLocator():


    @staticmethod
    def find_file(source,file,symbol, year, quarter):
        if SECReports.K10.value in source:
            # e.g. HD_2025_10-K.html
            return file.startswith(f"{symbol}_{year}_10-K")

        if SECReports.Q10.value in source:
            # e.g. GPI_2025_Q1_10-Q.html
            return file.startswith(f"{symbol}_{year}_{quarter}_10-Q")