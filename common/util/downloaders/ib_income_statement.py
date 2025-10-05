import os
import json
import time
import random
from ib_insync import IB, Stock
from common.enums.folders import Folders


class IBIncomeStatement:
    """
    Downloader for fundamental reports (Income Statement) from Interactive Brokers.
    Requires TWS or IB Gateway with API enabled (port default: 7497).
    """

    def __init__(self, host='127.0.0.1', port=7496, client_id=1):
        self.ib = IB()
        try:
            self.ib.connect(host, port, clientId=client_id)
            print("[IBIncomeStatement][INFO] ✅ Connected to IB API")
        except Exception as e:
            print(f"[IBIncomeStatement][ERROR] ❌ Failed to connect: {e}")

    def download(self, symbol, portfolio, report_type='ReportsFinStatements', pause=1.0):
        """
        Downloads fundamentals report for a given symbol.
        report_type options:
          - 'ReportsFinStatements' : Financial Statements (Income, Balance, Cash)
          - 'ReportSnapshot'        : Snapshot summary
        """
        print(f"\n[IBIncomeStatement][DEBUG] === Processing {symbol} ===")
        stock = Stock(symbol, 'SMART', 'USD')
        downloaded_files = []

        try:
            # pacing for API limits
            time.sleep(pause + random.random())

            # Request fundamentals
            xml_data = self.ib.reqFundamentalData(stock, reportType=report_type)

            if not xml_data:
                print(f"[IBIncomeStatement][WARN] No data for {symbol}")
                return []

            # Create output folder
            output_dir = f"{Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value}/{portfolio}/IB/{report_type}"
            os.makedirs(output_dir, exist_ok=True)

            # Save XML
            file_path = os.path.join(output_dir, f"{symbol}_{report_type}.xml")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(xml_data)

            print(f"[IBIncomeStatement][INFO] ✅ Saved {symbol} fundamentals -> {file_path}")
            downloaded_files.append(file_path)

        except Exception as e:
            print(f"[IBIncomeStatement][ERROR] ❌ {symbol} - {e}")

        return downloaded_files

    def disconnect(self):
        self.ib.disconnect()
        print("[IBIncomeStatement][INFO] 🔌 Disconnected from IB API")
