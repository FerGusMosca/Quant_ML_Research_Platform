import xml.etree.ElementTree as ET
import os


class Form4Processor:
    """
    Parses Form 4 XML files to identify significant insider transactions.
    Focuses on 'P' (Open Market Purchase) vs 'S' (Open Market Sale).
    """

    def __init__(self, logger, job_id):
        self.logger = logger
        self.job_id = job_id

    def process_file(self, xml_path):
        """
        Parses a single XML Form 4 and returns a list of actionable trades.
        """
        if not os.path.exists(xml_path):
            return []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract Insider Identity
        insider_name = root.find(".//reportingOwnerName").text if root.find(
            ".//reportingOwnerName") is not None else "Unknown"
        is_dir = root.find(".//isDirector").text == '1' if root.find(".//isDirector") is not None else False
        is_off = root.find(".//isOfficer").text == '1' if root.find(".//isOfficer") is not None else False
        officer_title = root.find(".//officerTitle").text if root.find(".//officerTitle") is not None else "N/A"

        trades = []
        # Non-Derivative transactions are usually standard stock buys/sells
        for trans in root.findall(".//nonDerivativeTransaction"):
            try:
                security_title = trans.find(".//securityTitle/value").text
                date = trans.find(".//transactionDate/value").text
                code = trans.find(".//transactionCode").text  # P=Purchase, S=Sale
                shares = float(trans.find(".//transactionShares/value").text)
                price = float(trans.find(".//transactionPricePerShare/value").text or 0)

                # Signal Logic: We care most about direct cash purchases (Bullish)
                is_bullish = (code == 'P')

                trades.append({
                    "insider": insider_name,
                    "title": officer_title if is_off else ("Director" if is_dir else "Major Owner"),
                    "security": security_title,
                    "date": date,
                    "type": "PURCHASE" if code == 'P' else ("SALE" if code == 'S' else "OTHER"),
                    "shares": shares,
                    "price": price,
                    "value": round(shares * price, 2),
                    "is_bullish_signal": is_bullish
                })
            except Exception as e:
                continue  # Skip malformed transaction entries

        return trades