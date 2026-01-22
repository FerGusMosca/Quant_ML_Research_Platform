import os

from framework.common.logger.message_type import MessageType


class ThirteenFGraphProcessor:
    """
    Transforms raw 13F XML filings into graph edges.
    """

    def __init__(self, logger, job_id):
        self.logger = logger
        self.job_id = job_id

    def _extract_manager(self, xml_file):
        return "UNKNOWN_MANAGER"

    def _extract_positions(self, xml_file):
        """
        Returns list of:
        {
            ticker,
            cusip,
            shares,
            value,
            weight
        }
        """
        return []

    def _process_single_filing(self, xml_file, year, quarter):
        """
        Returns edges:
        manager -> asset
        manager -> sector (optional)
        asset -> theme (optional)
        """
        edges = []

        manager_id = self._extract_manager(xml_file)
        positions = self._extract_positions(xml_file)

        for pos in positions:
            edges.append({
                "src": f"manager::{manager_id}",
                "dst": f"asset::{pos['ticker']}",
                "relation": "HOLDS",
                "score": pos["weight"],
                "file": os.path.basename(xml_file),
                "block_id": pos["cusip"]
            })

        return edges

    def process(self, raw_dir, year, quarter):
        edges = []

        for file in os.listdir(raw_dir):
            if not file.endswith(".xml"):
                continue

            try:
                filing_edges = self._process_single_filing(
                    os.path.join(raw_dir, file),
                    year,
                    quarter
                )
                edges.extend(filing_edges)

            except Exception as e:
                self.logger.do_log(
                    f"[13F] ⚠️ Failed processing {file} | {str(e)}",
                    MessageType.WARNING,
                    self.job_id
                )

        return edges
