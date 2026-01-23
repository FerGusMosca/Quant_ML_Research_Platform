import os
import xml.etree.ElementTree as ET

from framework.common.logger.message_type import MessageType


class ThirteenFGraphProcessor:
    """
    Transforms raw SEC 13F XML filings into graph edges.
    Each filing produces edges of type:
        manager -> asset (HOLDS)
    """

    def __init__(self, logger, job_id):
        self.logger = logger
        self.job_id = job_id

    # --------------------------------------------------
    # Metadata extractors
    # --------------------------------------------------
    def _extract_manager(self, xml_file):
        """
        Try to extract manager from sidecar .meta JSON first.
        Fallback to XML <filingManager>. If both fail, return UNKNOWN_MANAGER.
        """
        meta_file = xml_file.replace(".xml", ".meta.json")

        # 1) Try sidecar metadata (preferred)
        if os.path.exists(meta_file):
            try:
                import json
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                name = meta.get("company")
                cik = meta.get("cik")
                if name and cik:
                    return f"{name.strip()}|CIK_{cik.strip()}"
                elif name:
                    return f"{name.strip()}"
                else:
                    raise  Exception("Missing Name | CIK")
            except Exception:
                pass  # Ignore and fallback to XML

        # 2) Fallback to XML filingManager
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            name = root.findtext(".//{*}filingManager/{*}name")
            cik = root.findtext(".//{*}filingManager/{*}cik")
            if name and cik:
                return f"{name.strip()}|CIK_{cik.strip()}"
        except Exception:
            pass

        # 3) Final fallback
        return "UNKNOWN_MANAGER"

    def _extract_positions(self, xml_file):
        positions = []
        file_name = os.path.basename(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for it in root.findall(".//{*}infoTable"):
            cusip = it.findtext("{*}cusip")
            issuer = it.findtext("{*}nameOfIssuer")  # ← SECURITY NAME
            value = float(it.findtext("{*}value", "0")) * 1000

            shares_node = it.find("{*}shrsOrPrnAmt/{*}sshPrnamt")
            shares = int(shares_node.text) if shares_node is not None else 0

            positions.append({
                "cusip": cusip,
                "security_name": issuer,
                "shares": shares,
                "value": value,
                "weight": value,
                "file": file_name
            })

        return positions

    # --------------------------------------------------
    # Single filing processing
    # --------------------------------------------------
    def _process_single_filing(self, xml_file, year, quarter):
        """
        Processes a single 13F XML filing and returns graph edges.

        Edge format:
        {
            src      : manager node
            dst      : asset node
            relation : HOLDS
            score    : position weight
            file     : source XML filename
            block_id : CUSIP
        }
        """
        edges = []

        manager_id = self._extract_manager(xml_file)
        positions = self._extract_positions(xml_file)

        self.logger.do_log(
            f"[13F] Parsed {len(positions)} positions | file={os.path.basename(xml_file)}",
            MessageType.INFO,
            self.job_id
        )

        for pos in positions:
            edges.append({
                "src": f"manager::{manager_id}",
                "dst": f"asset::{pos['security_name']}",
                "relation": "HOLDS",
                "score": pos["weight"],
                "file": os.path.basename(xml_file),
                "block_id": pos["cusip"]
            })

        return edges

    # --------------------------------------------------
    # Batch processing
    # --------------------------------------------------
    def process(self, raw_dir, year, quarter):
        """
        Processes all XML filings in a directory and aggregates edges.
        """
        edges = []
        files = [f for f in os.listdir(raw_dir) if f.endswith(".xml")]

        self.logger.do_log(
            f"[13F] Starting graph generation | files={len(files)} | year={year} | quarter={quarter}",
            MessageType.INFO,
            self.job_id
        )

        for file in files:
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

        self.logger.do_log(
            f"[13F] Graph generation completed | total_edges={len(edges)}",
            MessageType.INFO,
            self.job_id
        )

        return edges
