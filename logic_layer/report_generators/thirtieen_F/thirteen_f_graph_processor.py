import os
import xml.etree.ElementTree as ET

from framework.common.logger.message_type import MessageType


class ThirteenFGraphProcessor:
    """
    Transforms raw SEC 13F XML filings into graph edges.
    Streams directly to Neo4j instead of accumulating in memory.
    """

    def __init__(self, logger, job_id, holdings_manager, year: int, quarter: str):
        self.logger = logger
        self.job_id = job_id
        self.holdings_manager = holdings_manager
        self.year = year
        self.quarter = quarter

        # Internal batch buffer
        self._batch = []
        self._total_persisted = 0
        self._estimated_total = 0  # For progress tracking

    # --------------------------------------------------
    # Pre-scan for progress estimation
    # --------------------------------------------------
    def _estimate_total_positions(self, raw_dir: str, files: list) -> int:
        """
        Quick scan to count <infoTable> elements across all files.
        Uses iterparse for memory efficiency - doesn't load full DOM.
        """
        total = 0
        for file in files:
            try:
                path = os.path.join(raw_dir, file)
                # Count infoTable tags without loading full tree
                for event, elem in ET.iterparse(path, events=["end"]):
                    if elem.tag.endswith("infoTable"):
                        total += 1
                        elem.clear()  # Free memory immediately
            except Exception:
                pass  # Skip problematic files in estimation
        return total

    # --------------------------------------------------
    # Metadata extractors (unchanged)
    # --------------------------------------------------
    def _extract_manager(self, xml_file):
        meta_file = xml_file.replace(".xml", ".meta.json")

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
                    raise Exception("Missing Name | CIK")
            except Exception:
                pass

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            name = root.findtext(".//{*}filingManager/{*}name")
            cik = root.findtext(".//{*}filingManager/{*}cik")
            if name and cik:
                return f"{name.strip()}|CIK_{cik.strip()}"
        except Exception:
            pass

        return "UNKNOWN_MANAGER"

    def _extract_positions(self, xml_file):
        positions = []
        file_name = os.path.basename(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for it in root.findall(".//{*}infoTable"):
            cusip = it.findtext("{*}cusip")
            issuer = it.findtext("{*}nameOfIssuer")
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
    # Batch management
    # --------------------------------------------------
    def _flush_batch(self):
        """Persist current batch to Neo4j and clear buffer."""
        if not self._batch:
            return

        self.holdings_manager.persist(self._batch, self.year, self.quarter)
        self._total_persisted += len(self._batch)

        # Progress with percentage
        if self._estimated_total > 0:
            pct = (self._total_persisted / self._estimated_total) * 100
            self.logger.do_log(
                f"[13F] Persisted {self._total_persisted:,} / {self._estimated_total:,} edges ({pct:.1f}%)",
                MessageType.INFO,
                self.job_id
            )
        else:
            self.logger.do_log(
                f"[13F] Persisted {self._total_persisted:,} edges",
                MessageType.INFO,
                self.job_id
            )

        self._batch.clear()

    def _add_edge(self, edge: dict):
        """Add edge to batch, flush if batch is full."""
        self._batch.append({
            "manager": edge["src"].replace("manager::", ""),
            "cusip": edge["block_id"],
            "asset_name": edge["dst"].replace("asset::", ""),
            "weight": edge.get("score", 0),
            "file": edge.get("file"),
        })

        if len(self._batch) >= self.holdings_manager.batch_size:
            self._flush_batch()

    # --------------------------------------------------
    # Single filing processing
    # --------------------------------------------------
    def _process_single_filing(self, xml_file):
        """Process a single 13F XML filing and stream edges to batch."""
        manager_id = self._extract_manager(xml_file)
        positions = self._extract_positions(xml_file)

        for pos in positions:
            self._add_edge({
                "src": f"manager::{manager_id}",
                "dst": f"asset::{pos['security_name']}",
                "relation": "HOLDS",
                "score": pos["weight"],
                "file": os.path.basename(xml_file),
                "block_id": pos["cusip"]
            })

        return len(positions)

    # --------------------------------------------------
    # Main entry point
    # --------------------------------------------------
    def process(self, raw_dir: str) -> int:
        """
        Processes all XML filings in a directory.
        Streams directly to Neo4j - does NOT accumulate in memory.

        Returns: total edges persisted
        """
        files = [f for f in os.listdir(raw_dir) if f.endswith(".xml")]

        self.logger.do_log(
            f"[13F] Scanning {len(files)} files to estimate total...",
            MessageType.INFO,
            self.job_id
        )

        # Quick pre-scan for progress estimation
        self._estimated_total = self._estimate_total_positions(raw_dir, files)

        self.logger.do_log(
            f"[13F] Starting graph generation | files={len(files)} | estimated_edges={self._estimated_total:,}",
            MessageType.INFO,
            self.job_id
        )

        for file in files:
            try:
                self._process_single_filing(os.path.join(raw_dir, file))
            except Exception as e:
                self.logger.do_log(
                    f"[13F] ⚠️ Failed processing {file} | {str(e)}",
                    MessageType.WARNING,
                    self.job_id
                )

        # Flush remaining edges
        self._flush_batch()

        self.logger.do_log(
            f"[13F] ✅ Graph generation completed | total_edges={self._total_persisted:,}",
            MessageType.INFO,
            self.job_id
        )

        return self._total_persisted