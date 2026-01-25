# ===== holdings_graph_manager.py =====

from neo4j import GraphDatabase
from typing import List, Dict


class HoldingsGraphManager:

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_pass: str,
        batch_size: int = 1000,
    ):
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_pass),
        )
        self.batch_size = batch_size

    def close(self):
        self.driver.close()

    # ---------- Cypher write ----------
    @staticmethod
    def _persist_batch(tx, rows: List[Dict], year: int, quarter: str):
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (m:Manager {name: row.manager})
            MERGE (a:Asset {cusip: row.cusip})
              ON CREATE SET a.name = row.asset_name
            MERGE (m)-[h:HOLDS {
                year: $year,
                quarter: $quarter
            }]->(a)
            SET h.weight = row.weight,
                h.file = row.file
            """,
            rows=rows,
            year=year,
            quarter=quarter,
        )

    # ---------- Public API ----------
    def persist(self, rows: List[Dict], year: int, quarter: str):
        with self.driver.session() as session:
            session.execute_write(
                self._persist_batch,
                rows,
                year,
                quarter,
            )
