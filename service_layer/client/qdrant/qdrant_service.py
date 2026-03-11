import traceback
from typing import Optional, Union

import requests

from business_entities.qdrant.chunk_management_entities import (
    CollectionInfo, ScrollResult, SearchResult,
    ChunkPoint,
)
from business_entities.qdrant.metadata_point import MetadataPoint

KNOWN_COLLECTIONS = ["zh_chunks", "zh_metadata"]

# Qdrant REST API pagination notes:
# - order_by + start_from: used for zh_chunks (epoch-ms cursor). next_page_offset
#   is always null when order_by is active — Qdrant does not support both at once.
# - UUID offset: used for all other collections. next_page_offset is a UUID
#   returned by Qdrant and passed directly as "offset" on the next request.
# - scroll_all() always uses UUID-based offset to traverse entire collections.


class QdrantService:
    """
    Encapsulates all communication with the Qdrant vector database.
    Uses the Qdrant REST API directly (no SDK dependency).
    """

    def __init__(self, config_settings: dict, logger):
        self.host     = config_settings.get("QDRANT_SERVER", "localhost")
        self.port     = int(config_settings.get("QDRANT_PORT", 6333))
        self.base_url = f"http://{self.host}:{self.port}"
        self.logger   = logger

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        url  = f"{self.base_url}{path}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        url  = f"{self.base_url}{path}"
        resp = requests.post(url, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str, body: Optional[dict] = None) -> dict:
        url  = f"{self.base_url}{path}"
        resp = requests.delete(url, json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── Point factory — returns the correct entity for the collection ─────────

    @staticmethod
    def _make_point(collection: str, raw: dict) -> Union[ChunkPoint, MetadataPoint]:
        if collection == "zh_metadata":
            return MetadataPoint.from_qdrant_point(raw)
        return ChunkPoint.from_qdrant_point(raw)

    # ── Collections ───────────────────────────────────────────────────────────

    def list_collections(self) -> list[CollectionInfo]:
        try:
            data      = self._get("/collections")
            available = {c["name"] for c in data.get("result", {}).get("collections", [])}
            result    = []
            for name in KNOWN_COLLECTIONS:
                if name in available:
                    info = self._get(f"/collections/{name}")
                    r    = info.get("result", {})
                    result.append(CollectionInfo(
                        name=name,
                        points_count=r.get("points_count", 0),
                        indexed_vectors_count=r.get("indexed_vectors_count", 0),
                        status=r.get("status", "unknown"),
                    ))
                else:
                    result.append(CollectionInfo(
                        name=name, points_count=0,
                        indexed_vectors_count=0, status="not_found",
                    ))
            return result
        except Exception:
            self.logger.do_log(f"QdrantService.list_collections: {traceback.format_exc()}", "ERROR")
            raise

    def get_collection_info(self, collection: str) -> CollectionInfo:
        info = self._get(f"/collections/{collection}")
        r    = info.get("result", {})
        return CollectionInfo(
            name=collection,
            points_count=r.get("points_count", 0),
            indexed_vectors_count=r.get("indexed_vectors_count", 0),
            status=r.get("status", "unknown"),
        )

    # ── Scrolling / Browsing ──────────────────────────────────────────────────

    def scroll_chunks(
        self,
        collection: str,
        limit: int = 20,
        from_order_value: Optional[int] = None,
        order_direction: str = "desc",
        source_filter: Optional[str] = None,
        date_from: Optional[str] = None,   # "YYYY-MM-DD"
        date_to: Optional[str] = None,     # "YYYY-MM-DD"
    ) -> ScrollResult:
        """
        Paginated browse of a collection.

        zh_chunks  — order_by ingest_ts_epoch (epoch-ms cursor, +/-1 for exclusivity).
        zh_metadata — UUID-based offset; from_order_value is passed directly as offset.
        """
        import datetime

        body: dict = {
            "limit":        limit + 1,   # fetch one extra to detect whether more pages exist
            "with_payload": True,
            "with_vector":  False,
        }

        use_order_by = (collection == "zh_chunks")

        if use_order_by:
            order_clause: dict = {
                "key":       "ingest_ts_epoch",
                "direction": order_direction,
            }
            if from_order_value is not None:
                order_clause["start_from"] = from_order_value
            body["order_by"] = order_clause
        else:
            # Pass the UUID cursor received from the previous page response
            if from_order_value is not None:
                body["offset"] = from_order_value

        # Build filter clauses
        must_clauses = []

        if source_filter:
            must_clauses.append({
                "key":   "source_pdf",
                "match": {"text": source_filter},
            })

        if date_from or date_to:
            range_clause: dict = {}
            if date_from:
                dt = datetime.datetime.strptime(date_from, "%Y-%m-%d")
                range_clause["gte"] = int(dt.timestamp() * 1000)
            if date_to:
                dt = datetime.datetime.strptime(date_to, "%Y-%m-%d")
                dt = dt.replace(hour=23, minute=59, second=59)
                range_clause["lte"] = int(dt.timestamp() * 1000)
            must_clauses.append({"key": "ingest_ts_epoch", "range": range_clause})

        if must_clauses:
            body["filter"] = {"must": must_clauses}

        data       = self._post(f"/collections/{collection}/points/scroll", body)
        result     = data.get("result", {})
        raw_points = result.get("points", [])

        has_more = len(raw_points) > limit
        if has_more:
            raw_points = raw_points[:limit]

        points = [self._make_point(collection, p) for p in raw_points]

        # Determine next-page cursor depending on pagination strategy
        next_cursor = None
        if use_order_by and has_more and points:
            last_ts = points[-1].ingest_ts_epoch
            if last_ts is not None:
                # +/- 1 makes start_from exclusive so the last point is not repeated
                next_cursor = str(last_ts - 1) if order_direction == "desc" else str(last_ts + 1)
        elif not use_order_by and has_more:
            next_cursor = result.get("next_page_offset")

        return ScrollResult(
            points=points,
            next_page_offset=next_cursor,
            total_returned=len(points),
        )

    # ── Full collection scroll (for aggregations) ─────────────────────────────

    def scroll_all(self, collection: str, batch_size: int = 250) -> list[dict]:
        """
        Scrolls the entire collection using UUID-based offset (no order_by).
        Used for source/run summary aggregations.
        """
        all_payloads = []
        offset       = None

        while True:
            body: dict = {
                "limit":        batch_size,
                "with_payload": True,
                "with_vector":  False,
            }
            if offset:
                body["offset"] = offset

            data   = self._post(f"/collections/{collection}/points/scroll", body)
            result = data.get("result", {})
            points = result.get("points", [])

            for p in points:
                all_payloads.append(p.get("payload", {}))

            offset = result.get("next_page_offset")
            if not offset or not points:
                break

        return all_payloads

    # ── Point detail ──────────────────────────────────────────────────────────

    def get_point(self, collection: str, point_id: str) -> Optional[Union[ChunkPoint, MetadataPoint]]:
        try:
            data = self._get(f"/collections/{collection}/points/{point_id}")
            raw  = data.get("result")
            if not raw:
                return None
            return self._make_point(collection, raw)
        except Exception:
            return None

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_point(self, collection: str, point_id: str) -> bool:
        try:
            self._post(
                f"/collections/{collection}/points/delete",
                {"points": [point_id]}
            )
            return True
        except Exception:
            self.logger.do_log(f"QdrantService.delete_point: {traceback.format_exc()}", "ERROR")
            return False

    # ── Stats — full-collection aggregations ──────────────────────────────────

    def get_ingest_run_summary(self, collection: str) -> list[dict]:
        """Counts points per ingest_run_id across the entire collection."""
        try:
            all_payloads = self.scroll_all(collection)
            run_map: dict[str, dict] = {}
            for pl in all_payloads:
                run_id = pl.get("ingest_run_id", "unknown")
                ts     = pl.get("ingest_timestamp", "")
                if run_id not in run_map:
                    run_map[run_id] = {"run_id": run_id, "count": 0, "last_ts": ts}
                run_map[run_id]["count"] += 1
            return sorted(run_map.values(), key=lambda x: x["last_ts"], reverse=True)
        except Exception:
            self.logger.do_log(f"QdrantService.get_ingest_run_summary: {traceback.format_exc()}", "ERROR")
            raise

    def get_source_summary(self, collection: str) -> list[dict]:
        """Counts points per source identifier across the entire collection."""
        try:
            all_payloads = self.scroll_all(collection)
            source_map: dict[str, dict] = {}
            for pl in all_payloads:
                # zh_chunks uses source_pdf, zh_metadata uses filename
                src = pl.get("source_pdf") or pl.get("filename", "unknown")
                ts  = pl.get("ingest_timestamp", "")
                if src not in source_map:
                    source_map[src] = {"source_pdf": src, "count": 0, "last_ts": ts}
                source_map[src]["count"] += 1
            return sorted(source_map.values(), key=lambda x: x["count"], reverse=True)
        except Exception:
            self.logger.do_log(f"QdrantService.get_source_summary: {traceback.format_exc()}", "ERROR")
            raise