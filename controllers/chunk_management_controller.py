import traceback
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from controllers.base_controller import BaseController
from service_layer.client.qdrant.qdrant_service import QdrantService


class ChunkManagementController(BaseController):
    """
    Controller for the Chunk Management section under Data menu.
    Exposes UI page + JSON API for browsing, inspecting, and deleting Qdrant points.
    Works across all known collections (zh_chunks, zh_metadata).
    """

    def __init__(self, config_settings: dict, logger):
        super().__init__()
        self.config  = config_settings
        self.logger  = logger
        self.qdrant  = QdrantService(config_settings, logger)

        self.router    = APIRouter()
        self.templates = Jinja2Templates(
            directory=str(Path(__file__).parent.parent / "templates")
        )

        # ── Page ──────────────────────────────────────────────────────────────
        self.router.get("/", response_class=HTMLResponse)(self.display_page)

        # ── Collection info ───────────────────────────────────────────────────
        self.router.get("/collections",     response_class=JSONResponse)(self.api_get_collections)
        self.router.get("/collection_info", response_class=JSONResponse)(self.api_get_collection_info)

        # ── Chunk browsing ────────────────────────────────────────────────────
        self.router.get("/chunks",          response_class=JSONResponse)(self.api_scroll_chunks)

        # ── Point detail ──────────────────────────────────────────────────────
        self.router.get("/chunk_detail",    response_class=JSONResponse)(self.api_get_chunk_detail)

        # ── Delete ────────────────────────────────────────────────────────────
        self.router.post("/delete_chunk",   response_class=JSONResponse)(self.api_delete_chunk)

        # ── Stats ─────────────────────────────────────────────────────────────
        self.router.get("/ingest_runs",     response_class=JSONResponse)(self.api_ingest_runs)
        self.router.get("/source_summary",  response_class=JSONResponse)(self.api_source_summary)

    # ── Page ──────────────────────────────────────────────────────────────────

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse(
            "chunk_management.html", {"request": request}
        )

    # ── Collections ───────────────────────────────────────────────────────────

    async def api_get_collections(self, request: Request):
        try:
            cols = self.qdrant.list_collections()
            return JSONResponse([{
                "name":                  c.name,
                "points_count":          c.points_count,
                "indexed_vectors_count": c.indexed_vectors_count,
                "status":                c.status,
            } for c in cols])
        except Exception as e:
            self.logger.do_log(f"api_get_collections: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_get_collection_info(self, request: Request, collection: str):
        try:
            c = self.qdrant.get_collection_info(collection)
            return JSONResponse({
                "name":                  c.name,
                "points_count":          c.points_count,
                "indexed_vectors_count": c.indexed_vectors_count,
                "status":                c.status,
            })
        except Exception as e:
            self.logger.do_log(f"api_get_collection_info: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Chunk browsing ────────────────────────────────────────────────────────

    async def api_scroll_chunks(
        self,
        request: Request,
        collection: str       = "zh_chunks",
        limit: int            = 20,
        from_order_value: int = None,
        source_filter: str    = None,
        date_from: str        = None,   # YYYY-MM-DD
        date_to: str          = None,   # YYYY-MM-DD
    ):
        try:
            result = self.qdrant.scroll_chunks(
                collection=collection,
                limit=limit,
                from_order_value=from_order_value,
                source_filter=source_filter or None,
                date_from=date_from or None,
                date_to=date_to or None,
            )
            return JSONResponse({
                "points":           [p.to_dict() for p in result.points],
                "next_page_offset": result.next_page_offset,
                "total_returned":   result.total_returned,
            })
        except Exception as e:
            self.logger.do_log(f"api_scroll_chunks: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Point detail ──────────────────────────────────────────────────────────

    async def api_get_chunk_detail(self, request: Request, collection: str, point_id: str):
        try:
            point = self.qdrant.get_point(collection, point_id)
            if not point:
                return JSONResponse({"ok": False, "error": "Point not found"}, status_code=404)
            return JSONResponse({"ok": True, "point": point.to_dict()})
        except Exception as e:
            self.logger.do_log(f"api_get_chunk_detail: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Delete ────────────────────────────────────────────────────────────────

    async def api_delete_chunk(self, request: Request):
        try:
            body       = await request.json()
            collection = body.get("collection")
            point_id   = body.get("point_id")
            if not collection or not point_id:
                return JSONResponse({"ok": False, "error": "collection and point_id required"}, status_code=400)
            ok = self.qdrant.delete_point(collection, point_id)
            return JSONResponse({"ok": ok})
        except Exception as e:
            self.logger.do_log(f"api_delete_chunk: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def api_ingest_runs(self, request: Request, collection: str = "zh_chunks"):
        try:
            runs = self.qdrant.get_ingest_run_summary(collection)
            return JSONResponse(runs)
        except Exception as e:
            self.logger.do_log(f"api_ingest_runs: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def api_source_summary(self, request: Request, collection: str = "zh_chunks"):
        try:
            sources = self.qdrant.get_source_summary(collection)
            return JSONResponse(sources)
        except Exception as e:
            self.logger.do_log(f"api_source_summary: {traceback.format_exc()}", "ERROR")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)