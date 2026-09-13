"""
Report Run DB Observer
======================
Se cuelga del mismo mecanismo de avisos que ya usa el MCP: cada vez que el
codigo de un reporte manda un aviso con su job_id, este observador lo mira y,
solo si es de los importantes, lo deja anotado en la base.

Por eso ningun reporte (ni el de 10-K ni ningun otro) tiene que sumar una sola
linea nueva: los avisos ya existen, esto solo los escucha.

Que se anota:
  - el aviso de cierre que ya manda cada reporte  -> la corrida queda terminada
  - los avisos de error                           -> se guarda el ultimo
  - cualquier otro aviso                          -> solo refresca la hora, y
                                                     como mucho una vez por minuto
"""

import json
import threading
from datetime import datetime

from framework.common.logger.message_type import MessageType


class ReportRunDBObserver:

    # Cada cuanto, como maximo, se toca la base solo para decir "sigo vivo".
    HEARTBEAT_SECONDS = 60

    def __init__(self, runs_mgr, logger=None):
        self._runs_mgr = runs_mgr
        self._logger = logger
        self._lock = threading.Lock()
        self._runs = {}          # job_id -> id de la corrida en la base
        self._last_touch = {}    # job_id -> ultima vez que se toco la base
        self._closed = set()     # job_id ya cerrados

    # -- alta y baja de corridas seguidas --------------------------------------

    def track(self, job_id, run_id):
        if job_id is None or not run_id:
            return
        key = str(job_id)
        with self._lock:
            self._runs[key] = run_id
            self._last_touch[key] = datetime.now()
            self._closed.discard(key)

    def forget(self, job_id):
        if job_id is None:
            return
        key = str(job_id)
        with self._lock:
            self._runs.pop(key, None)
            self._last_touch.pop(key, None)
            self._closed.discard(key)

    def is_closed(self, job_id) -> bool:
        if job_id is None:
            return False
        with self._lock:
            return str(job_id) in self._closed

    # -- lo que llama el logger ------------------------------------------------

    def on_log(self, msg, level, job_id):
        """
        Nunca puede romper nada: si la base no contesta, el reporte sigue igual.
        """
        try:
            key = str(job_id) if job_id is not None else None
            if key is None:
                return

            with self._lock:
                run_id = self._runs.get(key)
                already_closed = key in self._closed

            if not run_id or already_closed:
                return

            closing = self._read_closing_event(msg)

            if closing is not None:
                self._close(key, run_id, closing)
                return

            if level == MessageType.ERROR:
                self._runs_mgr.touch_report_run(run_id, last_error=self._shorten(msg))
                self._mark_touched(key)
                return

            self._heartbeat(key, run_id)

        except Exception:
            # La anotacion nunca puede voltear una corrida.
            pass

    # -- interno ---------------------------------------------------------------

    @staticmethod
    def _read_closing_event(msg):
        """
        Devuelve None si el aviso no es de cierre. Si lo es, devuelve el texto
        del error cuando el propio aviso dice que termino mal, o cadena vacia
        cuando termino bien.
        """
        if not isinstance(msg, str):
            return None

        text = msg.strip()
        if not text.startswith("{") or "completed" not in text:
            return None

        try:
            data = json.loads(text)
        except Exception:
            return None

        if not isinstance(data, dict) or data.get("event") != "completed":
            return None

        if str(data.get("status", "")).lower() == "error":
            return str(data.get("error") or data.get("result") or "completed with error")

        return ""

    def _close(self, key, run_id, closing):
        if closing:
            self._runs_mgr.close_report_run(run_id, status="error",
                                            last_error=self._shorten(closing))
        else:
            self._runs_mgr.close_report_run(run_id, status="finished")

        with self._lock:
            self._closed.add(key)
            self._last_touch[key] = datetime.now()

    def _heartbeat(self, key, run_id):
        with self._lock:
            last = self._last_touch.get(key)

        if last is not None and (datetime.now() - last).total_seconds() < self.HEARTBEAT_SECONDS:
            return

        self._runs_mgr.touch_report_run(run_id)
        self._mark_touched(key)

    def _mark_touched(self, key):
        with self._lock:
            self._last_touch[key] = datetime.now()

    @staticmethod
    def _shorten(text, limit=3900):
        text = str(text)
        return text if len(text) <= limit else text[:limit] + " ...[truncated]"
