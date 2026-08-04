import csv
import threading
import traceback
from datetime import datetime

from common.util.downloaders.SEC_securities_metadata_downloader import SECSecuritiesMetadataDownloader
from data_access_layer.sec_securities_metadata_manager import SECSecuritiesMetadataManager
from framework.common.logger.message_type import MessageType


# Shared run state. The screen reads it to paint the progress bar.
RUN_STATE = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "total": 0,
    "done": 0,
    "ok": 0,
    "failed": 0,
    "current": None,
    "cancel": False,
}
_STATE_LOCK = threading.Lock()


def set_run_state(**kwargs):
    with _STATE_LOCK:
        RUN_STATE.update(kwargs)


def get_run_state() -> dict:
    with _STATE_LOCK:
        return dict(RUN_STATE)


def request_cancel():
    set_run_state(cancel=True)


class SECMetadataOrchestationLogic:
    """
    Fills in sic / sicDescription / exchange / entityType on SEC_Securities.

    Why it is needed: process_download_sec_securities reads company_tickers.json,
    which only carries cik_str, ticker and title. The item.get("sic"),
    item.get("exchange") and item.get("entityType") calls there always return
    None, which is why those four columns stayed NULL. This job hits the
    submissions endpoint, which does carry them.

    How it works (loop t -->):
        t=0  the DB hands you the list of securities in PENDING
        t=1  you wait out whatever is left so you never exceed 7 requests/second
        t=2  you hit data.sec.gov/submissions/CIK0000320193.json
             if the server says "slow down" (429/503) you wait twice as long
             and go back to t=2
        t=3  you pull sic, sicDescription, exchanges[], entityType
        t=4  you look up the sector from the sic: 4 digits first, then 3, then 2
        t=5  UPDATE by cik, mark it OK, log [SEC-META] OK
             if it failed, mark it ERROR, log [SEC-META] FAIL, and move on
        t=6  back to t=1 with the next one

    Resumable: state lives in the meta_status column. If it dies, the next run
    asks for PENDING again and picks up where it left off.
    """

    OK_MARK = "\u2714"
    FAIL_MARK = "\u274c"

    def __init__(self, ml_reports_conn_str, logger, user_agent,
                 requests_per_second=7.0):
        self.logger = logger
        self.metadata_mgr = SECSecuritiesMetadataManager(ml_reports_conn_str, logger)
        self.downloader = SECSecuritiesMetadataDownloader(
            user_agent=user_agent, requests_per_second=requests_per_second)

    # ── Unit of work ──────────────────────────────────────────────────────────

    def __process_one__(self, security: dict) -> bool:
        cik = security.get("cik")
        label = security.get("symbol") or security.get("ticker") or f"CIK{cik}"

        if cik is None:
            self.logger.do_log(f"[SEC-META] {self.FAIL_MARK} {label} - no CIK",
                               MessageType.ERROR)
            return False

        try:
            metadata_dto = self.downloader.download_metadata(int(cik), label)
            self.metadata_mgr.persist_metadata(metadata_dto)
            self.logger.do_log(f"[SEC-META] {self.OK_MARK} {label} - {str(metadata_dto)}",
                               MessageType.INFO)
            return True

        except FileNotFoundError as e:
            self.metadata_mgr.mark_failed(int(cik), "NOT_FOUND", str(e))
            self.logger.do_log(f"[SEC-META] {self.FAIL_MARK} {label} - {str(e)}",
                               MessageType.WARNING)
            return False

        except Exception as e:
            self.metadata_mgr.mark_failed(int(cik), "ERROR", f"{type(e).__name__}: {str(e)}")
            self.logger.do_log(f"[SEC-META] {self.FAIL_MARK} {label} - {str(e)}",
                               MessageType.ERROR)
            return False

    # ── Runs ──────────────────────────────────────────────────────────────────

    def process_download_all_metadata(self, top=None, include_errors=False):
        pending = self.metadata_mgr.get_pending_securities(top=top,
                                                           include_errors=include_errors)

        set_run_state(running=True, cancel=False, started_at=datetime.now().isoformat(),
                      finished_at=None, total=len(pending), done=0, ok=0, failed=0,
                      current=None)

        self.logger.do_log(f"[SEC-META] run started - {len(pending)} pending securities",
                           MessageType.INFO)

        ok_qty, failed_qty = 0, 0

        try:
            for idx, security in enumerate(pending, start=1):
                if get_run_state().get("cancel"):
                    self.logger.do_log("[SEC-META] cancelled by the user",
                                       MessageType.WARNING)
                    break

                label = security.get("symbol") or security.get("ticker")
                set_run_state(current=label)

                if self.__process_one__(security):
                    ok_qty += 1
                else:
                    failed_qty += 1

                set_run_state(done=idx, ok=ok_qty, failed=failed_qty)

                if idx % 100 == 0:
                    self.logger.do_log(
                        f"[SEC-META] progress {idx}/{len(pending)} - ok={ok_qty} fail={failed_qty}",
                        MessageType.INFO)
        finally:
            set_run_state(running=False, current=None,
                          finished_at=datetime.now().isoformat())

        self.logger.do_log(
            f"[SEC-META] done - ok={ok_qty} fail={failed_qty} of {len(pending)}",
            MessageType.INFO)

        return {"total": len(pending), "ok": ok_qty, "failed": failed_qty}

    def process_download_single_metadata(self, symbol=None, cik=None):
        security = self.metadata_mgr.get_security_by_key(symbol=symbol, cik=cik)

        if not security:
            self.logger.do_log(
                f"[SEC-META] {self.FAIL_MARK} not found in SEC_Securities: {symbol or cik}",
                MessageType.ERROR)
            return {"ok": False, "reason": "not_in_db", "symbol": symbol, "cik": cik}

        ok = self.__process_one__(security)
        return {"ok": ok, "symbol": security.get("symbol"), "cik": security.get("cik")}

    # ── Tags ──────────────────────────────────────────────────────────────────

    @staticmethod
    def read_symbols_from_csv(file_path_or_bytes):
        """
        Accepts either a path or the raw file bytes. Picks the symbol / ticker /
        asset / security column, and falls back to the first column when there is
        no header.
        """
        if isinstance(file_path_or_bytes, (bytes, bytearray)):
            text = bytes(file_path_or_bytes).decode("utf-8-sig", errors="replace")
            lines = text.splitlines()
        else:
            with open(file_path_or_bytes, "r", encoding="utf-8-sig", newline="") as fh:
                lines = fh.read().splitlines()

        rows = [r for r in csv.reader(lines) if r and any(c.strip() for c in r)]
        if not rows:
            return []

        # The header is detected by column name. With a single column csv.Sniffer
        # misses it and "SYMBOL" would slip through as if it were a ticker.
        known = ("symbol", "ticker", "asset", "security")
        header = [c.strip().lower() for c in rows[0]]
        col, start = 0, 0
        if any(h in known for h in header):
            for candidate in known:
                if candidate in header:
                    col, start = header.index(candidate), 1
                    break

        symbols, seen = [], set()
        for row in rows[start:]:
            if col < len(row):
                value = row[col].strip().upper()
                if value and value not in seen:
                    seen.add(value)
                    symbols.append(value)
        return symbols

    def process_tag_securities(self, tag_code, symbols, tag_name=None, tag_group="CUSTOM"):
        self.metadata_mgr.persist_tag(tag_code, tag_name, tag_group)
        return self.metadata_mgr.apply_tag_to_symbols(tag_code, symbols)

    def process_tag_securities_from_csv(self, tag_code, csv_source,
                                        tag_name=None, tag_group="CUSTOM"):
        symbols = self.read_symbols_from_csv(csv_source)
        if not symbols:
            raise Exception("The CSV has no symbols")
        return self.process_tag_securities(tag_code, symbols, tag_name, tag_group)

    def process_tag_securities_by_sector(self, tag_code, sector_code):
        return self.metadata_mgr.apply_tag_by_sector(tag_code, sector_code)


def run_all_in_background(ml_reports_conn_str, logger, user_agent,
                          top=None, include_errors=False, requests_per_second=7.0):
    """
    Launches the sweep on a thread so the screen does not hang.
    Returns False when a run is already in progress.
    """
    if get_run_state().get("running"):
        return False

    def _target():
        try:
            orchestation = SECMetadataOrchestationLogic(
                ml_reports_conn_str, logger, user_agent, requests_per_second)
            orchestation.process_download_all_metadata(top=top, include_errors=include_errors)
        except Exception as e:
            print(traceback.format_exc())
            set_run_state(running=False, finished_at=datetime.now().isoformat())
            logger.do_log(f"[SEC-META] critical error during the run: {str(e)}",
                          MessageType.ERROR)

    threading.Thread(target=_target, name="sec-meta-runner", daemon=True).start()
    return True
