import os
import random
import time
import requests

from framework.common.logger.message_type import MessageType


class Q10Downloader:
    """
    Utility class to handle 10-Q downloads from SEC EDGAR
    """

    @staticmethod
    def download_q10s(symbol, cik, year, output_dir, logger, job_id=None):
        headers = {
            "User-Agent": "Q10Downloader/1.0 (fer.mosca@example.com)",
            "Accept-Encoding": "gzip, deflate",
        }

        if not cik:
            logger.do_log(
                f"[Q10] Missing CIK | {symbol}",
                MessageType.ERROR,
                job_id
            )
            raise ValueError(f"[Q10Downloader] Missing CIK for {symbol}")

        os.makedirs(output_dir, exist_ok=True)

        # Determine missing quarters
        missing_qs = []
        for q in range(1, 5):
            file_path = os.path.join(output_dir, f"{symbol}_{year}_Q{q}_10-Q.html")
            if not os.path.exists(file_path):
                missing_qs.append(q)

        if not missing_qs:
            logger.do_log(
                f"[Q10] All quarters already exist | {symbol} {year}",
                MessageType.INFO,
                job_id
            )
            return "EXISTS"

        # Fetch SEC submissions
        url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
        try:
            r = requests.get(url, headers=headers, timeout=10)

            if r.status_code == 404:
                logger.do_log(
                    f"[Q10] CIK not found (404) | {symbol} CIK={cik}",
                    MessageType.WARNING,
                    job_id
                )
                return "NOT_FOUND"

            r.raise_for_status()
            data = r.json()

        except requests.exceptions.Timeout as e:
            logger.do_log(
                f"[Q10] Timeout fetching submissions | {symbol} CIK={cik} | {e}",
                MessageType.ERROR,
                job_id
            )
            return "NOT_FOUND"

        except requests.exceptions.RequestException as e:
            logger.do_log(
                f"[Q10] Request error fetching submissions | {symbol} CIK={cik} | {e}",
                MessageType.ERROR,
                job_id
            )
            return "NOT_FOUND"

        filings = data.get("filings", {}).get("recent", {})

        # Map filings by quarter
        filings_by_q = {}
        for acc, fdate, form, doc in zip(
                filings.get("accessionNumber", []),
                filings.get("filingDate", []),
                filings.get("form", []),
                filings.get("primaryDocument", []),
        ):
            if form != "10-Q" or not fdate.startswith(str(year)):
                continue

            month = int(fdate.split("-")[1])
            quarter = (month - 1) // 3 + 1
            filings_by_q[quarter] = (acc, doc)

        downloaded = []
        found_any = False

        for q in missing_qs:
            filing = filings_by_q.get(q)
            if not filing:
                logger.do_log(
                    f"[Q10] No filing for Q{q} | {symbol} {year}",
                    MessageType.WARNING,
                    job_id
                )
                continue

            found_any = True
            acc, doc = filing
            acc_nodash = acc.replace("-", "")
            target_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"

            file_path = os.path.join(output_dir, f"{symbol}_{year}_Q{q}_10-Q.html")

            try:
                resp = requests.get(target_url, headers=headers, timeout=15)

                if resp.status_code == 404:
                    logger.do_log(
                        f"[Q10] Filing URL 404 | {symbol} {year} Q{q} | {target_url}",
                        MessageType.WARNING,
                        job_id
                    )
                    continue

                resp.raise_for_status()
                time.sleep(0.5 + random.random())

                with open(file_path, "wb") as f:
                    f.write(resp.content)

                downloaded.append(file_path)

                logger.do_log(
                    f"[Q10] Downloaded | {symbol} {year} Q{q}",
                    MessageType.INFO,
                    job_id
                )

            except requests.exceptions.Timeout as e:
                logger.do_log(
                    f"[Q10] Timeout downloading Q{q} | {symbol} {year} | {e}",
                    MessageType.ERROR,
                    job_id
                )

            except requests.exceptions.RequestException as e:
                logger.do_log(
                    f"[Q10] Error downloading Q{q} | {symbol} {year} | {e}",
                    MessageType.ERROR,
                    job_id
                )

            except Exception as e:
                logger.do_log(
                    f"[Q10] File write error Q{q} | {symbol} {year} | {e}",
                    MessageType.ERROR,
                    job_id
                )

        if downloaded:
            return "FOUND"

        if found_any:
            logger.do_log(
                f"[Q10] Filings exist but already present | {symbol} {year}",
                MessageType.INFO,
                job_id
            )
            return "EXISTS"

        logger.do_log(
            f"[Q10] No 10-Q filings found | {symbol} {year}",
            MessageType.WARNING,
            job_id
        )
        return "NOT_FOUND"



