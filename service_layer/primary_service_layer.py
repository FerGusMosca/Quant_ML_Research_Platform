import requests
import os
import json
import time
from datetime import datetime
from urllib.parse import quote
from common.dto.byma_rate_dto import BYMARateDTO
from common.util.classifiers.argy_instrument_classifiers import ArgyInstrumentClassifier
from urllib.parse import quote

class PrimaryServiceLayer:
    BASE_URL = "https://api.nasini.xoms.com.ar/"
    CACHE_DIR = "./cache/primary"
    LOGIN_USER = "fmosca"
    LOGIN_PASSWORD = "<mypwd>>"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.token = self._authenticate()
        self.session.headers["X-Auth-Token"] = self.token

    # ---------------- AUTH ----------------
    def _authenticate(self):
        url = f"{self.BASE_URL}auth/getToken"
        headers = {"X-Username": self.LOGIN_USER, "X-Password": self.LOGIN_PASSWORD}
        print(f"🔐 Authenticating to Primary API ({self.LOGIN_USER}) ...")

        resp = self.session.post(url, headers=headers)
        if resp.status_code == 200:
            token = resp.headers.get("X-Auth-Token")
            if not token:
                raise Exception("❌ No X-Auth-Token received from Primary API.")
            print("✅ Authenticated successfully.")
            return token
        else:
            raise Exception(f"❌ Authentication failed: {resp.status_code} {resp.text}")

    # ---------------- HELPERS ----------------
    def _get_json(self, endpoint: str):
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"⚠️ HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"❌ Network error: {e}")
        return None

    def _days_to_maturity(self, maturity_date: str):
        try:
            maturity = datetime.strptime(maturity_date[:8], "%Y%m%d").date()
            return (maturity - datetime.today().date()).days
        except Exception:
            return None

    def _probe_marketdata(self, symbol):
        """Tries multiple marketdata endpoints for a given symbol."""
        endpoints = [
            f"rest/marketdata/get?symbol={symbol}&entries=LA",
            f"rest/marketdata/byInstrument?symbol={symbol}",
            f"rest/marketdata/lastPrice?symbol={symbol}",
            f"rest/marketdata/all?symbol={symbol}",
        ]
        for ep in endpoints:
            data = self._get_json(ep)
            if data:
                print(f"✅ {ep} → {str(data)[:180]}")
            else:
                print(f"⚠️ {ep} → no data")

    # ---------------- MAIN ----------------
    def inspect_instrument_structure(self):
        print("📥 Fetching full instruments list to inspect fields...")
        data = self._get_json("rest/instruments/details")
        if not data or "instruments" not in data:
            print("⚠️ Invalid response from Primary.")
            return

        sample = data["instruments"][0]
        print("\n🧩 Keys available in the first instrument:")
        for k, v in sample.items():
            print(f"  {k:<25} → {str(v)[:100]}")

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        with open(os.path.join(self.CACHE_DIR, "instrument_sample.json"), "w") as f:
            json.dump(sample, f, indent=2)
        print("\n✅ Saved sample to cache/instrument_sample.json")

    def get_instrument_types_snapshot(self):
        print("📥 Fetching all instruments to list unique types...")
        data = self._get_json("rest/instruments/details")
        if not data or "instruments" not in data:
            print("⚠️ Invalid response from Primary.")
            return

        type_stats = {}
        for inst in data["instruments"]:
            symbol = inst.get("instrumentId", {}).get("symbol", "?")
            inst_type = inst.get("instrumentType", "UNKNOWN")
            print(f"  {symbol:<15} | {inst_type}")
            type_stats[inst_type] = type_stats.get(inst_type, 0) + 1

        print("\n📊 Instrument types summary:")
        for t, c in sorted(type_stats.items(), key=lambda x: -x[1]):
            print(f"  {t:<25} → {c}")

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        with open(os.path.join(self.CACHE_DIR, "instrument_types.json"), "w") as f:
            json.dump(type_stats, f, indent=2)
        print("\n✅ Saved to cache/instrument_types.json")

    def get_interest_rates_snapshot(self, limit=None):
        #self.get_instrument_types_snapshot()
        #self.inspect_instrument_structure()
        print("📥 Fetching instruments list from Primary ...")
        data = self._get_json("rest/instruments/details")
        if not data or "instruments" not in data:
            print("⚠️ Invalid or empty instrument structure.")
            return []

        instruments = data["instruments"]
        print(f"✅ Retrieved {len(instruments)} instruments.")

        keywords = ["AL", "AE", "GD", "T2", "T3", "T4", "TV", "DICP", "PBA", "BOPREAL"]
        fixed_income = []
        for inst in instruments:
            try:
                desc = inst.get("securityDescription", "").upper()
                symbol = inst.get("instrumentId", {}).get("symbol", "")
                curr = inst.get("currency")

                # Skip options, futures, derivatives (.OC, .DI, .FE, .O, .P)
                if any(x in desc for x in [".OC", ".DI", ".FE", ".O ", ".P "]) or \
                        symbol.endswith(("OC", "DI", "FE", "O", "P")):
                    continue

                if (
                        desc.startswith("MERV - XMEV -")
                        and any(k in desc for k in keywords)
                        and inst.get("maturityDate")
                        and curr in ("ARS", "USD")
                        and len(desc) < 35
                ):
                    fixed_income.append(inst)
            except Exception:
                continue

        print(f"✅ Filtered {len(fixed_income)} probable fixed-income instruments.\n")

        results = []
        total = len(fixed_income)
        selected = fixed_income if limit is None else fixed_income[:limit]

        for idx, inst in enumerate(selected, 1):
            try:
                instrument_id = inst.get("instrumentId") or {}
                symbol = instrument_id.get("symbol")
                market_id = instrument_id.get("marketId")
                maturity = inst.get("maturityDate")
                desc = inst.get("securityDescription", "")
                curr = inst.get("currency")

                # --- 🔎 EXTENDED DEBUG LOG ---
                if symbol and symbol.startswith(("AL35", "GD38", "GD46", "GD41")):
                    print(f"\n🔎 DEBUG: Instrument metadata for {symbol}")
                    for k, v in inst.items():
                        print(f"   {k}: {v}")
                    print("--------------------------------------------------")

                if not symbol or not maturity or not market_id:
                    continue

                # Classification (uses ArgyInstrumentClassifier)
                classification = ArgyInstrumentClassifier.classify(desc)
                category = classification["category"]
                method = classification["method"]

                # --- Market data query ---
                endpoint = f"rest/marketdata/get?marketId={market_id}&symbol={quote(symbol, safe='')}&entries=LA"
                md = self._get_json(endpoint)

                price = None
                if md and isinstance(md.get("marketData"), dict):
                    la = md["marketData"].get("LA")
                    if la and isinstance(la, dict):
                        price = la.get("price")

                if not price:
                    continue

                days = self._days_to_maturity(maturity)
                if not days or days <= 0:
                    continue

                tna = ((100 / float(price)) - 1) * (360 / days) * 100

                rate = BYMARateDTO(
                    name=f"{desc.strip()} ({curr or 'ARS'})",
                    date=datetime.today().date(),
                    value=round(tna, 2)
                )
                results.append(rate)

                print(f"✅ {desc[:38]:<40} | {category:<32} | {method:<25} → {tna:>7.2f}%")

                if idx % 25 == 0:
                    print(f"🧩 Progress: {idx}/{total} processed...")

                time.sleep(0.05)

            except Exception as e:
                sym = inst.get("instrumentId", {}).get("symbol", "?")
                print(f"⚠️ Error processing {sym}: {e}")

        print(f"\n✅ Completed snapshot. {len(results)} yields calculated from {total} instruments.\n")
        return results





