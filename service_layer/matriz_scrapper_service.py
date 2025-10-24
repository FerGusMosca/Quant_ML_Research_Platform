import time
import json
import websocket
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

class MatrizScraperService:
    """
    Handles login and data streaming from Matriz (Nasini DMA).
    - Uses Selenium to authenticate (user/password)
    - Reuses session cookies to connect to the WebSocket
    - Subscribes to market topics (e.g. Letras)
    """

    LOGIN_URL = "https://matriz.nasini.xoms.com.ar/"
    WS_BASE = "wss://matriz.nasini.xoms.com.ar/ws?session_id="

    def __init__(self, username: str, password: str, headless: bool = True):
        self.username = username
        self.password = password
        self.headless = headless
        self.session_id = None
        self.cookies = None
        self.ws_app = None
        self.messages = []
        self.connected = False

    # ------------------------------------------------------------
    # 1️⃣ LOGIN
    # ------------------------------------------------------------
    def login(self):
        print("🌐 Logging into Matriz...")
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=opts)
        driver.get(self.LOGIN_URL)
        time.sleep(3)

        try:
            driver.find_element(By.NAME, "user").send_keys(self.username)
            driver.find_element(By.NAME, "password").send_keys(self.password)
            driver.find_element(By.XPATH, "//button[contains(.,'Ingresar')]").click()
        except Exception:
            print("⚠️ Could not locate login elements.")
            driver.quit()
            return None

        print("⏳ Waiting for dashboard to load...")
        time.sleep(8)

        cookies = driver.get_cookies()
        session_cookie = next((c["value"] for c in cookies if "session_id" in c["name"].lower()), None)

        if not session_cookie:
            print("❌ Could not find session_id cookie.")
            driver.quit()
            return None

        self.session_id = session_cookie
        self.cookies = cookies
        driver.quit()
        print(f"✅ Logged in successfully. session_id={self.session_id}")
        return self.session_id

    # ------------------------------------------------------------
    # 2️⃣ CONNECT TO WEBSOCKET
    # ------------------------------------------------------------
    def connect_websocket(self, subscribe_payload: dict, listen_seconds: int = 10):
        if not self.session_id:
            raise ValueError("You must call login() first to get session_id.")

        ws_url = self.WS_BASE + self.session_id
        print(f"🔌 Connecting to WebSocket: {ws_url}")

        def on_open(ws):
            print("✅ WS Open - sending subscribe payload...")
            ws.send(json.dumps(subscribe_payload))

        def on_message(ws, msg):
            if msg.startswith("M:"):
                print(f"📩 MarketData: {msg[:100]}...")
            self.messages.append(msg)

        def on_error(ws, err):
            print("❌ WS Error:", err)

        def on_close(ws, code, reason):
            print(f"🔌 WS Closed (code={code}, reason={reason})")
            self.connected = False

        headers = [f"Cookie: session_id={self.session_id}"]
        self.ws_app = websocket.WebSocketApp(
            ws_url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        thread = threading.Thread(target=self.ws_app.run_forever, kwargs={'ping_interval': 20})
        thread.daemon = True
        thread.start()
        self.connected = True

        print(f"⏳ Listening for {listen_seconds} seconds...")
        time.sleep(listen_seconds)
        self.ws_app.close()
        return self.messages

    # ------------------------------------------------------------
    # 3️⃣ PARSE RAW STREAM MESSAGES
    # ------------------------------------------------------------
    @staticmethod
    def parse_market_data(messages):
        rows = []
        for msg in messages:
            if not msg.startswith("M:"):
                continue
            parts = msg.split("|")
            try:
                symbol = parts[0].split("_")[-2] if "_" in parts[0] else parts[0]
                rows.append({
                    "symbol": symbol,
                    "last_price": parts[6] if len(parts) > 6 else None,
                    "bid": parts[3] if len(parts) > 3 else None,
                    "ask": parts[4] if len(parts) > 4 else None,
                    "timestamp": parts[7] if len(parts) > 7 else None,
                    "raw": msg
                })
            except Exception:
                continue
        return rows
