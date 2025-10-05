from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

TEST_URL = "https://www.marketwatch.com/story/even-smaller-tech-stocks-are-getting-expensive-but-these-sectors-could-be-your-next-big-win-544e5289?mod=mw_FV"

def test_browser_basic():
    print("[TEST] Starting Chrome test...")

    options = Options()
    # 🧠 Usar Chrome visible para descartar headless crash
    # options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1200,800")

    service = Service(ChromeDriverManager().install())

    try:
        driver = webdriver.Chrome(service=service, options=options)
        print("[TEST] ✅ Chrome launched successfully.")
        driver.get(TEST_URL)
        print("[TEST] Page loaded, current title:")
        print(driver.title)
        time.sleep(5)
        driver.quit()
        print("[TEST] ✅ Browser closed correctly.")
    except Exception as e:
        print(f"[TEST] ❌ Browser failed to start -> {e}")

if __name__ == "__main__":
    test_browser_basic()
