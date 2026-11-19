
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent
SCREENSHOT_DIR = BASE_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

PORT = "8502"
URL = f"http://localhost:{PORT}"

app = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        PORT,
        "--server.headless",
        "true",
    ],
    cwd=BASE_DIR,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1
        )

        for attempt in range(30):
            try:
                page.goto(URL, wait_until="networkidle", timeout=5000)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("Streamlit application did not start.")

        page.wait_for_timeout(3000)

        # Main application
        page.screenshot(
            path=str(SCREENSHOT_DIR / "01_main_application.png"),
            full_page=True
        )

        # Spam email test
        page.get_by_label("Subject").fill(
            "Congratulations! You Have Won $5,000"
        )

        page.get_by_label("Message").fill(
            "Dear Customer, you have won a $5,000 cash prize. "
            "Claim your reward immediately. This offer expires soon. "
            "Click here to claim your reward."
        )

        page.get_by_role("button", name="Check Email").click()
        page.wait_for_timeout(2000)

        page.screenshot(
            path=str(SCREENSHOT_DIR / "02_spam_result.png"),
            full_page=True
        )

        # Legitimate email test
        page.get_by_label("Subject").fill(
            "Project Meeting Confirmation"
        )

        page.get_by_label("Message").fill(
            "Hello Ahmed, our project meeting is scheduled for "
            "Friday at 11 AM. We will review development progress "
            "and discuss the remaining tasks. Best regards, Muhammad."
        )

        page.get_by_role("button", name="Check Email").click()
        page.wait_for_timeout(2000)

        page.screenshot(
            path=str(SCREENSHOT_DIR / "03_legitimate_result.png"),
            full_page=True
        )

        browser.close()

    print("\nScreenshot capture completed.")
    print("Saved files:")

    for file in sorted(SCREENSHOT_DIR.glob("*.png")):
        print(file.name)

finally:
    app.terminate()
    app.wait(timeout=10)
