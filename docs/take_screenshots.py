#!/usr/bin/env python3
"""
take_screenshots.py — Automated screenshot capture for the docs.

Requires the app server running on localhost:8000:
    uvicorn app.main:app --port 8000

Usage:
    python3 docs/take_screenshots.py
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

BASE_URL = "http://localhost:8000"
OUT_DIR  = Path(__file__).parent / "screenshots"
OUT_DIR.mkdir(exist_ok=True)

VIEWPORT = {"width": 1400, "height": 860}


async def wait_heatmap(page, timeout_ms: int = 5000):
    """Wait until destination data appears in the sidebar (means fetch completed)."""
    try:
        await page.wait_for_selector(".dest-item", state="attached", timeout=timeout_ms)
    except Exception:
        pass
    await asyncio.sleep(0.8)  # extra pause for canvas render


async def take_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page(viewport=VIEWPORT)

        print(f"Opening {BASE_URL} …")
        await page.goto(BASE_URL, wait_until="networkidle")

        # Wait for the loading spinner to disappear (server may take ~30s)
        await page.wait_for_selector("#loading", state="hidden", timeout=120_000)
        await asyncio.sleep(1.2)

        # ── 1. Overview — full SEQ map, blue scale visible ─────────
        await page.screenshot(path=str(OUT_DIR / "screenshot_overview.png"))
        print("  ✓ screenshot_overview.png")

        # ── 2. Blue scale — Gold Coast zoomed in ───────────────────
        await page.evaluate("() => window._seqMap.setView([-28.00, 153.42], 12)")
        await asyncio.sleep(1.8)
        await page.screenshot(path=str(OUT_DIR / "screenshot_blue_scale.png"))
        print("  ✓ screenshot_blue_scale.png")

        # ── 3. Stop selected — Central Station + heatmap ───────────
        await page.evaluate("() => window._seqMap.setView([-27.48, 153.03], 13)")
        await asyncio.sleep(0.8)
        await page.evaluate("""() => {
            window._seqMap.eachLayer(function(layer) {
                if (layer.stopData && layer.stopData.stop_id === '600018') {
                    layer.fire('click');
                }
            });
        }""")
        await wait_heatmap(page, timeout_ms=6000)
        await page.screenshot(path=str(OUT_DIR / "screenshot_stop_selected.png"))
        print("  ✓ screenshot_stop_selected.png")

        # ── 4. Gravity model — same stop ───────────────────────────
        await page.evaluate("() => setModel('gravity')")
        await wait_heatmap(page, timeout_ms=6000)
        await page.screenshot(path=str(OUT_DIR / "screenshot_gravity.png"))
        print("  ✓ screenshot_gravity.png")

        # ── 5. Trend model — Central Station activity over time ────
        await page.evaluate("() => setModel('trend')")
        await asyncio.sleep(1.8)
        await page.screenshot(path=str(OUT_DIR / "screenshot_trend.png"))
        print("  ✓ screenshot_trend.png")

        await browser.close()

    print(f"\nAll screenshots saved to {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(take_screenshots())
