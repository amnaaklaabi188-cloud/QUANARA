"""Write the QuanAra brand assets (sidebar logo, collapsed-sidebar mark, favicon) from quanara/ui/brand.py.

Run:  python scripts/build_brand_assets.py
The favicon PNG is rasterised with a headless browser (Playwright, system Chrome); the SVGs are written directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quanara.ui import brand  # noqa: E402


def main() -> None:
    brand.ASSETS.mkdir(parents=True, exist_ok=True)
    brand.LOGO.write_text(brand.logo_svg())
    brand.MARK.write_text(brand.mark_svg(48, uid="mk", on_dark=True, title=True))
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 128, "height": 128}, device_scale_factor=1)
        pg.set_content(f'<html><body style="margin:0;background:transparent">{brand.favicon_svg(128)}</body></html>')
        pg.locator("svg").screenshot(path=str(brand.FAVICON), omit_background=True)
        b.close()
    for f in (brand.LOGO, brand.MARK, brand.FAVICON):
        print("wrote", f.relative_to(brand.ASSETS.parent.parent.parent), f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
