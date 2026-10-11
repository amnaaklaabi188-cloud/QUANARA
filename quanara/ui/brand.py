"""QuanAra brand identity — name, tagline, official description, the geometric mark and the sidebar globe.

The mark is an open coordinate ring (a destination fix) holding three connected data points; a
predictive trajectory leaves the last point through the ring's opening and ends in a forecast node.
Read as a "Q", it combines quantitative data, connected destinations and a forward projection.
Palette: black-green #04100b, emerald #059669 / #10b981, mint #34d399 / #6ee7b7, deep green #047857.

`scripts/build_brand_assets.py` writes the sidebar logo, the collapsed-sidebar mark and the favicon
from the functions below, so every brand surface is generated from one definition.
"""
from __future__ import annotations

import random
from pathlib import Path

NAME = "QuanAra"
TAGLINE = "Quantitative AI for Aviation & Tourism"
COMPETITION = "DCT Abu Dhabi ChallengeON 2026"
DESCRIPTION = ("QuanAra is a research-driven predictive intelligence platform that connects aviation dynamics with hospitality demand through "
               "rigorous machine learning benchmarking, temporal validation, and model-based forecasting. Beyond predictive accuracy, the platform "
               "investigates model sensitivity, scenario-dependent responses, and predictive limitations, translating the complexity of real-world "
               "forecasting into transparent, measurable, and actionable intelligence for destination planning.")
BLACK, GREEN, BRIGHT, MINT, MINT_L, DEEP = "#04100b", "#059669", "#10b981", "#34d399", "#6ee7b7", "#047857"
ASSETS = Path(__file__).parent / "assets"
LOGO, MARK, FAVICON = ASSETS / "logo.svg", ASSETS / "mark.svg", ASSETS / "favicon.png"

# geometry on a 48 × 48 grid: ring centre (22, 22), radius 15, opening between 20° and 70° (lower right)
_RING = "M27.13 36.10 A15 15 0 1 1 36.10 27.13"
_LINKS = ((14.5, 26.0, 20.0, 15.5), (20.0, 15.5, 27.5, 22.5), (14.5, 26.0, 27.5, 22.5))
_TAIL = "M27.5 22.5 C31 26 34.5 31.5 40.5 40.5"


def mark(uid: str = "q", on_dark: bool = False, animated: bool = False) -> str:
    """The symbol as inline SVG paths (48 × 48 user units). `uid` keeps gradient ids unique on a page."""
    ring0, ring1 = (MINT_L, BRIGHT) if on_dark else (MINT, DEEP)
    tail0, tail1 = (BRIGHT, MINT) if on_dark else (GREEN, DEEP)
    link, node_stroke, hub, end = (MINT, BRIGHT, MINT_L, MINT) if on_dark else ("#86d9b6", GREEN, MINT, DEEP)
    anim_tail = ' class="qm-tail"' if animated else ""
    anim_end = ' class="qm-end"' if animated else ""
    return (f'<defs><linearGradient id="{uid}-r" x1="6" y1="6" x2="38" y2="38" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{ring0}"/><stop offset="1" stop-color="{ring1}"/></linearGradient>'
            f'<linearGradient id="{uid}-t" x1="27" y1="22" x2="41" y2="41" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{tail0}"/><stop offset="1" stop-color="{tail1}"/></linearGradient></defs>'
            f'<path d="{_RING}" fill="none" stroke="url(#{uid}-r)" stroke-width="3.4" stroke-linecap="round"/>'
            + "".join(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}" stroke="{link}" stroke-width="1.5" stroke-linecap="round"/>' for a, b, c, d in _LINKS)
            + f'<path d="{_TAIL}" fill="none" stroke="url(#{uid}-t)" stroke-width="2.6" stroke-linecap="round" pathLength="100"{anim_tail}/>'
            f'<circle cx="14.5" cy="26" r="2.6" fill="#ffffff" stroke="{node_stroke}" stroke-width="1.6"/>'
            f'<circle cx="20" cy="15.5" r="2.6" fill="#ffffff" stroke="{node_stroke}" stroke-width="1.6"/>'
            f'<circle cx="27.5" cy="22.5" r="2.9" fill="{hub}" stroke="{node_stroke}" stroke-width="1.6"/>'
            f'<circle cx="40.5" cy="40.5" r="3.4" fill="{end}"{anim_end}/>')


def mark_svg(size: int = 48, uid: str = "q", on_dark: bool = False, animated: bool = False, title: bool = False) -> str:
    t = f"<title>{NAME}</title>" if title else ""
    aria = 'role="img" aria-label="QuanAra"' if title else 'aria-hidden="true"'
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" {aria}>{t}{mark(uid, on_dark, animated)}</svg>'


def wordmark_html() -> str:
    """The name for HTML headings: "Quan" in ink and "Ara" in the accent colour."""
    return 'Quan<span class="q-acc">Ara</span>'


# Inline-page animation for the mark (Welcome hero): the trajectory draws, the forecast node settles.
MARK_CSS = """
<style>
.qm-tail { stroke-dasharray:100; stroke-dashoffset:100; animation:qmDraw 7s cubic-bezier(.4,0,.2,1) infinite; }
.qm-end { transform-box:fill-box; transform-origin:center; animation:qmEnd 7s cubic-bezier(.4,0,.2,1) infinite; }
@keyframes qmDraw { 0%,8% { stroke-dashoffset:100; } 34%,88% { stroke-dashoffset:0; } 100% { stroke-dashoffset:100; } }
@keyframes qmEnd { 0%,30% { transform:scale(0); opacity:0; } 40% { transform:scale(1.25); opacity:1; } 48%,86% { transform:scale(1); opacity:1; } 100% { transform:scale(0); opacity:0; } }
@media (prefers-reduced-motion: reduce) { .qm-tail { animation:none; stroke-dashoffset:0; } .qm-end { animation:none; transform:none; opacity:1; } }
</style>
"""


def logo_svg() -> str:
    """Sidebar wordmark on the black-green sidebar (rendered as an image, so system fonts only)."""
    font = "'Helvetica Neue', Helvetica, Arial, sans-serif"
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="76" viewBox="0 0 300 76" role="img" aria-label="QuanAra">'
            f'<title>{NAME} — {TAGLINE.replace('&', '&amp;')}</title>'
            f'<g transform="translate(2 8) scale(1.25)">{mark("lg", on_dark=True)}</g>'
            f'<text x="70" y="38" font-family="{font}" font-size="31" font-weight="800" fill="#ffffff" letter-spacing="0.4">Quan<tspan fill="{MINT}">Ara</tspan></text>'
            f'<text x="71" y="56" font-family="{font}" font-size="11.5" fill="#c9d3ce" letter-spacing="0.2">Quantitative AI for</text>'
            f'<text x="71" y="70" font-family="{font}" font-size="11.5" fill="#c9d3ce" letter-spacing="0.2">Aviation &amp; Tourism</text></svg>')


def favicon_svg(size: int = 64) -> str:
    """Mark on a rounded black-green tile so it reads on light and dark browser tabs."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 64 64">'
            f'<rect width="64" height="64" rx="14" fill="{BLACK}"/><g transform="translate(5 5) scale(1.12)">{mark("fv", on_dark=True)}</g></svg>')


def globe_svg() -> str:
    """Sidebar illustration: a dark globe with flight arcs linking destination nodes (decorative only)."""
    cx, cy, r = 150, 200, 150
    rng = random.Random(7)
    dots = []
    for _ in range(170):  # faint dot-matrix "land" texture inside the sphere
        x, y = rng.uniform(cx - r, cx + r), rng.uniform(cy - r, cy + r)
        if (x - cx) ** 2 + (y - cy) ** 2 < (r - 6) ** 2 and rng.random() < 0.55:
            dots.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="1" fill="{MINT}" opacity="{rng.uniform(.10, .32):.2f}"/>')
    grid = "".join(f'<ellipse cx="{cx}" cy="{cy}" rx="{r * k:.0f}" ry="{r}" fill="none" stroke="{MINT}" stroke-opacity=".13"/>' for k in (.28, .62, .9))
    grid += "".join(f'<ellipse cx="{cx}" cy="{cy - r * k:.0f}" rx="{r * (1 - k * k) ** .5:.0f}" ry="{r * .16 * (1 - k * k) ** .5:.0f}" fill="none" '
                    f'stroke="{MINT}" stroke-opacity=".13"/>' for k in (-.55, -.2, .2, .55))
    nodes = [(70, 118), (152, 92), (236, 128), (118, 176), (200, 190), (46, 196)]
    arcs = [(0, 1, -40), (1, 2, -36), (3, 1, -34), (1, 4, -26), (5, 3, -22), (3, 4, -30)]
    paths = []
    for a, b, lift in arcs:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        paths.append(f"M{x1} {y1} Q{(x1 + x2) / 2:.0f} {min(y1, y2) + lift:.0f} {x2} {y2}")
    arcs_svg = "".join(f'<path d="{d}" fill="none" stroke="{MINT}" stroke-width="1.4" stroke-opacity=".85" filter="url(#qg-glow)"/>'
                       f'<path d="{d}" fill="none" stroke="#d1fae5" stroke-width="1" class="qg-flow"/>' for d in paths)
    nodes_svg = "".join(f'<circle cx="{x}" cy="{y}" r="6" fill="{MINT}" opacity=".18"/><circle cx="{x}" cy="{y}" r="2.6" fill="#ecfdf5"/>' for x, y in nodes)
    plane = ('<path d="M0 3.2 L9 1.4 L13 -2.6 L14.8 -2.6 L12.4 1.2 L17 0.4 L18.4 -1.4 L19.6 -1.4 L19 1.4 L19.6 4 L18.4 4 L17 2.4 '
             'L12.4 1.8 L14.8 5.6 L13 5.6 L9 2.2 Z" fill="#ffffff"/>')
    route = "M30 70 Q150 10 272 64"
    return ('<svg viewBox="0 0 300 260" preserveAspectRatio="xMidYMax slice" aria-hidden="true">'
            '<defs><radialGradient id="qg-sphere" cx="42%" cy="34%" r="70%"><stop offset="0" stop-color="#0d3324"/><stop offset=".6" stop-color="#06170f"/>'
            '<stop offset="1" stop-color="#020604"/></radialGradient>'
            '<filter id="qg-glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="1.6"/></filter></defs>'
            f'<circle cx="{cx}" cy="{cy}" r="{r + 10}" fill="{BRIGHT}" opacity=".07"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#qg-sphere)" stroke="{MINT}" stroke-opacity=".35"/>'
            + grid + "".join(dots) + arcs_svg + nodes_svg
            + f'<path d="{route}" fill="none" stroke="#d1fae5" stroke-opacity=".35" stroke-dasharray="2 5"/>'
            f'<g class="qg-motion"><g>{plane}<animateMotion path="{route}" dur="9s" repeatCount="indefinite" rotate="auto"/></g></g>'
            f'<g class="qg-static" transform="translate(196 34) rotate(8)">{plane}</g></svg>')
