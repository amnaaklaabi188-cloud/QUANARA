"""QuanAra animated page banners — one reusable component, ten page-specific illustrations.

Layout (every page): a strict two-zone grid. The left ≈63% holds the section label, title and one
subtitle and never contains decoration. The right ≈37% holds a self-contained SVG in its own clipped
box; nothing animates outside it.

Design system shared by all variants (the QuanAra visual language): navy text, royal blue
(#059669), soft violet (#047857), pale cyan (#6ee7b7) and white; 1.6 px strokes; a faint dotted
coordinate grid; slowly drifting quantitative data particles; the brand's open coordinate ring
("Q" node) marking destinations and merge points; one 7 s cycle with smooth easing everywhere.
Each variant depicts its section's scientific step:

  overview     an aircraft trajectory connects to a destination fix above the Abu Dhabi skyline
  data         particles flowing from aviation and hospitality databases into one merge node
  features     scattered raw-data nodes reorganising into a structured feature network
  correlation  scatter points appearing, then a regression line drawing through them
  training     a layered model graph with pulses on its links, feeding a tree glyph and a line glyph
  evaluation   an actual curve and a slightly different predicted curve drawing in
  simulator    baseline and scenario trajectories diverging subtly, with a demand-response curve
  copilot      connected analytical nodes pulsing around a central evidence node
  methodology  four stages (Data · Features · Models · Simulation) lighting up in sequence
  demo         aircraft → chart → model → skyline, joined by a travelling highlight

Motion uses SMIL / CSS on transform, opacity and stroke-dashoffset. With prefers-reduced-motion all
animation stops and a static frame is shown; on narrow screens the artwork is hidden. Illustrations
are decorative (no axes values, no numbers) and never stand in for results.
"""
from __future__ import annotations

import random
from html import escape

from .hero import _plane, _skyline

W, H = 600, 220
BLUE, VIOLET, CYAN, INKL = "#059669", "#047857", "#6ee7b7", "#bcd8cb"  # emerald, deep green, mint, light rule
CYCLE = 7  # seconds

CSS = """
<style>
.hz { display:grid; grid-template-columns:minmax(0,62fr) minmax(0,38fr); overflow:hidden; border-radius:20px; border:1px solid #e1ebe6; height:160px;
      background:linear-gradient(100deg,#ffffff 0%,#fbfdfc 42%,#eef8f3 72%,#dff2e8 100%); box-shadow:0 1px 2px rgba(15,23,20,.04),0 10px 28px rgba(15,23,20,.07);
      animation:dsFadeUp .5s ease both; }
.hz-text { align-self:center; padding:20px 18px 20px 30px; font-family:'Plus Jakarta Sans','Inter',-apple-system,'Segoe UI',sans-serif; min-width:0; }
.hz-kick { font-size:12px; font-weight:800; letter-spacing:.14em; color:#047857; text-transform:uppercase; }
.hz-title { font-weight:800; letter-spacing:-.02em; line-height:1.08; color:#0b1110; margin-top:6px; font-size:36px; }
.hz-mark { display:inline-flex; vertical-align:-5px; margin-right:12px; } .hz-mark svg { width:46px; height:46px; }
.hz-title span { background:linear-gradient(90deg,#047857,#10b981); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hz-sub { font-size:14.5px; color:#444846; margin-top:8px; line-height:1.5; max-width:620px; }
.hz-art { position:relative; overflow:hidden; min-width:0;
          -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 14%); mask-image:linear-gradient(90deg,transparent 0,#000 14%); }
.hz-art > svg { position:absolute; inset:0; width:100%; height:100%; display:block; }
.hz-pill { position:absolute; right:12px; top:10px; display:flex; gap:7px; align-items:center; padding:6px 10px; border-radius:11px;
           background:rgba(255,255,255,.88); border:1px solid rgba(255,255,255,.95); box-shadow:0 4px 12px rgba(15,23,20,.07);
           font-family:'Plus Jakarta Sans',-apple-system,sans-serif; z-index:2; }
.hz-pill svg { width:15px; height:15px; color:#059669; flex:none; }
.hz-pill b { display:block; font-size:12px; color:#101614; line-height:1.2; } .hz-pill small { font-size:10.8px; color:#059669; font-weight:600; }
/* shared motion vocabulary (7 s loops) */
@keyframes hzDash { to { stroke-dashoffset:-60; } }
@keyframes hzPulse { 0%,100% { opacity:.3; transform:scale(.8); } 50% { opacity:1; transform:scale(1.1); } }
@keyframes hzAppear { 0% { opacity:0; transform:translateY(4px); } 10% { opacity:1; transform:none; } 86% { opacity:1; } 100% { opacity:0; } }
@keyframes hzDrawLoop { 0%, 28% { stroke-dashoffset:1000; opacity:1; } 52% { stroke-dashoffset:0; } 86% { stroke-dashoffset:0; opacity:1; } 100% { stroke-dashoffset:0; opacity:0; } }
@keyframes hzDrawLate { 0%, 40% { stroke-dashoffset:1000; opacity:1; } 64% { stroke-dashoffset:0; } 86% { stroke-dashoffset:0; opacity:1; } 100% { stroke-dashoffset:0; opacity:0; } }
@keyframes hzLight { 0%, 100% { fill:#ffffff; } 12%, 40% { fill:#059669; } }
@keyframes hzLightText { 0%, 100% { fill:#059669; } 12%, 40% { fill:#ffffff; } }
@keyframes hzGlow { 0%,100% { opacity:.25; } 50% { opacity:.95; } }
@keyframes hzSpin { to { transform:rotate(360deg); } }
.hz .dash { stroke-dasharray:3 7; animation:hzDash 3.5s linear infinite; }
.hz .pulse { transform-box:fill-box; transform-origin:center; animation:hzPulse 3.5s ease-in-out infinite; }
.hz .appear { opacity:0; transform-box:fill-box; animation:hzAppear 7s cubic-bezier(.4,0,.2,1) infinite; }
.hz .drawloop { stroke-dasharray:1000; stroke-dashoffset:1000; animation:hzDrawLoop 7s cubic-bezier(.4,0,.2,1) infinite; }
.hz .drawlate { stroke-dasharray:1000; stroke-dashoffset:1000; animation:hzDrawLate 7s cubic-bezier(.4,0,.2,1) infinite; }
.hz .light { animation:hzLight 7s ease-in-out infinite; } .hz .lighttext { animation:hzLightText 7s ease-in-out infinite; }
.hz .glow { animation:hzGlow 3.5s ease-in-out infinite; }
.hz .spin { transform-box:fill-box; transform-origin:center; animation:hzSpin 24s linear infinite; }
.hz .static { display:none; }
@media (prefers-reduced-motion: reduce) {
  .hz, .hz * { animation:none !important; }
  .hz .appear { opacity:1; } .hz .drawloop, .hz .drawlate { stroke-dashoffset:0; }
  .hz .motion { display:none; } .hz .static { display:inline; }
}
@media (max-width: 1100px) { .hz-pill { display:none; } }
@media (max-width: 760px) { .hz { grid-template-columns:1fr; height:auto; } .hz-art { display:none; } .hz-text { padding:18px 20px; } .hz-title { font-size:30px !important; } }
</style>
"""

_PIN = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>')

_DEFS = f"""<defs>
<linearGradient id="ov-glassA" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c9ddd4"/><stop offset="1" stop-color="#557f6e"/></linearGradient>
<linearGradient id="ov-glassB" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e1ece7"/><stop offset="1" stop-color="#7fa595"/></linearGradient>
<linearGradient id="ov-glassC" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8fb3a3"/><stop offset="1" stop-color="#244a3c"/></linearGradient>
<linearGradient id="ov-glassD" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#eef4f1"/><stop offset="1" stop-color="#adc6ba"/></linearGradient>
<radialGradient id="hz-glow" cx="380" cy="90" r="280" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#ffffff" stop-opacity="0.95"/><stop offset="0.6" stop-color="#e6f5ee" stop-opacity="0.4"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/></radialGradient>
<linearGradient id="hz-trail" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{CYAN}" stop-opacity="0"/><stop offset="0.5" stop-color="#10b981" stop-opacity="0.9"/><stop offset="1" stop-color="{VIOLET}" stop-opacity="0.9"/></linearGradient>
<pattern id="hz-dots" width="16" height="16" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#c3ddd1"/></pattern>
</defs>"""


# ------------------------------------------------------------------ shared building blocks

def _backdrop() -> str:
    return f'<rect width="{W}" height="{H}" fill="url(#hz-dots)" opacity="0.55"/><rect width="{W}" height="{H}" fill="url(#hz-glow)"/>'


def _skyline_full(scale: float = 0.6, right: float = 590, base: float = 206) -> str:
    tx, ty = right - 1190 * scale, base - 268 * scale
    rng, haze, x = random.Random(21), [], right - 644 * scale - 20
    while x < right:
        w, h = rng.randint(7, 14), rng.randint(14, 54)
        haze.append(f'<rect x="{x:.0f}" y="{base - h}" width="{w}" height="{h}" fill="#cdf0e3"/>')
        x += w + rng.randint(0, 3)
    return ("".join(haze) + f'<g transform="translate({tx:.1f} {ty:.1f}) scale({scale})">{_skyline()}</g>'
            f'<line x1="{right - 660 * scale}" y1="{base}" x2="{W}" y2="{base}" stroke="{INKL}" stroke-width="1"/>')


def _node(x: float, y: float, r: float = 4.0, color: str = BLUE, delay: float = 0.0, fill: str = "#ffffff") -> str:
    return (f'<circle cx="{x}" cy="{y}" r="{r * 2.4:.1f}" fill="{color}" opacity="0.16" class="pulse" style="animation-delay:{delay:.2f}s"/>'
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="1.6"/>')


def _plane_on(path: str, scale: float = 0.42, dur: float = CYCLE, static_xy: tuple[float, float, float] = (0, 0, 0)) -> str:
    plane = f'<g transform="scale({scale})">{_plane()}</g>'
    return (f'<g class="motion"><animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.82;1" dur="{dur}s" repeatCount="indefinite"/>'
            f'<path d="{path}" pathLength="1000" fill="none" stroke="url(#hz-trail)" stroke-width="1.8" stroke-dasharray="1000" stroke-dashoffset="1000">'
            f'<animate attributeName="stroke-dashoffset" values="1000;0;0" keyTimes="0;0.82;1" dur="{dur}s" repeatCount="indefinite"/></path>'
            f'<g>{plane}<animateMotion path="{path}" rotate="auto" keyPoints="0;1;1" keyTimes="0;0.82;1" keySplines="0.4 0 0.2 1;0 0 1 1" '
            f'calcMode="spline" dur="{dur}s" repeatCount="indefinite"/></g></g>'
            f'<g class="static"><path d="{path}" fill="none" stroke="url(#hz-trail)" stroke-width="1.8" opacity="0.8"/>'
            f'<g transform="translate({static_xy[0]} {static_xy[1]}) rotate({static_xy[2]})">{plane}</g></g>')


def _qnode(x: float, y: float, r: float = 9.0, delay: float = 0.0) -> str:
    """The brand's open coordinate ring with a centre point and a short forecast tail (the "Q" mark, simplified)."""
    import math
    p = lambda a, rr=r: (x + rr * math.cos(math.radians(a)), y + rr * math.sin(math.radians(a)))  # noqa: E731
    (sx, sy), (ex, ey), (tx, ty), (ux, uy) = p(70), p(20), p(45, r * 0.45), p(45, r * 1.55)
    return (f'<circle cx="{x}" cy="{y}" r="{r * 1.9:.1f}" fill="{BLUE}" opacity="0.10" class="pulse" style="animation-delay:{delay:.2f}s"/>'
            f'<path d="M{sx:.1f} {sy:.1f} A{r} {r} 0 1 1 {ex:.1f} {ey:.1f}" fill="#ffffff" fill-opacity="0.85" stroke="{BLUE}" stroke-width="2" stroke-linecap="round"/>'
            f'<line x1="{tx:.1f}" y1="{ty:.1f}" x2="{ux:.1f}" y2="{uy:.1f}" stroke="{VIOLET}" stroke-width="1.8" stroke-linecap="round"/>'
            f'<circle cx="{ux:.1f}" cy="{uy:.1f}" r="2.2" fill="{VIOLET}"/><circle cx="{x}" cy="{y}" r="2.6" fill="{CYAN}" stroke="{BLUE}" stroke-width="1.2"/>')


def _signature(seed: int) -> str:
    """Shared QuanAra layer drawn under every illustration: a corner coordinate fix and drifting data particles."""
    rng = random.Random(seed)
    parts = []
    for i in range(7):
        x, y = rng.uniform(190, 560), rng.uniform(30, 196)
        col = (CYAN, BLUE, VIOLET)[i % 3]
        b = -CYCLE * i / 7
        parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="3" height="3" rx="0.8" fill="{col}" opacity="0">'
                     f'<animate attributeName="opacity" values="0;0.55;0.55;0" keyTimes="0;0.2;0.75;1" dur="{CYCLE}s" begin="{b:.2f}s" repeatCount="indefinite"/>'
                     f'<animateTransform attributeName="transform" type="translate" values="0 0;22 -12" dur="{CYCLE}s" begin="{b:.2f}s" repeatCount="indefinite"/></rect>')
    fix = (f'<g opacity="0.55"><path d="M583.4 32.6 A8 8 0 1 1 587.5 24.7" fill="none" stroke="{BLUE}" stroke-width="1.4" stroke-linecap="round"/>'
           f'<circle cx="580" cy="25" r="1.6" fill="{BLUE}"/><path d="M580 12 v4 M580 34 v4 M567 25 h4 M589 25 h4" stroke="{INKL}" stroke-width="1.2" stroke-linecap="round"/></g>')
    return f'<g class="motion">{"".join(parts)}</g>{fix}'


def _axes(x0=150, y0=184, x1=560, y1=34) -> str:
    return (f'<path d="M{x0} {y1} V{y0} H{x1}" fill="none" stroke="{INKL}" stroke-width="1.4" stroke-linecap="round"/>')


# ------------------------------------------------------------------ the ten illustrations

def _overview() -> str:
    hotel_windows = "".join(f'<rect x="{x}" y="{y}" width="3.5" height="2.6" rx="1" fill="#ffe7a8" class="glow" style="animation-delay:{d:.1f}s"/>'
                            for x, y, d in ((494, 136, 0), (499, 150, .6), (494, 164, 1.2), (521, 128, .3), (526, 146, .9), (521, 162, 1.5), (548, 150, .4), (553, 166, 1.1)))
    link = (f'<path d="M290 84 C340 66 392 52 442 58" fill="none" stroke="{CYAN}" stroke-width="1.5" class="dash" opacity="0.9"/>'
            f'<path d="M452 70 V96" stroke="{INKL}" stroke-width="1.2" stroke-dasharray="2 3"/>')
    return (_skyline_full(scale=0.47) + hotel_windows + link + _qnode(452, 60, 8, 1.2)
            + _plane_on("M140 62 C190 56 236 64 284 82", 0.4, CYCLE, (250, 68, 16)))


def _db(x: float, y: float, glyph: str) -> str:
    return (f'<g><ellipse cx="{x}" cy="{y}" rx="34" ry="10" fill="#ffffff" stroke="{BLUE}" stroke-width="1.6"/>'
            f'<path d="M{x - 34} {y} v46 a34 10 0 0 0 68 0 v-46" fill="#e2fcf2" stroke="{BLUE}" stroke-width="1.6"/>'
            f'<path d="M{x - 34} {y + 23} a34 10 0 0 0 68 0" fill="none" stroke="{BLUE}" stroke-width="1.2" opacity="0.5"/>'
            f'<g transform="translate({x - 9} {y + 18})">{glyph}</g></g>')


def _data() -> str:
    plane = f'<path d="M0 9 L18 4 L18 7 L9 10 L12 17 L9 17 L5 11 L0 12 Z" fill="{BLUE}"/>'
    bed = f'<path d="M0 6 V17 M0 12 H18 V17 M3 8 H9 V12" stroke="{VIOLET}" stroke-width="2" fill="none" stroke-linecap="round"/>'
    up, lo = "M224 104 C270 70 310 76 350 104", "M496 104 C440 70 400 76 350 104"
    parts = "".join(f'<circle r="3" fill="{c}"><animateMotion path="{p}" dur="3.5s" begin="{-0.7 * i:.1f}s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear"/>'
                    f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.15;0.8;1" dur="3.5s" begin="{-0.7 * i:.1f}s" repeatCount="indefinite"/></circle>'
                    for i in range(5) for p, c in ((up, BLUE), (lo, VIOLET)))
    return (_db(190, 88, plane) + _db(530, 88, bed)
            + f'<path d="{up}" fill="none" stroke="#85ebc5" stroke-width="1.6"/><path d="{lo}" fill="none" stroke="#6befbe" stroke-width="1.6"/>'
            + f'<g class="motion">{parts}</g>' + _qnode(350, 104, 10)
            + f'<path d="M350 112 V150" stroke="{CYAN}" stroke-width="1.6" stroke-dasharray="2 4"/><rect x="326" y="150" width="48" height="28" rx="6" fill="#ffffff" stroke="{CYAN}" stroke-width="1.6"/>'
            + "".join(f'<line x1="334" y1="{158 + 6 * k}" x2="366" y2="{158 + 6 * k}" stroke="#c2ecdd" stroke-width="2" stroke-linecap="round"/>' for k in range(3)))


def _features() -> str:
    rng = random.Random(9)
    grid = [(250 + 90 * c, 62 + 48 * r) for c in range(3) for r in range(3)]
    scat = [(rng.uniform(180, 560), rng.uniform(40, 190)) for _ in grid]
    colors = [BLUE, VIOLET, "#34d399"]
    edges_anim, edges_static, nodes_anim, nodes_static = [], [], [], []
    links = [(i, i + 1) for i in range(9) if i % 3 != 2] + [(i, i + 3) for i in range(6)]
    for a, b in links:
        (x1, y1), (x2, y2) = grid[a], grid[b]
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#71e4ba" stroke-width="1.4"/>'
        edges_static.append(line)
        edges_anim.append(line.replace("/>", f'><animate attributeName="opacity" values="0;0;1;1;0" keyTimes="0;0.34;0.46;0.86;1" dur="{CYCLE}s" repeatCount="indefinite"/></line>'))
    for i, ((gx, gy), (sx, sy)) in enumerate(zip(grid, scat)):
        col = colors[i // 3]
        spl = 'calcMode="spline" keySplines="0.4 0 0.2 1;0 0 1 1;0.4 0 0.2 1"'
        nodes_anim.append(f'<circle r="6" fill="#ffffff" stroke="{col}" stroke-width="1.8">'
                          f'<animate attributeName="cx" values="{sx:.0f};{gx};{gx};{sx:.0f}" keyTimes="0;0.36;0.86;1" {spl} dur="{CYCLE}s" repeatCount="indefinite"/>'
                          f'<animate attributeName="cy" values="{sy:.0f};{gy};{gy};{sy:.0f}" keyTimes="0;0.36;0.86;1" {spl} dur="{CYCLE}s" repeatCount="indefinite"/></circle>')
        nodes_static.append(f'<circle cx="{gx}" cy="{gy}" r="6" fill="#ffffff" stroke="{col}" stroke-width="1.8"/>')
    heads = "".join(f'<rect x="{250 + 90 * c - 16}" y="28" width="32" height="5" rx="2.5" fill="{colors[c]}" opacity="0.55"/>' for c in range(3))
    return (f'<g class="motion">{"".join(edges_anim)}{"".join(nodes_anim)}</g><g class="static">{"".join(edges_static)}{"".join(nodes_static)}</g>' + heads)


def _correlation() -> str:
    rng = random.Random(4)
    pts, xy = [], []
    for i in range(28):
        x = 172 + i * 13 + rng.uniform(-5, 5)
        y = 170 - (x - 172) * 0.34 + rng.gauss(0, 11)
        xy.append((x, y))
        pts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{BLUE}" fill-opacity="0.6" class="appear" style="animation-delay:{0.06 * i:.2f}s"/>')
    rel = "".join(f'<line x1="{xy[i][0]:.1f}" y1="{xy[i][1]:.1f}" x2="{xy[i + 3][0]:.1f}" y2="{xy[i + 3][1]:.1f}" stroke="{CYAN}" stroke-width="1.1" '
                  f'class="appear" style="animation-delay:{0.9 + 0.08 * i:.2f}s"/>' for i in range(2, 25, 4))
    return (_axes(156, 188, 560, 36) + rel + "".join(pts)
            + f'<path d="M166 176 L548 46" pathLength="1000" fill="none" stroke="{VIOLET}" stroke-width="2.2" stroke-linecap="round" class="drawloop"/>')


def _training() -> str:
    layers = [(220, [70, 108, 146]), (290, [52, 90, 128, 166]), (360, [84, 132])]
    links, link_paths, nodes = [], [], []
    for (x0, ys0), (x1, ys1) in zip(layers, layers[1:]):
        for a in ys0:
            for b in ys1:
                links.append(f'<line x1="{x0}" y1="{a}" x2="{x1}" y2="{b}" stroke="#8ce5c4" stroke-width="1.1"/>')
                link_paths.append(f"M{x0} {a} L{x1} {b}")
    for li, (x, ys) in enumerate(layers):
        for j, y in enumerate(ys):
            nodes.append(_node(x, y, 5, VIOLET if li == 1 else BLUE, 0.3 * (li + j)))
    pulses = "".join(f'<circle r="2.6" fill="{CYAN if i % 2 else BLUE}"><animateMotion path="{p}" dur="2.4s" begin="{-0.35 * i:.2f}s" repeatCount="indefinite"/>'
                     f'<animate attributeName="opacity" values="0;1;0" dur="2.4s" begin="{-0.35 * i:.2f}s" repeatCount="indefinite"/></circle>'
                     for i, p in enumerate(link_paths[::3]))
    tree = (f'<g transform="translate(430 64)"><line x1="20" y1="8" x2="6" y2="34" stroke="{BLUE}" stroke-width="1.6"/><line x1="20" y1="8" x2="34" y2="34" stroke="{BLUE}" stroke-width="1.6"/>'
            f'<circle cx="20" cy="8" r="6" fill="#ffffff" stroke="{BLUE}" stroke-width="1.6"/><circle cx="6" cy="36" r="5" fill="#ffffff" stroke="{VIOLET}" stroke-width="1.6"/>'
            f'<circle cx="34" cy="36" r="5" fill="#ffffff" stroke="{VIOLET}" stroke-width="1.6"/></g>')
    linear = (f'<g transform="translate(430 128)"><path d="M0 34 V0 M0 34 H44" stroke="{INKL}" stroke-width="1.4" fill="none"/>'
              f'<path d="M4 28 L40 6" stroke="{CYAN}" stroke-width="2" stroke-linecap="round"/></g>')
    return ("".join(links) + f'<g class="motion">{pulses}</g>' + "".join(nodes)
            + f'<path d="M366 84 C392 84 404 80 428 76" fill="none" stroke="#8ce5c4" stroke-width="1.3" class="dash"/>'
            + f'<path d="M366 132 C392 132 404 140 428 146" fill="none" stroke="#8ce5c4" stroke-width="1.3" class="dash"/>' + tree + linear)


def _evaluation() -> str:
    xs = [170 + i * 32 for i in range(13)]
    act = [150, 132, 140, 112, 122, 96, 108, 84, 92, 74, 82, 66, 72]
    pred = [146, 138, 134, 120, 116, 104, 102, 92, 88, 82, 78, 72, 70]
    pa = "M" + " L".join(f"{x} {y}" for x, y in zip(xs, act))
    pp = "M" + " L".join(f"{x} {y}" for x, y in zip(xs, pred))
    legend = (f'<g transform="translate(470 34)"><line x1="0" y1="0" x2="16" y2="0" stroke="{BLUE}" stroke-width="2.4" stroke-linecap="round"/>'
              f'<line x1="0" y1="12" x2="16" y2="12" stroke="{VIOLET}" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="4 3"/></g>')
    return (_axes(158, 188, 572, 30)
            + f'<path d="{pa}" pathLength="1000" fill="none" stroke="{BLUE}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" class="drawloop"/>'
            + f'<path d="{pp}" pathLength="1000" fill="none" stroke="{VIOLET}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="1000" class="drawlate" opacity="0.9"/>'
            + legend)


def _simulator() -> str:
    base, scen = "M170 150 C260 140 360 128 470 118", "M170 150 C260 138 360 112 470 88"
    resp_b = '<path d="M372 196 C410 194 450 194 520 193" fill="none" stroke="#b2dbcc" stroke-width="1.8" stroke-dasharray="4 4"/>'
    resp_s = f'<path d="M372 196 C410 192 450 182 520 172" pathLength="1000" fill="none" stroke="{CYAN}" stroke-width="2.2" stroke-linecap="round" class="drawlate"/>'
    hotel = (f'<g transform="translate(528 160)"><rect x="0" y="4" width="24" height="30" rx="3" fill="#ffffff" stroke="{VIOLET}" stroke-width="1.6"/>'
             f'<path d="M7 13 v12 M17 13 v12 M7 19 h10" stroke="{VIOLET}" stroke-width="1.8" stroke-linecap="round"/></g>')
    return (f'<path d="{base}" fill="none" stroke="#b2dbcc" stroke-width="1.8" stroke-dasharray="5 5"/>'
            + _node(170, 150, 4.5, BLUE) + _node(470, 118, 3.5, "#a5a9a7") + _node(470, 88, 3.5, VIOLET, 0.5)
            + f'<path d="M478 92 C488 98 488 108 478 114" fill="none" stroke="{INKL}" stroke-width="1.2" stroke-dasharray="2 3"/>'
            + _plane_on(scen, 0.4, CYCLE, (389, 106, -12)) + resp_b + resp_s + hotel)


def _copilot() -> str:
    cx, cy = 360, 108
    pts = [(cx, cy - 52), (cx + 46, cy - 24), (cx + 46, cy + 28), (cx, cy + 54), (cx - 46, cy + 28), (cx - 46, cy - 24), (cx, cy)]
    lines = "".join(f'<line x1="{pts[i][0]}" y1="{pts[i][1]}" x2="{pts[j][0]}" y2="{pts[j][1]}" stroke="#8ce5c4" stroke-width="1.3"/>'
                    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (6, 0), (6, 2), (6, 4), (6, 1), (6, 3), (6, 5)])
    nodes = "".join(_node(x, y, 4.5, BLUE, 0.45 * k) for k, (x, y) in enumerate(pts[:6])) + _qnode(cx, cy, 11, 0.6)
    orbit = f'<g class="spin"><circle cx="{cx}" cy="{cy}" r="84" fill="none" stroke="{CYAN}" stroke-width="1.4" stroke-dasharray="2 10"/><circle cx="{cx + 84}" cy="{cy}" r="3.2" fill="{CYAN}"/></g>'
    return orbit + lines + nodes


def _methodology() -> str:
    xs, y = [196, 296, 396, 496], 104
    labels = ["Data", "Features", "Models", "Simulation"]
    out = [f'<line x1="{xs[0]}" y1="{y}" x2="{xs[-1]}" y2="{y}" stroke="#8ce5c4" stroke-width="1.8"/>']
    for i, (x, lab) in enumerate(zip(xs, labels)):
        d = i * CYCLE / 4.6
        out.append(f'<circle cx="{x}" cy="{y}" r="24" fill="{BLUE}" opacity="0.12" class="glow" style="animation-delay:{d:.2f}s"/>'
                   f'<circle cx="{x}" cy="{y}" r="16" fill="#ffffff" stroke="{BLUE}" stroke-width="1.8" class="light" style="animation-delay:{d:.2f}s"/>'
                   f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="13" font-weight="800" fill="{BLUE}" font-family="Plus Jakarta Sans, sans-serif" class="lighttext" style="animation-delay:{d:.2f}s">{i + 1}</text>'
                   f'<text x="{x}" y="{y + 42}" text-anchor="middle" font-size="12.5" font-weight="650" fill="#515553" font-family="Plus Jakarta Sans, sans-serif">{lab}</text>')
    return "".join(out)


def _demo() -> str:
    xs, y = [190, 290, 390, 500], 104
    plane = f'<g transform="translate({xs[0] - 15} {y - 9}) scale(0.34)">{_plane()}</g>'
    chart = "".join(f'<rect x="{xs[1] - 14 + 10 * k}" y="{y + 12 - h}" width="7" height="{h}" rx="2" fill="{BLUE if k == 2 else "#a7f3d0"}"/>' for k, h in enumerate((10, 16, 24)))
    model = "".join(f'<circle cx="{xs[2] + dx}" cy="{y + dy}" r="4" fill="#ffffff" stroke="{VIOLET}" stroke-width="1.6"/>' for dx, dy in ((-12, -8), (-12, 8), (10, 0)))
    model = (f'<line x1="{xs[2] - 12}" y1="{y - 8}" x2="{xs[2] + 10}" y2="{y}" stroke="#8ce5c4" stroke-width="1.3"/>'
             f'<line x1="{xs[2] - 12}" y1="{y + 8}" x2="{xs[2] + 10}" y2="{y}" stroke="#8ce5c4" stroke-width="1.3"/>' + model)
    city = "".join(f'<rect x="{xs[3] - 16 + 11 * k}" y="{y + 14 - h}" width="9" height="{h}" rx="1.5" fill="url(#ov-glass{"ACB"[k]})"/>' for k, h in enumerate((20, 30, 24)))
    rings = "".join(f'<circle cx="{x}" cy="{y}" r="26" fill="#ffffff" stroke="#c6f0dc" stroke-width="1.6"/>'
                    f'<circle cx="{x}" cy="{y}" r="26" fill="none" stroke="{BLUE}" stroke-width="1.8" class="glow" style="animation-delay:{i * CYCLE / 4.4:.2f}s"/>' for i, x in enumerate(xs))
    track = f"M{xs[0]} {y} L{xs[-1]} {y}"
    dot = (f'<circle r="4" fill="{CYAN}" class="motion"><animateMotion path="{track}" dur="{CYCLE}s" repeatCount="indefinite" keyPoints="0;1;1" keyTimes="0;0.85;1" calcMode="linear"/></circle>')
    return (f'<line x1="{xs[0]}" y1="{y}" x2="{xs[-1]}" y2="{y}" stroke="#8ce5c4" stroke-width="1.6" stroke-dasharray="3 6"/>' + rings + plane + chart + model + city + dot
            + _qnode(xs[3] + 22, y - 34, 6.5, 2.0))


VARIANTS = {"overview": _overview, "data": _data, "features": _features, "correlation": _correlation, "training": _training,
            "evaluation": _evaluation, "simulator": _simulator, "copilot": _copilot, "methodology": _methodology, "demo": _demo}


def banner(variant: str, kicker: str, title_html: str, subtitle: str, subtitle2: str = "", pill: tuple[str, str] | None = None,
           pill_style: str = "place", chip: bool = False, height: int = 160, title_px: int = 36) -> str:
    """title_html is trusted, app-generated markup; every other argument is escaped. `subtitle2`, `pill_style`, `chip`
    and `height` are accepted for compatibility: banners show one subtitle and share one height."""
    seed = sorted(VARIANTS).index(variant) + 3
    svg = (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMaxYMid slice" aria-hidden="true">{_DEFS}{_backdrop()}{_signature(seed)}'
           f'{VARIANTS[variant]()}</svg>')
    pill_html = ""  # location badges are no longer drawn over the artwork; the kicker carries that context
    return (f'<div class="hz"><div class="hz-text"><div class="hz-kick">{escape(kicker)}</div>'
            f'<div class="hz-title" style="font-size:{title_px}px">{title_html}</div><div class="hz-sub">{escape(subtitle)}</div></div>'
            f'<div class="hz-art">{svg}{pill_html}</div></div>')
