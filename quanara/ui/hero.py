"""Animated aviation hero: an inline-SVG Abu Dhabi skyline with an aircraft that flies along a
curved path while its glowing trail draws in behind it (SMIL, no JavaScript, no external files).
Users with prefers-reduced-motion get a static aircraft and trail instead."""
from __future__ import annotations

import math
import random

B = 268  # waterline / building baseline in viewBox units (viewBox 0 0 1200 320)
P0, P1, P2, P3 = (560, 206), (720, 196), (880, 150), (1072, 70)  # flight path (cubic Bézier)
FLIGHT_PATH = f"M{P0[0]} {P0[1]} C{P1[0]} {P1[1]} {P2[0]} {P2[1]} {P3[0]} {P3[1]}"


def _bez(t: float) -> tuple[float, float]:
    u = 1 - t
    x = u**3 * P0[0] + 3 * u * u * t * P1[0] + 3 * u * t * t * P2[0] + t**3 * P3[0]
    y = u**3 * P0[1] + 3 * u * u * t * P1[1] + 3 * u * t * t * P2[1] + t**3 * P3[1]
    return x, y


def _point_at_length(frac: float) -> tuple[float, float, float]:
    """Point and heading (deg) at a fraction of the path's arc length (for the static frame)."""
    pts = [_bez(i / 400) for i in range(401)]
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    target, acc = frac * sum(seg), 0.0
    for i, s in enumerate(seg):
        if acc + s >= target:
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            return x0, y0, math.degrees(math.atan2(y1 - y0, x1 - x0))
        acc += s
    return pts[-1][0], pts[-1][1], 0.0


def _haze(rng: random.Random, x0: int, x1: int, hmin: int, hmax: int, fill: str, opacity: float) -> str:
    out, x = [], x0
    while x < x1:
        w = rng.randint(10, 24)
        h = rng.randint(hmin, hmax)
        out.append(f'<rect x="{x}" y="{B - h}" width="{w}" height="{h}" fill="{fill}" opacity="{opacity}"/>')
        x += w + rng.randint(0, 3)
    return "".join(out)


def _windows(x: float, w: float, top: float, step: int = 9, color: str = "#ffffff", op: float = 0.28) -> str:
    return "".join(f'<line x1="{x + 2}" y1="{y}" x2="{x + w - 2}" y2="{y}" stroke="{color}" stroke-width="1" opacity="{op}"/>'
                   for y in range(int(top) + 10, B - 4, step))


def _skyline() -> str:
    s = []
    # Sheikh Zayed Grand Mosque (white domes and minarets)
    s.append('<g fill="#f0faf7" stroke="#add8c9" stroke-width="0.9">'
             '<rect x="548" y="246" width="104" height="22"/>'
             '<path d="M582 246 C582 216 618 216 618 246 Z"/><path d="M558 246 C558 231 574 231 574 246 Z"/>'
             '<path d="M626 246 C626 231 642 231 642 246 Z"/>'
             '<rect x="549" y="192" width="5" height="54"/><path d="M548 192 L551.5 180 L555 192 Z"/>'
             '<rect x="646" y="192" width="5" height="54"/><path d="M645 192 L648.5 180 L652 192 Z"/>'
             '<rect x="547" y="205" width="9" height="3"/><rect x="644" y="205" width="9" height="3"/>'
             '<rect x="599.2" y="207" width="1.6" height="9" stroke="none" fill="#c9a85a"/></g>')
    # Aldar HQ (circular disc)
    s.append('<clipPath id="ov-disc"><circle cx="700" cy="231" r="36"/></clipPath>'
             '<circle cx="700" cy="231" r="36" fill="url(#ov-glassB)" stroke="#54b692" stroke-width="1"/>'
             '<g clip-path="url(#ov-disc)" stroke="#ffffff" stroke-width="1" opacity="0.45">'
             + "".join(f'<line x1="660" y1="{y}" x2="740" y2="{y}"/>' for y in range(199, 268, 8))
             + "".join(f'<line x1="{x}" y1="190" x2="{x}" y2="270"/>' for x in range(668, 736, 10)) + "</g>")
    # Capital Gate (leaning tower with its "splash")
    s.append('<path d="M752 268 L776 268 L800 132 L779 129 Z" fill="url(#ov-glassA)" stroke="#4cb08b" stroke-width="0.8"/>'
             '<path d="M746 268 C758 232 770 204 806 178" fill="none" stroke="#74cdac" stroke-width="3" stroke-linecap="round"/>')
    # Etihad Towers (five curved-top towers)
    for i, (x, w, h, g) in enumerate([(815, 20, 140, "A"), (838, 22, 185, "C"), (863, 24, 212, "A"), (890, 22, 172, "C"), (915, 20, 135, "A")]):
        top = B - h
        s.append(f'<path d="M{x} {B} L{x} {top + 16} Q{x + w * 0.45} {top - 6} {x + w} {top} L{x + w} {B} Z" fill="url(#ov-glass{g})" stroke="#4ba886" stroke-width="0.6"/>'
                 f'<line x1="{x + w * 0.3}" y1="{top + 12}" x2="{x + w * 0.3}" y2="{B}" stroke="#ffffff" stroke-width="1.2" opacity="0.35"/>')
    # Central tall towers
    s.append('<path d="M950 268 L950 60 L963 36 L976 60 L976 268 Z" fill="url(#ov-glassC)" stroke="#449b7b" stroke-width="0.6"/>'
             '<line x1="963" y1="36" x2="963" y2="18" stroke="#4fb28e" stroke-width="1.5"/>' + _windows(950, 26, 60)
             + '<path d="M982 268 L982 82 L1006 70 L1006 268 Z" fill="url(#ov-glassA)" stroke="#4ba886" stroke-width="0.6"/>' + _windows(982, 24, 82, 10))
    # ADNOC HQ (tapered tower)
    s.append('<path d="M1018 268 L1021 80 Q1031 62 1041 80 L1044 268 Z" fill="url(#ov-glassB)" stroke="#4cb08b" stroke-width="0.6"/>' + _windows(1019, 24, 82, 8, "#4ba886", 0.25))
    for x, w, h, g in [(1052, 24, 110, "D"), (1080, 18, 128, "A"), (1102, 26, 92, "D"), (1132, 16, 104, "B")]:
        s.append(f'<rect x="{x}" y="{B - h}" width="{w}" height="{h}" fill="url(#ov-glass{g})" stroke="#54b692" stroke-width="0.6"/>' + _windows(x, w, B - h, 10))
    # AUH air-traffic control tower
    s.append('<g transform="translate(0 34)"><path d="M1152 234 L1158 200 L1177 200 L1183 234 Z" fill="#b5e2d1" stroke="#6fc2a3" stroke-width="0.6"/>'
             '<rect x="1163" y="116" width="9" height="86" fill="#bee6d7" stroke="#6fc2a3" stroke-width="0.6"/>'
             '<path d="M1146 112 Q1167 92 1190 104 L1186 118 Q1167 110 1150 122 Z" fill="url(#ov-glassC)" stroke="#449b7b" stroke-width="0.6"/>'
             '<rect x="1153" y="103" width="26" height="7" rx="3" fill="#40886d"/>'
             '<line x1="1167" y1="96" x2="1167" y2="84" stroke="#4fb28e" stroke-width="1.4"/><circle cx="1167" cy="83" r="2" fill="#1f2937"/></g>')
    return "".join(s)


def _plane() -> str:
    return ('<g transform="scale(1.15)">'
            '<path d="M-36 -2 L-46 -21 L-38 -21 L-27 -4 Z" fill="#059669"/>'
            '<path d="M-2 -3 L-16 -16 L-10 -16 L8 -4 Z" fill="#bcead9"/>'
            '<path d="M-40 -2 C-30 -7 20 -7 34 -4 C42 -2 44 2 34 4 C20 7 -30 7 -40 2 Z" fill="#ffffff" stroke="#93d0ba" stroke-width="0.8"/>'
            '<path d="M28 -3.6 C33 -3.6 37 -2 38.4 0 L30 0 Z" fill="#3a7b63"/>'
            '<line x1="-24" y1="-1.6" x2="22" y2="-1.6" stroke="#6fc2a3" stroke-width="1.4" stroke-dasharray="2 2.2"/>'
            '<path d="M-36 1 L-48 8 L-41 8 L-30 2 Z" fill="#aedfcd"/>'
            '<path d="M-4 2 L-22 22 L-14 22 L10 3 Z" fill="#e1f8ef" stroke="#93d0ba" stroke-width="0.6"/>'
            '<ellipse cx="-8" cy="9" rx="7" ry="3" fill="#a6dac7"/></g>')


def hero_svg() -> str:
    rng = random.Random(7)  # deterministic skyline haze
    fx, fy, fa = _point_at_length(0.62)
    trail = (f'<path d="{FLIGHT_PATH}" pathLength="1000" fill="none" stroke="url(#ov-trail)" stroke-width="11" stroke-linecap="round" opacity="0.28" filter="url(#ov-glow)" stroke-dasharray="1000" {{dash}}/>'
             f'<path d="{FLIGHT_PATH}" pathLength="1000" fill="none" stroke="url(#ov-trail)" stroke-width="2.6" stroke-linecap="round" stroke-dasharray="1000" {{dash}}/>')
    anim_dash = 'stroke-dashoffset="1000"'
    dash_anim = ('<animate attributeName="stroke-dashoffset" values="1000;0;0" keyTimes="0;0.78;1" calcMode="spline" '
                 'keySplines="0.42 0 0.58 1;0 0 1 1" dur="14s" repeatCount="indefinite"/>')
    animated_trail = trail.replace("{dash}", anim_dash).replace('/>', '>' + dash_anim + '</path>')
    static_trail = trail.replace("{dash}", 'stroke-dashoffset="380"')
    return f"""<svg class="ov-hero-art" viewBox="0 0 1200 320" preserveAspectRatio="xMaxYMax slice" role="img" aria-label="Illustrated Abu Dhabi skyline with an aircraft flying overhead">
<defs>
<linearGradient id="ov-glassA" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#a4e3cc"/><stop offset="1" stop-color="#48ac87"/></linearGradient>
<linearGradient id="ov-glassB" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c9f0e2"/><stop offset="1" stop-color="#60c19e"/></linearGradient>
<linearGradient id="ov-glassC" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6cceaa"/><stop offset="1" stop-color="#3e7d66"/></linearGradient>
<linearGradient id="ov-glassD" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d9f3ea"/><stop offset="1" stop-color="#8dd3b9"/></linearGradient>
<linearGradient id="ov-water" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c0eddc"/><stop offset="1" stop-color="#a1dec8"/></linearGradient>
<radialGradient id="ov-sun" cx="930" cy="130" r="320" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#fff6e6" stop-opacity="0.95"/><stop offset="0.45" stop-color="#e2fcf3" stop-opacity="0.35"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/></radialGradient>
<linearGradient id="ov-trail" gradientUnits="userSpaceOnUse" x1="560" y1="206" x2="1072" y2="70"><stop offset="0" stop-color="#17cf8b" stop-opacity="0"/><stop offset="0.35" stop-color="#17cf8b" stop-opacity="0.95"/><stop offset="1" stop-color="#16c383" stop-opacity="0.95"/></linearGradient>
<filter id="ov-glow" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="3"/></filter>
<clipPath id="ov-waterclip"><rect x="0" y="{B}" width="1200" height="60"/></clipPath>
</defs>
<rect x="0" y="0" width="1200" height="320" fill="url(#ov-sun)"/>
{_haze(rng, 470, 1200, 40, 118, "#d4f0e5", 0.9)}
{_haze(rng, 520, 1200, 22, 70, "#bee6d7", 0.9)}
<g id="ov-fg">{_skyline()}</g>
<rect x="0" y="{B}" width="1200" height="60" fill="url(#ov-water)"/>
<g clip-path="url(#ov-waterclip)" opacity="0.17"><use href="#ov-fg" transform="translate(0 {2 * B}) scale(1 -1)"/></g>
<g stroke="#ffffff" stroke-linecap="round" opacity="0.7">{''.join(f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y}" stroke-width="1.4"/>' for x, y, w in [(600, 280, 40), (760, 288, 60), (930, 279, 34), (1050, 294, 70), (690, 302, 46), (1140, 284, 30)])}</g>
<line x1="0" y1="{B}" x2="1200" y2="{B}" stroke="#9fd5c1" stroke-width="1"/>
<g class="ov-flight-anim">
<animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;0.06;0.72;0.8;1" dur="14s" repeatCount="indefinite"/>
{animated_trail}
<g>{_plane()}<animateMotion path="{FLIGHT_PATH}" rotate="auto" keyPoints="0;1;1" keyTimes="0;0.78;1" calcMode="spline" keySplines="0.42 0 0.58 1;0 0 1 1" dur="14s" repeatCount="indefinite"/></g>
</g>
<g class="ov-flight-static">{static_trail}<g transform="translate({fx:.1f} {fy:.1f}) rotate({fa:.1f})">{_plane()}</g></g>
</svg>"""
