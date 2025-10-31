import os
import cv2
import numpy as np
from typing import Tuple, Optional
from cfg import cfg
from core.utils import resolve_anchor_path

def detect_mode_auto(img_bgr) -> str:
    # Heuristik: bandingkan anchor_mode untuk android vs iphone (pakai light sbg patokan)
    scores = []
    for mode in ("android", "iphone"):
        p = resolve_anchor_path(mode, "light", "anchor_mode")
        if not p or not os.path.exists(p):
            continue
        anchor = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        s = _best_match_score(img_bgr, anchor)
        scores.append((s, mode))
    if not scores:
        return "android"
    scores.sort(reverse=True)
    return scores[0][1]

def detect_theme_hint(img_bgr) -> str:
    h, w = img_bgr.shape[:2]
    cx1, cy1, cx2, cy2 = int(w*0.35), int(h*0.25), int(w*0.65), int(h*0.45)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "light"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) / 255.0
    return "light" if float(gray.mean()) > cfg["theme_threshold"] else "dark"

def locate_number_roi(img_bgr, mode: str, theme_hint: Optional[str] = None):
    """
    Cari anchor 'anggota' pada kedua tema (light & dark), pilih yang skornya terbaik.
    Kembalikan ((rx, ry, rw, rh), theme_used)
    """
    themes_try = ["light", "dark"]
    if theme_hint in ("light", "dark"):
        # prioritaskan hint terlebih dulu
        themes_try = [theme_hint] + [t for t in themes_try if t != theme_hint]

    best = None  # (score, x, y, w, h, theme)
    for th in themes_try:
        anchor_path = resolve_anchor_path(mode, th, "anggota")
        if not anchor_path:
            continue
        anchor = cv2.imread(anchor_path, cv2.IMREAD_GRAYSCALE)
        s, (x, y, w, h) = _match_anchor_with_score(img_bgr, anchor)
        if best is None or s > best[0]:
            best = (s, x, y, w, h, th)

    if best is None or best[0] < max(0.6, cfg["match_threshold"] - 0.1):
        # fallback kasar jika anchor tidak ketemu sama sekali
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        rx = int(W*0.42); ry = int(H*0.22); rw = int(W*0.20); rh = int(H*0.06)
        return (rx, ry, rw, rh), (theme_hint or "light")

    _, x, y, w, h, theme_used = best

    # ROI proporsional terhadap tinggi anchor
    H, W = img_bgr.shape[:2]
    ah = max(12, h)
    pad_right = int(0.28 * ah)     # jarak dari kata 'anggota'
    rw = int(8.2 * ah)             # lebar cukup untuk 5 digit + titik
    rh = int(1.9 * ah)             # tinggi sedikit > anchor
    rx = x - pad_right - rw        # di kiri anchor
    ry = y - int(0.38 * ah)

    rx = max(0, rx); ry = max(0, ry)
    if rx + rw > W: rw = W - rx
    if ry + rh > H: rh = H - ry

    return (rx, ry, rw, rh), theme_used

def _match_anchor_with_score(img_bgr, anchor_gray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    best = (-1.0, (0, 0, anchor_gray.shape[1], anchor_gray.shape[0]))
    for s in cfg["match_scales"]:
        ah = int(anchor_gray.shape[0]*s)
        aw = int(anchor_gray.shape[1]*s)
        if ah < 8 or aw < 8:
            continue
        a = cv2.resize(anchor_gray, (aw, ah), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray, a, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best[0]:
            best = (max_val, (max_loc[0], max_loc[1], aw, ah))
    return best

def _best_match_score(img_bgr, anchor_gray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    best = -1.0
    for s in cfg["match_scales"]:
        ah = int(anchor_gray.shape[0]*s)
        aw = int(anchor_gray.shape[1]*s)
        if ah < 8 or aw < 8:
            continue
        a = cv2.resize(anchor_gray, (aw, ah), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray, a, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(res.max()))
    return best
