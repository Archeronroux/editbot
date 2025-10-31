import os
import cv2
import numpy as np
from typing import Tuple
from cfg import cfg
from core.utils import tpl_root, resolve_anchor_path

def detect_mode_auto(img_bgr) -> str:
    # Heuristik: bandingkan anchor_mode untuk android vs iphone (light pack)
    scores = []
    for mode in ("android", "iphone"):
        p = resolve_anchor_path(mode, "light", "anchor_mode.png")
        if not p or not os.path.exists(p):
            continue
        anchor = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        s = _best_match_score(img_bgr, anchor)
        scores.append((s, mode))
    if not scores:
        return "android"
    scores.sort(reverse=True)
    return scores[0][1]

def detect_theme(img_bgr) -> str:
    h, w = img_bgr.shape[:2]
    cx1, cy1, cx2, cy2 = int(w*0.35), int(h*0.25), int(w*0.65), int(h*0.45)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "light"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) / 255.0
    lightness = float(gray.mean())
    return "light" if lightness > cfg["theme_threshold"] else "dark"

def locate_number_roi(img_bgr, mode: str, theme: str) -> Tuple[int, int, int, int]:
    """
    Temukan area angka '… anggota' secara dinamis.
    Strategi:
    - Temukan anchor 'anggota' via template matching.
    - ROI diletakkan di kiri anchor dengan ukuran proporsional terhadap tinggi anchor,
      sehingga stabil di berbagai skala (DPI).
    """
    anchor_path = resolve_anchor_path(mode, theme, "anggota.png")
    if not anchor_path:
        # fallback kasar
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        return (int(W*0.42), int(H*0.22), int(W*0.18), int(H*0.05))

    anchor = cv2.imread(anchor_path, cv2.IMREAD_GRAYSCALE)
    x, y, w, h = _match_anchor(img_bgr, anchor)

    # Hitung ROI berbasis tinggi anchor
    H, W = img_bgr.shape[:2]
    ah = max(12, h)
    # padding kecil dari sisi kanan agar tidak menimpa kata 'anggota'
    pad_right = int(0.25 * ah)
    # lebar ROI cukup untuk sampai 5 digit + titik ribuan
    rw = int(7.5 * ah)
    # tinggi ROI sedikit lebih tinggi dari anchor
    rh = int(1.8 * ah)
    rx = x - pad_right - rw
    ry = y - int(0.35 * ah)

    # Clamp ke batas gambar
    rx = max(0, rx)
    ry = max(0, ry)
    if rx + rw > W:
        rw = W - rx
    if ry + rh > H:
        rh = H - ry

    return (rx, ry, rw, rh)

def _match_anchor(img_bgr, anchor_gray):
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
    score, (x, y, w, h) = best
    if score < cfg["match_threshold"]:
        H, W = gray.shape[:2]
        w, h = anchor_gray.shape[1], anchor_gray.shape[0]
        return (W//2 - w//2, int(H*0.35), w, h)
    return (x, y, w, h)

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
