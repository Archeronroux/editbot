import os
import cv2
import numpy as np
from typing import Tuple
from cfg import cfg

def _templates_dir(mode: str, theme: str) -> str:
    return os.path.join(cfg["templates_dir"], mode, theme)

def detect_mode_auto(img_bgr) -> str:
    # Heuristik ringan: match anchor khusus mode
    anchors = [
        ("android", "anchors/anchor_mode.png"),
        ("iphone",  "anchors/anchor_mode.png"),
    ]
    scores = []
    for mode, rel in anchors:
        p = os.path.join(cfg["templates_dir"], mode, "light", rel)  # light sebagai patokan
        if not os.path.exists(p):
            continue
        s = _best_match_score(img_bgr, cv2.imread(p, cv2.IMREAD_GRAYSCALE))
        scores.append((s, mode))
    if not scores:
        return "android"
    scores.sort(reverse=True)
    return scores[0][1]

def detect_theme(img_bgr) -> str:
    # Luminance kasar di area tengah
    h, w = img_bgr.shape[:2]
    cx1, cy1, cx2, cy2 = int(w*0.35), int(h*0.25), int(w*0.65), int(h*0.45)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "light"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) / 255.0
    lightness = float(gray.mean())
    return "light" if lightness > cfg["theme_threshold"] else "dark"

def locate_number_roi(img_bgr, mode: str, theme: str) -> Tuple[int, int, int, int]:
    # Cari posisi anchor "anggota"
    anchor_path = os.path.join(_templates_dir(mode, theme), "anchors", "anggota.png")
    if not os.path.exists(anchor_path):
        # fallback ke light
        anchor_path = os.path.join(_templates_dir(mode, "light"), "anchors", "anggota.png")
    anchor = cv2.imread(anchor_path, cv2.IMREAD_GRAYSCALE)
    x, y, w, h = _match_anchor(img_bgr, anchor)

    # Hitung ROI angka berdasarkan offset per-mode
    off = cfg["roi"][mode]
    rx = max(0, x + off["dx"])
    ry = max(0, y + off["dy"])
    rw = off["w"]
    rh = off["h"]

    # Clamp
    H, W = img_bgr.shape[:2]
    rx = min(max(0, rx), W-1)
    ry = min(max(0, ry), H-1)
    rw = min(rw, W - rx)
    rh = min(rh, H - ry)
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
        # jika gagal, ambil posisi kira-kira tengah sebagai fallback
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
