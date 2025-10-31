import os
from typing import Optional, Tuple
import cv2
import numpy as np

from cfg import cfg
from core.utils import resolve_digit_path, tpl_root, ensure_dirs

def ensure_dot_template(mode: str, theme: str, roi_bgr: np.ndarray) -> str:
    """
    Pastikan dot.png tersedia. Jika belum ada, coba ekstrak dari ROI.
    Jika tetap gagal, buat dot sintetis berdasarkan karakteristik teks di ROI.
    Return: path dot.png (dipastikan ada ketika fungsi selesai).
    """
    # Sudah ada?
    existing = resolve_digit_path(mode, theme, "dot.png")
    if existing:
        return existing

    # Tentukan direktori tempat menyimpan (sejajar dengan 0.png)
    zero_path = resolve_digit_path(mode, theme, "0.png")
    if not zero_path:
        # kalau 0.png pun belum ada, fallback ke root theme
        save_dir = tpl_root(mode, theme)
    else:
        save_dir = os.path.dirname(zero_path)
    ensure_dirs(save_dir)
    out_path = os.path.join(save_dir, "dot.png")

    # 1) Coba ekstraksi dari ROI
    dot_rgba = _extract_dot_from_roi(roi_bgr)
    if dot_rgba is None:
        # 2) Fallback: buat dot sintetis
        dot_rgba = _synthesize_dot_from_roi(roi_bgr)

    # Simpan
    cv2.imwrite(out_path, dot_rgba)
    return out_path

def _extract_dot_from_roi(roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Cari kandidat titik pemisah ribuan di ROI:
    - Threshold Otsu (teks biasanya lebih gelap/berwarna dari background)
    - Connected components -> pilih komponen kecil, relatif bulat, di tengah vertikal
    Return RGBA kecil; None jika gagal.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # Ambil mask teks
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Cari kontur kecil yang bulat
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape[:2]
    area_total = H * W
    best = None
    best_score = -1.0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 0.002 * area_total or area > 0.05 * area_total:
            continue
        ar = w / max(1, h)
        if ar < 0.6 or ar > 1.4:
            continue
        # circularity
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))
        if circularity < 0.5:
            continue
        # posisi kira-kira di tengah vertikal baris angka
        cy = y + h / 2
        pos_score = 1.0 - abs((cy / H) - 0.5)  # maksimum jika di tengah
        score = circularity * 0.6 + pos_score * 0.4
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    pad = max(1, int(0.2 * max(w, h)))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    crop_bgr = roi_bgr[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    # Buat RGBA dari mask
    rgba = np.dstack([
        crop_bgr[:, :, 2],  # R
        crop_bgr[:, :, 1],  # G
        crop_bgr[:, :, 0],  # B
        crop_mask           # A
    ])
    return rgba

def _synthesize_dot_from_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Buat dot buatan dengan ukuran proporsional terhadap tinggi ROI,
    warna diambil dari median warna teks (piksel gelap/berwarna dalam ROI).
    """
    H, W = roi_bgr.shape[:2]
    # Estimasi ukuran: ±10% tinggi ROI
    radius = max(2, int(0.10 * H))

    # Estimasi warna teks: ambil piksel gelap (nilai V rendah)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    thresh = np.percentile(V, 35)  # ambil 35% piksel tergelap
    text_mask = (V <= thresh).astype(np.uint8)
    if text_mask.sum() == 0:
        color_bgr = roi_bgr.mean(axis=(0,1)).astype(np.uint8)
    else:
        color_bgr = np.median(roi_bgr[text_mask.astype(bool)], axis=0).astype(np.uint8)

    size = radius * 4
    canvas = np.zeros((size, size, 4), dtype=np.uint8)

    # gambar lingkaran dengan alpha gradient
    center = (size // 2, size // 2)
    # channel warna
    canvas[:, :, 0] = color_bgr[0]
    canvas[:, :, 1] = color_bgr[1]
    canvas[:, :, 2] = color_bgr[2]

    # alpha mask
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    alpha = np.clip(1.0 - (dist - (radius - 1)) / max(1, radius * 0.6), 0, 1)
    alpha[dist > radius * 1.6] = 0
    alpha = (alpha * 255).astype(np.uint8)
    canvas[:, :, 3] = alpha

    # trim ke bounding box alpha
    xs = np.where(alpha.any(axis=0))[0]
    ys = np.where(alpha.any(axis=1))[0]
    x0, x1 = xs[0], xs[-1] + 1
    y0, y1 = ys[0], ys[-1] + 1
    return canvas[y0:y1, x0:x1]
