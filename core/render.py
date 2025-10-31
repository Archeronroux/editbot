import os
import cv2
import numpy as np
from typing import Tuple
from cfg import cfg
from core.utils import resolve_digit_path

def inpaint_region(img_bgr, roi: Tuple[int, int, int, int]):
    x, y, w, h = roi
    out = img_bgr.copy()
    patch = out[y:y+h, x:x+w]
    mask = np.full((h, w), 255, dtype=np.uint8)
    repaired = cv2.inpaint(patch, mask, 3, cv2.INPAINT_TELEA)
    out[y:y+h, x:x+w] = repaired
    return out

def _estimate_bg_color(bgr):
    h, w = bgr.shape[:2]
    m = max(2, min(h, w) // 10)
    edges = np.vstack([
        bgr[0:m, :, :].reshape(-1, 3),
        bgr[-m:, :, :].reshape(-1, 3),
        bgr[:, 0:m, :].reshape(-1, 3),
        bgr[:, -m:, :].reshape(-1, 3),
    ])
    return np.median(edges, axis=0).astype(np.uint8)

def _to_rgba_with_alpha(img):
    """
    Pastikan template digit RGBA. Jika hanya 3 channel (BGR),
    buat alpha otomatis dari perbedaan warna terhadap background.
    """
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        return img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        bgr = img
        bg = _estimate_bg_color(bgr)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(bg.reshape(1,1,3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3,)
        d = np.sqrt(np.sum((lab - bg_lab.reshape(1,1,3))**2, axis=2))
        # Ambang smooth untuk tepi halus
        t0, t1 = 8.0, 25.0
        a = np.clip((d - t0) / max(1e-6, (t1 - t0)), 0.0, 1.0)
        alpha = (a * 255.0).astype(np.uint8)
        alpha = cv2.medianBlur(alpha, 3)
        # Pertebal inti glyph
        _, hard = cv2.threshold(alpha, 210, 255, cv2.THRESH_BINARY)
        alpha = np.maximum(alpha, hard)
        rgba = np.dstack([bgr, alpha])
        return rgba
    # fallback: tambahkan alpha penuh
    return np.dstack([img[:, :, :3], np.full(img.shape[:2], 255, np.uint8)])

def compose_number_from_templates(text: str, mode: str, theme: str):
    images = []
    for ch in text:
        name = "dot.png" if ch == "." else f"{ch}.png"
        p = resolve_digit_path(mode, theme, name)
        if not p:
            raise RuntimeError(f"Template digit tidak ditemukan: {mode}/{theme}/{name}")
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Gagal membaca file template: {p}")
        img = _to_rgba_with_alpha(img)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            raise RuntimeError(f"Gagal mengonversi template ke RGBA: {p}")
        images.append(img)

    # Trim alpha padding & kerning ringan
    strips = []
    for img in images:
        alpha = img[:, :, 3]
        xs = np.where(alpha.any(axis=0))[0]
        ys = np.where(alpha.any(axis=1))[0]
        if xs.size and ys.size:
            img = img[ys[0]:ys[-1]+1, xs[0]:xs[-1]+1, :]
        strips.append(img)

    space = 4
    width = sum(s.shape[1] for s in strips) + space*(len(strips)-1)
    height = max(s.shape[0] for s in strips)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    x = 0
    for i, s in enumerate(strips):
        h = s.shape[0]
        y = (height - h)//2
        out[y:y+h, x:x+s.shape[1]] = s
        x += s.shape[1] + (space if i < len(strips)-1 else 0)
    return out

def render_number_into_roi(img_bgr, number_rgba, roi: Tuple[int, int, int, int]):
    x, y, w, h = roi
    scale = h / number_rgba.shape[0]
    new_w = max(1, int(number_rgba.shape[1] * scale))
    number = cv2.resize(number_rgba, (new_w, h), interpolation=cv2.INTER_AREA)
    # Align kanan
    pos_x = x + w - number.shape[1]
    pos_y = y

    out = img_bgr.copy()
    overlay_rgba(out, number, (pos_x, pos_y))
    return out

def overlay_rgba(dst_bgr, src_rgba, top_left):
    x, y = top_left
    h, w = src_rgba.shape[:2]
    h0, w0 = dst_bgr.shape[:2]
    if x < 0 or y < 0:
        return
    x2 = min(x + w, w0)
    y2 = min(y + h, h0)
    if x2 <= x or y2 <= y:
        return
    region = dst_bgr[y:y2, x:x2]
    src = src_rgba[0:(y2-y), 0:(x2-x)]
    alpha = (src[:, :, 3:4] / 255.0)
    rgb = src[:, :, :3]
    region[:] = (alpha * rgb + (1 - alpha) * region).astype(np.uint8)
