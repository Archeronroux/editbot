import os
import cv2
import numpy as np
from typing import Tuple
from cfg import cfg

def inpaint_region(img_bgr, roi: Tuple[int, int, int, int]):
    x, y, w, h = roi
    out = img_bgr.copy()
    patch = out[y:y+h, x:x+w]
    mask = np.full((h, w), 255, dtype=np.uint8)
    # Inpaint cepat
    repaired = cv2.inpaint(patch, mask, 3, cv2.INPAINT_TELEA)
    out[y:y+h, x:x+w] = repaired
    return out

def compose_number_from_templates(text: str, mode: str, theme: str):
    # text termasuk separator "." bila perlu
    folder = os.path.join(cfg["templates_dir"], mode, theme, "digits")
    images = []
    for ch in text:
        name = "dot.png" if ch == "." else f"{ch}.png"
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            # fallback ke light
            p = os.path.join(cfg["templates_dir"], mode, "light", "digits", name)
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # RGBA
        if img is None:
            raise RuntimeError(f"Template digit tidak ditemukan: {p}")
        images.append(img)

    # Gabung horizontal dengan kerning ringan berdasar alpha bounding box
    strips = []
    for img in images:
        # trim alpha padding
        alpha = img[:, :, 3]
        xs = np.where(alpha.any(axis=0))[0]
        if xs.size == 0:
            strips.append(img)
        else:
            img = img[:, xs[0]:xs[-1]+1, :]
            strips.append(img)

    # sisipkan jarak kecil antar digit
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
    # Resize jumlah digit ke tinggi ROI dengan aspect ratio
    scale = h / number_rgba.shape[0]
    new_w = max(1, int(number_rgba.shape[1]*scale))
    number = cv2.resize(number_rgba, (new_w, h), interpolation=cv2.INTER_AREA)

    # Align ke kanan (umum di UI)
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
    x2 = min(x+w, w0)
    y2 = min(y+h, h0)
    if x2 <= x or y2 <= y:
        return
    region = dst_bgr[y:y2, x:x2]
    src = src_rgba[0:(y2-y), 0:(x2-x)]
    alpha = (src[:, :, 3:4] / 255.0)
    rgb = src[:, :, :3]
    region[:] = (alpha * rgb + (1 - alpha) * region).astype(np.uint8)
