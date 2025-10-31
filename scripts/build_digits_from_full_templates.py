#!/usr/bin/env python3
import argparse, os, re, sys
from typing import List, Tuple
import cv2
import numpy as np

# Agar bisa import modul dari root repo saat skrip dijalankan dari mana pun
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cfg import cfg
from core.detect import locate_number_roi, detect_theme_hint
from core.templates import ensure_dot_template
from core.utils import ensure_dirs

NUM_RE = re.compile(r"(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)

def list_numeric_screenshots(folder: str) -> List[str]:
    files = []
    for n in os.listdir(folder):
        if n.startswith("."): 
            continue
        low = n.lower()
        if not low.endswith((".png",".jpg",".jpeg")): 
            continue
        if NUM_RE.search(low):
            files.append(os.path.join(folder, n))
    return sorted(files)

def _estimate_bg_color(bgr):
    h, w = bgr.shape[:2]
    m = max(2, min(h,w)//10)
    edges = np.vstack([
        bgr[0:m, :, :].reshape(-1,3),
        bgr[-m:, :, :].reshape(-1,3),
        bgr[:, 0:m, :].reshape(-1,3),
        bgr[:, -m:, :].reshape(-1,3),
    ])
    return np.median(edges, axis=0).astype(np.uint8)

def _alpha_from_bg(bgr, t0=8.0, t1=25.0):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = _estimate_bg_color(bgr)
    bg_lab = cv2.cvtColor(bg.reshape(1,1,3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3,)
    d = np.sqrt(np.sum((lab - bg_lab.reshape(1,1,3))**2, axis=2))
    a = np.clip((d - t0) / max(1e-6, (t1 - t0)), 0.0, 1.0)
    alpha = (a * 255.0).astype(np.uint8)
    alpha = cv2.medianBlur(alpha, 3)
    _, hard = cv2.threshold(alpha, 210, 255, cv2.THRESH_BINARY)
    alpha = np.maximum(alpha, hard)
    return alpha

def _trim_rgba(rgba):
    a = rgba[:, :, 3]
    ys = np.where(a.any(axis=1))[0]
    xs = np.where(a.any(axis=0))[0]
    if ys.size and xs.size:
        return rgba[ys[0]:ys[-1]+1, xs[0]:xs[-1]+1]
    return rgba

def _split_digits(roi_bgr) -> List[np.ndarray]:
    """
    Pecah ROI menjadi potongan per-digit pakai komponen terhubung pada mask alpha.
    """
    alpha = _alpha_from_bg(roi_bgr)
    # Bersihkan noise kecil
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    # Cari kontur
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    # Urutkan dari kiri ke kanan
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes.sort(key=lambda b: b[0])

    parts = []
    for (x,y,w,h) in boxes:
        # Saring area yang terlalu kecil/tinggi
        if w*h < 50:
            continue
        pad = max(1, int(0.15 * max(w,h)))
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(alpha.shape[1], x+w+pad); y1 = min(alpha.shape[0], y+h+pad)
        crop_bgr = roi_bgr[y0:y1, x0:x1]
        crop_a   = alpha[y0:y1, x0:x1]
        rgba = np.dstack([crop_bgr, crop_a])
        rgba = _trim_rgba(rgba)
        parts.append(rgba)
    return parts

def main():
    ap = argparse.ArgumentParser(description="Build digit (0-9) PNG RGBA from full screenshots (0..10).")
    ap.add_argument("--src", required=True, help="Folder sumber, mis. /root/editctc/android/light")
    ap.add_argument("--mode", default="android", choices=["android","iphone"])
    ap.add_argument("--theme", default="light", choices=["light","dark"])
    ap.add_argument("--out", help="Folder output digit; default ke folder src.")
    args = ap.parse_args()

    src = args.src
    out = args.out or args.src
    ensure_dirs(out)

    files = list_numeric_screenshots(src)
    if not files:
        print("Tidak menemukan file 0.png..10.png di folder sumber", file=sys.stderr)
        sys.exit(1)

    collected = {}
    made = 0

    for f in files:
        m = NUM_RE.search(os.path.basename(f).lower())
        if not m:
            continue
        num_str = m.group(1)  # "0".."10"
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip (tidak bisa dibaca): {f}", file=sys.stderr)
            continue

        theme_hint = detect_theme_hint(img)
        (rx, ry, rw, rh), theme_used = locate_number_roi(img, args.mode, theme_hint)
        roi = img[ry:ry+rh, rx:rx+rw]
        if roi.size == 0:
            print(f"ROI kosong untuk {f}", file=sys.stderr)
            continue

        parts = _split_digits(roi)
        # Pemetaan label dari file num_str
        labels = list(num_str)
        if len(parts) != len(labels):
            # jika mismatch, coba susun kasar: satu digit saja
            if len(labels) == 1 and len(parts) >= 1:
                parts = [max(parts, key=lambda a: a.shape[1]*a.shape[0])]
            else:
                print(f"Peringatan: jumlah komponen {len(parts)} != digit {len(labels)} pada {f}", file=sys.stderr)

        for i, rgba in enumerate(parts[:len(labels)]):
            d = labels[i]
            save_path = os.path.join(out, f"{d}.png")
            cv2.imwrite(save_path, rgba)
            collected[d] = save_path
            print(f"Buat {save_path}")
            made += 1

        # Sekaligus pastikan dot.png jika tersedia di ROI
        try:
            from core.templates import ensure_dot_template
            ensure_dot_template(args.mode, theme_used, roi)
        except Exception as e:
            print(f"Gagal membuat dot dari {f}: {e}", file=sys.stderr)

    # Validasi minimum
    missing = [str(i) for i in range(10) if str(i) not in collected]
    if missing:
        print(f"Digit yang belum terambil: {missing}. Anda bisa menambah beberapa file contoh lagi di {src} (mis. 2.png, 3.png, dst).", file=sys.stderr)

    print(f"Selesai. Digit dibuat/ditulis: {made}. Output: {out}")

if __name__ == "__main__":
    main()
