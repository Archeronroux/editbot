#!/usr/bin/env python3
"""
Bangun digit pack (0–9) PNG transparan dari kumpulan screenshot penuh
yang dinamai sesuai angka target (mis. 0.png, 1.png, ... 10.png).

Perbaikan utama:
- Pakai OCR (pytesseract) untuk menemukan kata "anggota" agar tidak
  salah match ke judul grup.
- Batasi pencarian pada band vertikal baris "Grup · X anggota".
- ROI di kiri "anggota" proporsional terhadap tinggi anchor (stabil pada DPI).
- Segmentasi per-digit, filter bullet "·" otomatis.
- Otomatis membuat dot.png dari ROI jika belum ada.

Persiapan:
- apt-get install -y tesseract-ocr tesseract-ocr-ind
- pip install pytesseract opencv-python-headless numpy

Contoh:
  python3 scripts/build_digits_from_full_templates.py \
      --src /root/editctc/android/light --mode android --theme light
"""

import argparse
import os
import re
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output as TessOutput
    HAS_TESS = True
except Exception:
    HAS_TESS = False

# agar bisa import modul dari root repo
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.utils import ensure_dirs
from core.templates import ensure_dot_template

NUM_RE = re.compile(r"(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)

def list_numeric_screenshots(folder: str) -> List[str]:
    out = []
    for n in os.listdir(folder):
        if n.startswith("."):
            continue
        low = n.lower()
        if low.endswith((".png", ".jpg", ".jpeg")) and NUM_RE.search(low):
            out.append(os.path.join(folder, n))
    return sorted(out)

def _estimate_bg_color(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(2, min(h, w) // 10)
    edges = np.vstack([
        bgr[0:m, :, :].reshape(-1, 3),
        bgr[-m:, :, :].reshape(-1, 3),
        bgr[:, 0:m, :].reshape(-1, 3),
        bgr[:, -m:, :].reshape(-1, 3),
    ])
    return np.median(edges, axis=0).astype(np.uint8)

def _alpha_from_bg(bgr: np.ndarray, t0=8.0, t1=25.0) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = _estimate_bg_color(bgr)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3,)
    d = np.sqrt(np.sum((lab - bg_lab.reshape(1, 1, 3)) ** 2, axis=2))
    a = np.clip((d - t0) / max(1e-6, (t1 - t0)), 0.0, 1.0)
    alpha = (a * 255.0).astype(np.uint8)
    alpha = cv2.medianBlur(alpha, 3)
    _, hard = cv2.threshold(alpha, 210, 255, cv2.THRESH_BINARY)
    alpha = np.maximum(alpha, hard)
    return alpha

def _trim_rgba(rgba: np.ndarray) -> np.ndarray:
    a = rgba[:, :, 3]
    ys = np.where(a.any(axis=1))[0]
    xs = np.where(a.any(axis=0))[0]
    if ys.size and xs.size:
        return rgba[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    return rgba

def detect_theme_hint(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    cx1, cy1, cx2, cy2 = int(w * 0.35), int(h * 0.25), int(w * 0.65), int(h * 0.45)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "light"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) / 255.0
    return "light" if float(gray.mean()) > 0.6 else "dark"

def _find_anggota_bbox_ocr(img_bgr: np.ndarray, mode: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Cari bbox kata 'anggota' dengan OCR, dibatasi ke band vertikal baris informasi grup.
    Return (x, y, w, h) dalam koordinat global atau None jika gagal.
    """
    if not HAS_TESS:
        return None

    H, W = img_bgr.shape[:2]
    # Band vertikal: mencegah OCR membaca judul grup
    if mode == "android":
        y0, y1 = int(0.30 * H), int(0.52 * H)
    else:  # iphone
        y0, y1 = int(0.24 * H), int(0.48 * H)

    y0 = max(0, min(H-1, y0)); y1 = max(y0+8, min(H, y1))
    band = img_bgr[y0:y1, :]

    # OCR
    try:
        data = pytesseract.image_to_data(
            cv2.cvtColor(band, cv2.COLOR_BGR2RGB),
            lang="ind+eng",
            output_type=TessOutput.DICT
        )
    except Exception:
        # fallback bahasa Inggris saja
        data = pytesseract.image_to_data(
            cv2.cvtColor(band, cv2.COLOR_BGR2RGB),
            lang="eng",
            output_type=TessOutput.DICT
        )

    best = None  # (conf, x, y, w, h)
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip().lower()
        if txt != "anggota":
            continue
        conf = float(data.get("conf", ["-1"])[i])
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        # validasi minimal ukuran
        if w < 10 or h < 8:
            continue
        if best is None or conf > best[0]:
            best = (conf, x, y, w, h)

    if best is None:
        return None

    _, bx, by, bw, bh = best
    # Offset y ke koordinat global
    return (bx, by + y0, bw, bh)

def _split_digits_from_roi(roi_bgr: np.ndarray, ah: int) -> List[np.ndarray]:
    """
    Pecah ROI menjadi per-digit.
    Filter out bullet '·' (umumnya kecil dan hampir bulat) berdasarkan ukuran relatif terhadap ah.
    """
    alpha = _alpha_from_bg(roi_bgr)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[0])

    parts = []
    for (x, y, w, h) in boxes:
        area = w * h
        # heuristik: buang bullet '·' yang jauh lebih kecil dari tinggi anchor
        # dianggap bullet bila tinggi < 0.55*ah dan lebar < 0.55*ah
        if h < 0.55 * ah and w < 0.55 * ah:
            continue
        pad = max(1, int(0.15 * max(w, h)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(alpha.shape[1], x + w + pad); y1 = min(alpha.shape[0], y + h + pad)
        crop_bgr = roi_bgr[y0:y1, x0:x1]
        crop_a   = alpha[y0:y1, x0:x1]
        rgba = np.dstack([crop_bgr, crop_a])
        rgba = _trim_rgba(rgba)
        parts.append(rgba)
    return parts

def build_from_folder(src: str, mode: str, theme: str, out_dir: Optional[str] = None):
    files = list_numeric_screenshots(src)
    if not files:
        print(f"Tidak ada file bernama angka di {src} (mis. 0.png, 1.jpg, ...).", file=sys.stderr)
        sys.exit(1)

    out = out_dir or src
    ensure_dirs(out)

    made = 0
    collected = {}

    for f in files:
        m = NUM_RE.search(os.path.basename(f).lower())
        if not m:
            continue
        label = m.group(1)  # "0".."10" (digunakan sebagai ground truth)
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip (gagal baca): {f}", file=sys.stderr)
            continue

        # 1) Cari bbox 'anggota' via OCR
        bbox = _find_anggota_bbox_ocr(img, mode)
        if bbox is None:
            print(f"Gagal menemukan kata 'anggota' via OCR pada: {f}. Pastikan tesseract-ocr dan tesseract-ocr-ind terinstal.", file=sys.stderr)
            continue
        ax, ay, aw, ah = bbox

        # 2) ROI di kiri kata 'anggota' pada baseline yang sama
        # padding kanan agar tidak menimpa huruf 'a' pertama
        pad_right = int(0.28 * ah)
        # lebar ROI: cukup untuk 5 digit (ribuan) + spasi. konservatif.
        rw = int(8.5 * ah)
        rh = int(1.9 * ah)
        rx = ax - pad_right - rw
        ry = ay - int(0.38 * ah)

        H, W = img.shape[:2]
        rx = max(0, rx); ry = max(0, ry)
        if rx + rw > W: rw = W - rx
        if ry + rh > H: rh = H - ry

        roi = img[ry:ry+rh, rx:rx+rw]
        if roi.size == 0:
            print(f"ROI kosong di {f} (periksa OCR/posisi).", file=sys.stderr)
            continue

        # 3) Pecah ROI jadi digit; filter bullet
        parts = _split_digits_from_roi(roi, ah)

        # Jika label hanya satu digit, pilih komponen terbesar (menghindari bullet)
        if len(label) == 1:
            if not parts:
                print(f"Tidak ada komponen digit terdeteksi pada {f}", file=sys.stderr)
                continue
            part = max(parts, key=lambda a: a.shape[0] * a.shape[1])
            save_path = os.path.join(out, f"{label}.png")
            cv2.imwrite(save_path, part)
            made += 1
            collected[label] = save_path
            print(f"Buat {save_path}")
        else:
            # multi-digit (mis. 10.png): cocokkan jumlah komponen; jika mismatch, ambil sebanyak mungkin dari kiri
            if not parts:
                print(f"Tidak ada komponen digit pada {f}", file=sys.stderr)
                continue
            want = list(label)
            take = min(len(want), len(parts))
            for i in range(take):
                d = want[i]
                save_path = os.path.join(out, f"{d}.png")
                cv2.imwrite(save_path, parts[i])
                collected[d] = save_path
                made += 1
                print(f"Buat {save_path}")

        # 4) Pastikan dot.png tersedia (ambil dari ROI bila ada)
        try:
            ensure_dot_template(mode, theme, roi)
        except Exception as e:
            print(f"Gagal membuat dot dari {f}: {e}", file=sys.stderr)

    # Validasi minimum
    missing = [str(i) for i in range(10) if str(i) not in collected]
    if missing:
        print(f"Digit yang belum terbangun: {missing}. Tambahkan beberapa contoh lagi di {src} (mis. 2.png, 3.png).", file=sys.stderr)
    print(f"Selesai. Digit dibuat/ditulis: {made}. Output: {out}")

def main():
    ap = argparse.ArgumentParser(description="Build digit (0–9) RGBA from full screenshots using OCR on 'anggota'.")
    ap.add_argument("--src", required=True, help="Folder sumber (berisi 0.png..10.png screenshot penuh)")
    ap.add_argument("--mode", default="android", choices=["android", "iphone"])
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--out", help="Folder output digit (default: sama dengan --src)")
    args = ap.parse_args()

    if not HAS_TESS:
        print("pytesseract tidak terpasang. Jalankan: pip install pytesseract && apt-get install -y tesseract-ocr tesseract-ocr-ind", file=sys.stderr)
        sys.exit(1)

    build_from_folder(args.src, args.mode, args.theme, args.out)

if __name__ == "__main__":
    main()
