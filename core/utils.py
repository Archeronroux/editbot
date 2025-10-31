import os
import re
import io
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import piexif
import cv2
from cfg import cfg

def ensure_dirs(p: str):
    os.makedirs(p, exist_ok=True)

def parse_target_number(text: str) -> Optional[int]:
    # Ambil digit dari caption
    m = re.findall(r"\d+", text)
    if not m:
        return None
    try:
        return int("".join(m))
    except ValueError:
        return None

def normalize_number_for_mode(caption: str, mode: str) -> str:
    n = parse_target_number(caption) or 0
    # WhatsApp Indonesia cenderung pakai '.' sebagai thousand separator
    s = f"{n:,}".replace(",", ".")
    return s

def load_image_with_exif(path: str):
    # Baca bytes untuk EXIF
    with open(path, "rb") as f:
        data = f.read()
    exif = {}
    fmt = "JPEG"
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format or "JPEG"
        if "exif" in img.info:
            exif = piexif.load(img.info["exif"])
        img = img.convert("RGB")
        # to BGR for cv2
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return bgr, exif, fmt
    except Exception:
        # fallback via cv2
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        return bgr, exif, fmt

def save_with_exif(img_bgr, out_path: str, exif_dict, quality=95):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    ensure_dirs(os.path.dirname(out_path))
    if exif_dict:
        exif_bytes = piexif.dump(exif_dict)
        pil.save(out_path, "JPEG", quality=quality, subsampling=0, exif=exif_bytes)
    else:
        pil.save(out_path, "JPEG", quality=quality, subsampling=0)
