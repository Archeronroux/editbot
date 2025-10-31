import os
import re
import io
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import piexif
import cv2
from cfg import cfg

def ensure_dirs(p: str):
    os.makedirs(p, exist_ok=True)

def parse_target_number(text: str) -> Optional[int]:
    m = re.findall(r"\d+", text or "")
    if not m:
        return None
    try:
        return int("".join(m))
    except ValueError:
        return None

def normalize_number_for_mode(caption: str, mode: str) -> str:
    n = parse_target_number(caption) or 0
    s = f"{n:,}".replace(",", ".")
    return s

def load_image_with_exif(path: str):
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
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return bgr, exif, fmt
    except Exception:
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

# ===== Helpers untuk template path (mendukung flat pack dan structured) =====

def tpl_root(mode: str, theme: str) -> str:
    return os.path.join(cfg["templates_dir"], mode, theme)

def resolve_digit_path(mode: str, theme: str, name: str) -> Optional[str]:
    cand = [
        os.path.join(tpl_root(mode, theme), "digits", name),
        os.path.join(tpl_root(mode, theme), name),
    ]
    # fallback ke light pack
    if theme != "light":
        cand += [
            os.path.join(tpl_root(mode, "light"), "digits", name),
            os.path.join(tpl_root(mode, "light"), name),
        ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None

def resolve_anchor_path(mode: str, theme: str, name: str) -> Optional[str]:
    cand = [
        os.path.join(tpl_root(mode, theme), "anchors", name),
        os.path.join(tpl_root(mode, theme), name),
    ]
    if theme != "light":
        cand += [
            os.path.join(tpl_root(mode, "light"), "anchors", name),
            os.path.join(tpl_root(mode, "light"), name),
        ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None
