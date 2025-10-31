import os
import uuid
import cv2
import numpy as np
from typing import Tuple, Dict

from cfg import cfg
from core.detect import detect_mode_auto, detect_theme, locate_number_roi
from core.render import inpaint_region, render_number_into_roi, compose_number_from_templates
from core.utils import load_image_with_exif, save_with_exif, normalize_number_for_mode, ensure_dirs
from core.templates import ensure_dot_template  # <- tambahan

async def process_image_pipeline(file_path: str, mode: str, user_caption: str, user_id: int) -> Tuple[str, Dict]:
    img_bgr, exif, fmt = load_image_with_exif(file_path)

    final_mode = mode if mode != "all" else detect_mode_auto(img_bgr)
    theme = detect_theme(img_bgr)

    target_num_text = normalize_number_for_mode(user_caption, final_mode)

    roi = locate_number_roi(img_bgr, final_mode, theme)
    x, y, w, h = roi
    roi_crop = img_bgr[y:y+h, x:x+w]

    # Pastikan dot.png ada (auto-learn dari ROI jika belum ada)
    ensure_dot_template(final_mode, theme, roi_crop)

    base = inpaint_region(img_bgr, roi)
    composed = compose_number_from_templates(target_num_text, final_mode, theme)
    out_img = render_number_into_roi(base, composed, roi)

    ensure_dirs(cfg["output_dir"])
    out_path = os.path.join(cfg["output_dir"], f"{user_id}_{uuid.uuid4().hex}.jpg")
    save_with_exif(out_img, out_path, exif, quality=cfg["save_quality_jpeg"])

    meta = {"mode": final_mode, "theme": theme}
    return out_path, meta
