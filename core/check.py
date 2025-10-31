import os
from typing import List, Tuple
from core.utils import resolve_digit_path, resolve_anchor_path

REQUIRED_DIGITS = [*(f"{i}.png" for i in range(10)), "dot.png"]

def verify_template_pack(mode: str, theme: str) -> List[str]:
    errs = []
    # digits
    for name in REQUIRED_DIGITS:
        if not resolve_digit_path(mode, theme, name):
            errs.append(f"Missing {mode}/{theme}/{name}")
    # anchor
    if not resolve_anchor_path(mode, theme, "anggota.png"):
        errs.append(f"Missing anchor {mode}/{theme}/anggota.png")
    return errs
