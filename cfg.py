import os

cfg = {
    "work_dir": "./.work",
    "max_global_workers": 2,     # aman untuk VPS 1GB
    "per_user_queue": 5,
    "number_locale": "id_ID",    # hanya untuk referensi
    "templates_dir": "./tpl",
    "output_dir": "./out",
    "roi": {
        # ROI dihitung relatif dari anchor 'anggota' (match top-left x,y dan w,h template anchor).
        # number_box = (anchor_x + dx, anchor_y + dy, width, height)
        "android": {"dx": -300, "dy": 20, "w": 260, "h": 72},
        "iphone":  {"dx": -300, "dy": 30, "w": 260, "h": 78}
    },
    "theme_threshold": 0.6,      # > 0.6 = light; else dark (berdasarkan luminance)
    "match_scales": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    "match_threshold": 0.82,     # template matching correlation threshold
    "save_quality_jpeg": 95
}