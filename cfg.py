import os

TEMPLATES_DIR = os.getenv("TEMPLATES_DIR", "./tpl")
TEMPLATES_DIR = os.path.abspath(os.path.expanduser(TEMPLATES_DIR))

cfg = {
    "work_dir": "./.work",
    "max_global_workers": 2,
    "per_user_queue": 5,
    "number_locale": "id_ID",
    "templates_dir": TEMPLATES_DIR,  # bisa /root/editctc
    "output_dir": "./out",
    "roi": {
        "android": {"dx": -300, "dy": 20, "w": 260, "h": 72},
        "iphone":  {"dx": -300, "dy": 30, "w": 260, "h": 78}
    },
    "theme_threshold": 0.6,
    "match_scales": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    "match_threshold": 0.82,
    "save_quality_jpeg": 95
}
