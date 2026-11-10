#!/usr/bin/env python3
"""Push the host-sync profile kernel (local use)."""
import os
import sys

os.environ.setdefault("KAGGLE_CONFIG_DIR", r"C:\Users\carth\.kaggle")

import kaggle  # noqa: E402

api = kaggle.api
folder = sys.argv[1]
print("pushing", folder)
resp = api.kernels_push(folder, timeout="14400", acc="NvidiaTeslaT4")
d = resp if isinstance(resp, dict) else {
    k: getattr(resp, k) for k in dir(resp)
    if not k.startswith("_") and not callable(getattr(resp, k))}
print(__import__("json").dumps(d, indent=1, default=str)[:1500])
