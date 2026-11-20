#!/usr/bin/env python3
"""Download only a Kaggle kernel's execution log, not its working-directory files."""

import argparse
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="owner/kernel-slug")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    # Kaggle API 2.x exposes the persisted/current session log directly.
    # The removed ``process_response(kernel_output_with_http_info(...))``
    # path failed before it could inspect a running DS10 campaign.
    log = api.kernels_logs(args.kernel)
    if not log:
        raise RuntimeError(f"Kaggle returned no log for {args.kernel}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(log, encoding="utf-8")
    print(f"Kernel log downloaded to {args.output}")


if __name__ == "__main__":
    main()
