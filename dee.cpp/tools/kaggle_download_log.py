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
    owner, slug = args.kernel.split("/", 1)

    api = KaggleApi()
    api.authenticate()
    response = api.process_response(api.kernel_output_with_http_info(owner, slug))
    log = response.get("log")
    if not log:
        raise RuntimeError(f"Kaggle returned no log for {args.kernel}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(log, encoding="utf-8")
    print(f"Kernel log downloaded to {args.output}")


if __name__ == "__main__":
    main()

