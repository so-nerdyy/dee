#!/usr/bin/env python3
"""Push through official Kaggle API 1.x while setting the current machineShape field.

Kaggle's legacy username/key authentication still works with the official 1.x
CLI, while its generated request model predates ``machineShape``. The server
already accepts that field. This compatibility shim adds it to the generated
model immediately before the normal official ``kernels_push`` request.

No credential is read or printed here; ``KaggleApi.authenticate`` uses the
normal local ``~/.kaggle/kaggle.json`` contract.
"""

import argparse

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.models.kernel_push_request import KernelPushRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--accelerator", required=True)
    args = parser.parse_args()

    # The official current CLI calls this field ``machine_shape`` and exposes
    # it as ``kaggle kernels push --accelerator``. Extend only the generated
    # serialization table used by the authenticated legacy client.
    KernelPushRequest.swagger_types = dict(KernelPushRequest.swagger_types)
    KernelPushRequest.attribute_map = dict(KernelPushRequest.attribute_map)
    KernelPushRequest.swagger_types["machine_shape"] = "str"
    KernelPushRequest.attribute_map["machine_shape"] = "machineShape"

    api = KaggleApi()
    api.authenticate()
    original = api.kernel_push_with_http_info

    def with_accelerator(*positional, **kwargs):
        request = kwargs.get("kernel_push_request")
        if request is None and positional:
            request = positional[0]
        if request is None:
            raise RuntimeError("official Kaggle client produced no kernel request")
        request.machine_shape = args.accelerator
        return original(*positional, **kwargs)

    api.kernel_push_with_http_info = with_accelerator
    result = api.kernels_push(args.path)
    if result is None or result.error:
        raise RuntimeError("Kaggle push failed: " + (result.error if result else "no response"))
    print(f"Kernel version {result.versionNumber} successfully pushed. {result.url}")


if __name__ == "__main__":
    main()

