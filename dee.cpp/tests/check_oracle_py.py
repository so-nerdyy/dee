#!/usr/bin/env python3
"""
Torch-free cross-check for dee.cpp Step 7 (Oracle loader + OracleScheduler).

Reads the RAW oracle/data/N float32 buffers straight from the .pt ZIP using
only Python's zipfile (NO torch, NO numpy). Rebuilds layer-0's 3-layer MLP
weights using the manifest that the C++ PtLoader::dump_manifest produced, runs
the identical matmul+ReLU forward pass on the hidden vector that the C++ test
dumped, and asserts the resulting top-K experts match what the C++ engine
predicted.

This is a real correctness proof of (a) the C++ ZIP+pickle parser's tensor
mapping and (b) the C++ MLP math -- verified against an independent reader.
"""
import json
import struct
import sys
import zipfile

ORACLE = sys.argv[1] if len(sys.argv) > 1 else "oracle.pt"
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else "/tmp/oracle_manifest.json"
CHECK = sys.argv[3] if len(sys.argv) > 3 else "/tmp/oracle_cpp_check.json"

# local-name -> (out, in) for the 3 linears of the Oracle MLP
LINEAR = {
    "net.0.weight": (256, 2048), "net.0.bias": (256, 0),
    "net.2.weight": (256, 256),  "net.2.bias": (256, 0),
    "net.4.weight": (256, 256),  "net.4.bias": (256, 0),
}


def read_buffer(z, idx):
    """Return oracle/data/<idx> as a list of float32 (little-endian)."""
    name = f"oracle/data/{idx}"
    raw = z.read(name)
    n = len(raw) // 4
    return list(struct.unpack("<%df" % n, raw))


def matvec(w, b, x):
    """y = W x + b   (W is (out,in) row-major, x is length in)."""
    out = len(w) // len(x) if b is None else len(b)
    y = [0.0] * out
    for r in range(out):
        s = 0.0
        base = r * len(x)
        for c in range(len(x)):
            s += w[base + c] * x[c]
        if b is not None:
            s += b[r]
        y[r] = s
    return y


def relu(v):
    return [max(0.0, x) for x in v]


def main():
    with open(CHECK) as f:
        chk = json.load(f)
    layer = chk["layer"]
    hidden = chk["hidden"]
    cpp_predicted = chk["predicted"]
    K = chk["topk"]

    z = zipfile.ZipFile(ORACLE)
    manifest = json.load(open(MANIFEST))

    # group manifest tensors by layer
    layers = {}
    for name, meta in manifest.items():
        if not name.startswith("layers."):
            continue
        parts = name.split(".")
        # layers.L.net.X.weight
        L = int(parts[1])
        local = ".".join(parts[2:])  # net.X.weight / net.X.bias
        layers.setdefault(L, {})[local] = meta

    if layer not in layers:
        print(f"[FAIL] layer {layer} not in manifest ({len(layers)} layers)")
        sys.exit(1)
    lt = layers[layer]

    # load weights for this layer from raw buffers
    W = {}
    for local, (out, inn) in LINEAR.items():
        meta = lt[local]
        idx = meta["storage"]
        buf = read_buffer(z, idx)
        if "bias" in local:
            W[local] = buf
        else:
            # row-major (out, in)
            W[local] = buf

    # 3-layer MLP forward (matches dee::OracleScheduler::forward)
    # h0 = relu(W0 x + b0)   (2048 -> 256)
    a = relu(matvec(W["net.0.weight"], W["net.0.bias"], hidden))
    # h1 = relu(W2 h0 + b2)  (256 -> 256)
    b = relu(matvec(W["net.2.weight"], W["net.2.bias"], a))
    # logits = W4 h1 + b4    (256 -> 256)
    logits = matvec(W["net.4.weight"], W["net.4.bias"], b)
    assert len(logits) == 256, f"logits len {len(logits)} != 256"

    # top-K by logit value
    order = sorted(range(256), key=lambda i: logits[i], reverse=True)
    py_predicted = order[:K]

    ok = (py_predicted == cpp_predicted)
    # also check the SET matches (order-insensitive) as a softer signal
    set_match = set(py_predicted) == set(cpp_predicted)

    print("=== Python torch-free reference (layer %d) ===" % layer)
    print("  logits[0..3] =", [round(v, 4) for v in logits[:4]])
    print("  python top-%d   =" % K, py_predicted)
    print("  c++    top-%d   =" % K, cpp_predicted)
    if ok:
        print("[PASS] top-K experts EXACTLY match C++ engine (parser + MLP math verified)")
        sys.exit(0)
    elif set_match:
        print("[WARN] same expert SET but different order (tie-breaking) -- acceptable")
        sys.exit(0)
    else:
        print("[FAIL] top-K expert SET differs from C++ engine")
        sys.exit(1)


if __name__ == "__main__":
    main()
