import json, struct, sys

p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
keys = [k for k in hdr if k != "__metadata__"]
print("file:", p)
print("header_bytes:", n)
print("num_tensors:", len(keys))
for k in keys[:5]:
    t = hdr[k]
    print("  name:", k)
    print("    dtype:", t.get("dtype"), "shape:", t.get("shape"), "offsets:", t.get("data_offsets"))
print("metadata:", hdr.get("__metadata__"))
print("first_keys_total:", keys)
