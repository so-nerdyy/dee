# PROJECT_STATE — DeepSeek-V4-Flash-0731 campaign

Updated: 2026-07-31.

## Pinned official assets (all at revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb)

| Asset | SHA256 (local download) |
|---|---|
| config.json | `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023` |
| generation_config.json | `5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0` |
| model.safetensors.index.json | `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` |
| tokenizer.json | `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf` |
| tokenizer_config.json | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| inference/convert.py | `6efe65ebc66b18c9f2656816608f941cacfe20da79c2dee19040ecbee8b42bfe` |
| inference/generate.py | `775fcfee2344e21a7b02c73161c517763e4348b84cf2eb266353e0857b9c8812` |
| inference/kernel.py | `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2` |
| inference/model.py | `c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f` |
| encoding/encoding_dsv4.py | `abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6` |

Local audit copies: `dee.cpp/tmp/dsv4-official-audit/` (headers + index +
source, no weight bytes).

Pinned source is committed with the campaign at
`benchmark_reports/deepseek-v4-flash-0731-t4/official-source/` (config,
generation_config, index, tokenizer, inference/, encoding/) so the DS1
revision pin is reproducible from a fresh clone and the real-index invariant
test can run without `tmp/`.

## Ledger

- `MODEL_LEDGER.json` — 72,317 validated tensor rows.
- `CHECKPOINT_MANIFEST.json` — 48 shards, header-validated sizes.
- Declared total = validated total = `166,878,536,440` bytes. PASS.

## Branch

`freebuff/deepseek-v4-flash-0731-t4` (off `9ff967ef4429fb08a433d6ef0a4495468d89b4ba`).

## Sealed Ornith

Immutable on `codex/phase2-cap32-matrix` @ `9ff967e...`. M5G-v3 = INVALID_EXPERIMENT;
+9.21% claim quarantined.
