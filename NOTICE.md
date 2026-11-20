# Third-party notices

DEE is MIT licensed (see [`LICENSE`](LICENSE)). It includes or redistributes
the following third-party material.

| component | where | license |
|---|---|---|
| ggml (from llama.cpp), v0.17.0 | `dee.cpp/third_party/ggml/` | MIT, [`dee.cpp/third_party/ggml/LICENSE`](dee.cpp/third_party/ggml/LICENSE), Copyright (c) The ggml authors |
| DeepSeek-V4-Flash-0731 config, tokenizer, chat encoding and reference inference code | `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/official-source/`, `tokenizer-assets/` | MIT, Copyright (c) DeepSeek, as published with the model on Hugging Face |

No model weights are stored in this repository. Runs download the DeepSeek-V4-Flash
and MiMo-V2.6-Flash checkpoints from their publishers, under those models'
own licenses. `dee.cpp/oracle.pt` (Git LFS) is a small expert predictor trained
by this project.
