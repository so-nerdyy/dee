"""DS4: official DeepSeek-V4-Flash-0731 tokenizer + encoding parity wrapper.

DeepSeek-V4-Flash-0731 does NOT use an ordinary Jinja chat template. Its
prompt format (full-width special tokens, reasoning-effort prefixes, DSML
tool markup, thinking-mode handling) is defined by the official encoding
package ``encoding_dsv4.py`` and the official tokenizer (``tokenizer.json``,
vocab 128000, BOS=0, EOS=1, PAD=1).

This module pins both artifacts and exposes the exact official behaviors
through a small Freebuff-facing API:

- ``encode_messages`` / ``parse_completion``: direct passthroughs to the
  pinned official ``encoding_dsv4`` module (no reimplementation).
- ``tokenize`` / ``encode_message_ids``: exact official token IDs.
- ``verify_tokenizer_assets``: fail-closed SHA-256 pin check on the official
  tokenizer assets.

Golden parity tests pin exact token IDs from the pinned tokenizer; any drift
in the official assets or this wrapper fails closed.

Asset resolution (first match wins):

1. ``DEE_CPP_DSV4_TOKENIZER_DIR`` environment variable;
2. repository ``benchmark_reports/deepseek-v4-flash-0731-t4/tokenizer-assets``
   (the clean two-file set: ``tokenizer.json`` + ``tokenizer_config.json``;
   the CDN-stub ``special_tokens_map.json`` / ``vocab.json`` are excluded);
3. a flat ``tokenizer-assets`` directory next to this module (kernel-style
   deployment).

Nothing here requires CUDA, a GPU, or the model checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

# Pinned official repository revision (mirrors scripts/deepseek_v4_support
# OFFICIAL_REVISION). Kept local so this module stays importable as a flat
# copy on Kaggle (the harness flattens modules to ``deepseek_v4_module_*.py``
# where ``import scripts...`` would fail).
OFFICIAL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

# ---------------------------------------------------------------------------
# Pinned official identity
# ---------------------------------------------------------------------------

# SHA-256 of the official assets copied from the pinned repository revision.
TOKENIZER_JSON_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
TOKENIZER_CONFIG_SHA256 = "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547"

VOCAB_SIZE = 128000
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 1

# Full-width special tokens rendered by the official encoder.
BOS_TEXT = "<｜begin▁of▁sentence｜>"
USER_SP = "<｜User｜>"
ASSISTANT_SP = "<｜Assistant｜>"
LATEST_REMINDER_SP = "<｜latest_reminder｜>"
THINK_START = "<think>"
THINK_END = "</think>"
EOS_TEXT = "<｜end▁of▁sentence｜>"

_ENCODING_MODULE_NAME = "_freebuff_official_encoding_dsv4"
_TOKENIZER_CACHE: dict[str, Any] = {}
_ENCODING_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Asset resolution + verification
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    # scripts/deepseek_v4_encoding.py -> repo root
    return Path(__file__).resolve().parent.parent


def _candidate_asset_dirs() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("DEE_CPP_DSV4_TOKENIZER_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(
        _repo_root()
        / "benchmark_reports"
        / "deepseek-v4-flash-0731-t4"
        / "tokenizer-assets"
    )
    candidates.append(Path(__file__).resolve().parent.parent / "tokenizer-assets")
    candidates.append(Path(__file__).resolve().parent / "tokenizer-assets")
    return candidates


def tokenizer_assets_dir() -> Path:
    """Resolve the directory holding tokenizer.json + tokenizer_config.json."""
    for candidate in _candidate_asset_dirs():
        if (candidate / "tokenizer.json").is_file() and (
            candidate / "tokenizer_config.json"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "DeepSeek-V4 tokenizer assets not found; tried: "
        + ", ".join(str(c) for c in _candidate_asset_dirs())
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tokenizer_assets() -> dict[str, str]:
    """Fail-closed SHA-256 pin check on the official tokenizer assets.

    Returns the two hashes on success; raises ValueError on any mismatch so a
    drifted or substituted tokenizer can never silently change token IDs.
    """
    assets = tokenizer_assets_dir()
    actual_json = _sha256(assets / "tokenizer.json")
    actual_config = _sha256(assets / "tokenizer_config.json")
    if actual_json != TOKENIZER_JSON_SHA256:
        raise ValueError(
            f"tokenizer.json hash mismatch: expected {TOKENIZER_JSON_SHA256}, "
            f"got {actual_json} from {assets}"
        )
    if actual_config != TOKENIZER_CONFIG_SHA256:
        raise ValueError(
            f"tokenizer_config.json hash mismatch: expected "
            f"{TOKENIZER_CONFIG_SHA256}, got {actual_config} from {assets}"
        )
    return {"tokenizer_json_sha256": actual_json,
            "tokenizer_config_sha256": actual_config}


# ---------------------------------------------------------------------------
# Official tokenizer + encoding module loaders (lazy, cached)
# ---------------------------------------------------------------------------

def load_tokenizer() -> Any:
    """Load the official tokenizer via AutoTokenizer (lazy, cached)."""
    if _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE["tokenizer"]
    verify_tokenizer_assets()
    from transformers import AutoTokenizer  # local import: not a hard dep
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_assets_dir()))
    if tokenizer.vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"unexpected vocab size {tokenizer.vocab_size}, expected {VOCAB_SIZE}"
        )
    if tokenizer.bos_token_id != BOS_TOKEN_ID or tokenizer.eos_token_id != EOS_TOKEN_ID:
        raise ValueError(
            f"unexpected BOS/EOS ids {tokenizer.bos_token_id}/"
            f"{tokenizer.eos_token_id}"
        )
    _TOKENIZER_CACHE["tokenizer"] = tokenizer
    return tokenizer


def load_official_encoding() -> Any:
    """Import the pinned official ``encoding_dsv4`` module (lazy, cached)."""
    if _ENCODING_CACHE:
        return _ENCODING_CACHE["module"]
    encoding_py = (
        _repo_root()
        / "benchmark_reports"
        / "deepseek-v4-flash-0731-t4"
        / "official-source"
        / "encoding"
        / "encoding_dsv4.py"
    )
    if not encoding_py.is_file():
        raise FileNotFoundError(f"official encoding module missing: {encoding_py}")
    spec = importlib.util.spec_from_file_location(_ENCODING_MODULE_NAME, encoding_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {encoding_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ENCODING_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _ENCODING_CACHE["module"] = module
    return module


# ---------------------------------------------------------------------------
# Freebuff-facing API (exact official semantics)
# ---------------------------------------------------------------------------

def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str,
    *,
    context: list[dict[str, Any]] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    """Encode an OpenAI-style message list into the official V4 prompt text.

    Pure passthrough to the pinned official ``encode_messages``.
    """
    return load_official_encoding().encode_messages(
        messages,
        thinking_mode,
        context=context,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
    )


def tokenize(text: str) -> list[int]:
    """Exact official token IDs for an already-encoded prompt string.

    Mirrors the official inference path: ``tokenizer.encode(prompt)`` with the
    default special-token handling (the prompt itself already contains BOS as
    text, so no extra BOS is injected). The pinned ``tokenizer_config.json``
    declares ``add_bos_token: false`` / ``add_eos_token: false``, so the
    default ``encode`` adds no special tokens regardless.
    """
    return load_tokenizer().encode(text)


def encode_message_ids(
    messages: list[dict[str, Any]],
    thinking_mode: str,
    *,
    reasoning_effort: str | None = None,
    **kwargs: Any,
) -> tuple[str, list[int]]:
    """Encode messages to prompt text AND the exact official token IDs."""
    prompt = encode_messages(
        messages, thinking_mode, reasoning_effort=reasoning_effort, **kwargs
    )
    return prompt, tokenize(prompt)


def parse_completion(text: str, thinking_mode: str) -> dict[str, Any]:
    """Parse a model completion into a structured message (official parser)."""
    return load_official_encoding().parse_message_from_completion_text(
        text, thinking_mode
    )


def official_task_special_tokens() -> dict[str, str]:
    """Official DS task special tokens (action/query/authority/...)."""
    return dict(load_official_encoding().DS_TASK_SP_TOKENS)
