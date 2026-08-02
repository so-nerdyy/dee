"""DS4 golden tests: official DeepSeek-V4-Flash-0731 tokenizer + encoding parity.

Every golden value below was produced by the PINNED official implementation
(``encoding_dsv4.py`` + official ``tokenizer.json``) and is frozen here.
Any change to the official assets or the Freebuff wrapper that shifts a token
ID or prompt string fails these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    sys.stderr.write("tests/test_deepseek_v4_encoding.py requires pytest\n")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import deepseek_v4_encoding as enc  # noqa: E402


# ---------------------------------------------------------------------------
# Golden prompts (pinned from the official implementation, 2026-08-02)
# ---------------------------------------------------------------------------

CANONICAL_USER = [{"role": "user", "content": "Hello, world!"}]

GOLD_CHAT_TEXT = (
    "<｜begin▁of▁sentence｜><｜User｜>Hello, world!<｜Assistant｜></think>"
)
GOLD_CHAT_IDS = [0, 128803, 19923, 14, 2058, 3, 128804, 128822]

GOLD_THINK_LOW_TEXT = (
    "<｜begin▁of▁sentence｜><｜User｜>Hello, world!<｜Assistant｜><think>"
)
GOLD_THINK_LOW_IDS = [0, 128803, 19923, 14, 2058, 3, 128804, 128821]

GOLD_THINK_HIGH_TEXT = (
    "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts "
    "permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose "
    "the problem to resolve the root cause, rigorously stress-testing your "
    "logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every "
    "intermediate step, considered alternative, and rejected hypothesis to "
    "ensure absolutely no assumption is left unchecked.\n"
    "\n"
    "<｜User｜>Hello, world!<｜Assistant｜><think>"
)
GOLD_THINK_HIGH_IDS = [
    0, 77666, 288, 9504, 482, 28, 65174, 8173, 418, 1119, 102810, 23463, 603,
    3476, 74366, 366, 1855, 15146, 295, 782, 6892, 305, 98919, 107029, 270,
    3295, 304, 19727, 270, 4798, 4776, 14, 127468, 5505, 30181, 288, 782,
    14188, 2765, 710, 3283, 20829, 14, 9449, 4599, 14, 305, 101818, 21805, 603,
    2700, 19894, 367, 5085, 798, 782, 5221, 111618, 1699, 14, 76323, 1750,
    20368, 3132, 14, 5083, 9235, 14, 305, 23047, 16915, 304, 5261, 16808, 1119,
    20539, 344, 3001, 114867, 339, 128803, 19923, 14, 2058, 3, 128804, 128821,
]

GOLD_THINK_MAX_TEXT = (
    "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum — exhaustive, "
    "relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely "
    "nothing to chance: exhaustively decompose the problem into its most "
    "fundamental components, trace every causal chain to its root, and resolve "
    "the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution "
    "from multiple angles and are certain that no assumption remains unchecked "
    "and no error remains undiscovered.\n"
    "\n"
    "<｜User｜>Hello, world!<｜Assistant｜><think>"
)
GOLD_THINK_MAX_IDS = [
    0, 77666, 288, 9504, 482, 28, 30041, 8173, 2136, 72604, 14, 65766, 14, 305,
    112934, 496, 4142, 603, 3476, 74366, 3986, 418, 270, 57633, 9335, 305,
    64229, 14, 10981, 16808, 5760, 304, 8369, 28, 17167, 2391, 107029, 270,
    3295, 1055, 1009, 1473, 11264, 7257, 14, 19685, 1750, 37772, 10562, 304,
    1009, 4798, 14, 305, 19727, 270, 13716, 4776, 4562, 1099, 1117, 4433,
    37175, 603, 8041, 554, 6409, 22805, 3514, 440, 611, 21632, 32457, 270,
    4630, 538, 4990, 18534, 305, 477, 3480, 396, 1119, 20539, 7926, 114867,
    305, 1119, 5610, 7926, 64887, 41866, 339, 128803, 19923, 14, 2058, 3,
    128804, 128821,
]

MULTITURN = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4."},
    {"role": "user", "content": "And 3+3?"},
]
GOLD_MULTITURN_TEXT = (
    "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>4."
    "<｜end▁of▁sentence｜><｜User｜>And 3+3?<｜Assistant｜></think>"
)
GOLD_MULTITURN_IDS = [
    0, 128803, 3085, 344, 223, 20, 13, 20, 33, 128804, 128822, 22, 16, 1,
    128803, 4195, 223, 21, 13, 21, 33, 128804, 128822,
]

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    },
}

# Official behavior: the tools template is rendered on SYSTEM/DEVELOPER
# messages only; tools on a user message are silently ignored.
TOOL_SYSTEM = [
    {"role": "system", "content": "You are a helpful assistant.", "tools": [TOOL_SCHEMA]},
    {"role": "user", "content": "Get weather for Beijing"},
]
GOLD_TOOL_SYS_IDS_LEN = 288
GOLD_TOOL_SYS_IDS_HEAD = [0, 3476, 477, 260, 11502, 22896, 339, 372]
GOLD_TOOL_SYS_IDS_TAIL = [603, 128803, 6287, 9670, 362, 29660, 128804, 128821]

GOLD_ASST_TOOL_TURN_IDS = [
    43, 1531, 1347, 270, 1178, 65, 50219, 4105, 16, 128822, 271, 30, 128825,
    72461, 4941, 12548, 1018, 30, 128825, 40148, 5406, 2329, 1281, 1133, 65,
    50219, 3816, 30, 128825, 41523, 2329, 1281, 33182, 4, 3418, 1281, 11476,
    3320, 121458, 1718, 128825, 41523, 1018, 30, 128825, 41523, 2329, 1281,
    15165, 4, 3418, 1281, 11476, 3320, 69, 33030, 1718, 128825, 41523, 1018,
    1718, 128825, 40148, 5406, 1018, 1718, 128825, 72461, 4941, 12548, 32, 1,
]

GOLD_PARSED = {
    "role": "assistant",
    "content": "2 + 2 = 4.",
    "reasoning_content": "Simple arithmetic.",
    "tool_calls": [],
}


# ---------------------------------------------------------------------------
# Asset pins
# ---------------------------------------------------------------------------

def test_tokenizer_asset_hash_pins() -> None:
    hashes = enc.verify_tokenizer_assets()
    assert hashes["tokenizer_json_sha256"] == enc.TOKENIZER_JSON_SHA256
    assert hashes["tokenizer_config_sha256"] == enc.TOKENIZER_CONFIG_SHA256


def test_tokenizer_basics() -> None:
    tok = enc.load_tokenizer()
    # transformers may wrap the backend class differently across versions;
    # the stable pins are the vocab size and the special-token IDs below.
    assert type(tok).__name__ in ("TokenizersBackend", "PreTrainedTokenizerFast")
    assert tok.vocab_size == 128000
    assert tok.bos_token_id == 0
    assert tok.eos_token_id == 1
    assert tok.pad_token_id == 1


# ---------------------------------------------------------------------------
# Golden exact token IDs
# ---------------------------------------------------------------------------

def test_canonical_chat_exact_ids() -> None:
    text, ids = enc.encode_message_ids(CANONICAL_USER, thinking_mode="chat")
    assert text == GOLD_CHAT_TEXT
    assert ids == GOLD_CHAT_IDS


def test_thinking_low_exact_ids() -> None:
    text, ids = enc.encode_message_ids(CANONICAL_USER, thinking_mode="thinking")
    assert text == GOLD_THINK_LOW_TEXT
    assert ids == GOLD_THINK_LOW_IDS


def test_reasoning_effort_high_exact_ids() -> None:
    text, ids = enc.encode_message_ids(
        CANONICAL_USER, thinking_mode="thinking", reasoning_effort="high"
    )
    assert text == GOLD_THINK_HIGH_TEXT
    assert ids == GOLD_THINK_HIGH_IDS


def test_reasoning_effort_max_exact_ids() -> None:
    text, ids = enc.encode_message_ids(
        CANONICAL_USER, thinking_mode="thinking", reasoning_effort="max"
    )
    assert text == GOLD_THINK_MAX_TEXT
    assert ids == GOLD_THINK_MAX_IDS


def test_multi_turn_exact_ids() -> None:
    text, ids = enc.encode_message_ids(MULTITURN, thinking_mode="chat")
    assert text == GOLD_MULTITURN_TEXT
    assert ids == GOLD_MULTITURN_IDS


def test_tool_message_encoding() -> None:
    # Tools render only on system/developer role messages (official behavior).
    text, ids = enc.encode_message_ids(TOOL_SYSTEM, thinking_mode="thinking")
    assert text.startswith(enc.BOS_TEXT)
    assert "## Tools" in text
    assert "### Available Tool Schemas" in text
    assert "get_weather" in text
    assert enc.USER_SP in text
    assert text.endswith(enc.ASSISTANT_SP + enc.THINK_START)
    # tools present -> drop_thinking disabled (effective False)
    assert len(ids) == GOLD_TOOL_SYS_IDS_LEN
    assert ids[:8] == GOLD_TOOL_SYS_IDS_HEAD
    assert ids[-8:] == GOLD_TOOL_SYS_IDS_TAIL


def test_parse_message_roundtrip() -> None:
    completion = "Simple arithmetic.</think>2 + 2 = 4." + enc.EOS_TEXT
    parsed = enc.parse_completion(completion, thinking_mode="thinking")
    assert parsed == GOLD_PARSED


def test_parse_tool_call_completion() -> None:
    # Build the assistant turn with the OFFICIAL encoder, then parse it back:
    # the exact DSML rendering (｜DSML｜-prefixed tags, \n\n<｜DSML｜tool_calls>
    # delimiter) must round-trip through the official parser.
    import json as _json
    official = enc.load_official_encoding()
    messages = [{
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should use the get_weather tool.",
        "tool_calls": [{
            "function": {
                "name": "get_weather",
                "arguments": _json.dumps({"location": "Beijing", "unit": "celsius"}),
            },
        }],
    }]
    completion = official.encode_messages(
        messages, thinking_mode="thinking", add_default_bos_token=False
    )
    # exact official token IDs of the rendered assistant turn
    assert enc.tokenize(completion) == GOLD_ASST_TOOL_TURN_IDS
    parsed = enc.parse_completion(completion, thinking_mode="thinking")
    assert parsed["reasoning_content"] == "I should use the get_weather tool."
    assert parsed["content"] == ""
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    args = _json.loads(call["function"]["arguments"])
    assert args == {"location": "Beijing", "unit": "celsius"}


# ---------------------------------------------------------------------------
# Wrapper parity + no-generic-template guarantees
# ---------------------------------------------------------------------------

def test_wrapper_parity_with_official_direct() -> None:
    """The wrapper must be a faithful passthrough of the official module."""
    official = enc.load_official_encoding()
    conversations = [
        (CANONICAL_USER, "chat", None),
        (CANONICAL_USER, "thinking", None),
        (CANONICAL_USER, "thinking", "high"),
        (CANONICAL_USER, "thinking", "max"),
        (MULTITURN, "chat", None),
        (TOOL_SYSTEM, "thinking", None),
        ([{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is 2+2?"}], "thinking", "low"),
    ]
    for messages, mode, effort in conversations:
        wrapped = enc.encode_messages(
            messages, mode, reasoning_effort=effort
        )
        direct = official.encode_messages(
            messages, mode, reasoning_effort=effort
        )
        assert wrapped == direct
        assert enc.tokenize(wrapped) == enc.tokenize(direct)


def test_official_encoding_not_generic_chat_template() -> None:
    """The prompt is the official V4 format, not a generic chat template."""
    text, ids = enc.encode_message_ids(CANONICAL_USER, thinking_mode="thinking")
    # full-width special tokens (U+FF5C half-width bar inside the glyphs)
    assert "｜" in text
    assert enc.USER_SP in text
    assert enc.ASSISTANT_SP in text
    assert text.endswith(enc.THINK_START) or text.endswith(enc.THINK_END)
    # the official tokenizer's own (generic) chat-template path differs:
    # for this backend there is no Jinja template configured in the pinned
    # config, so the official encoding is the only supported prompt format.
    tok = enc.load_tokenizer()
    assert not hasattr(tok, "chat_template") or tok.chat_template is None
    # ids are stable across the two modes' boundary tokens
    assert ids[-1] != GOLD_CHAT_IDS[-1]  # 128821 <think> vs 128822 </think>


def test_reasoning_effort_invalid_rejected() -> None:
    with pytest.raises(AssertionError):
        enc.encode_messages(
            CANONICAL_USER, thinking_mode="thinking", reasoning_effort="turbo"
        )


def test_invalid_thinking_mode_rejected() -> None:
    with pytest.raises(AssertionError):
        enc.encode_messages(CANONICAL_USER, thinking_mode="autocomplete")


GOLD_LATEST_REMINDER_TEXT = (
    "<｜begin▁of▁sentence｜><｜latest_reminder｜>You are DeepSeek-V4, today is "
    "2026-08-02.<｜User｜>hi<｜Assistant｜></think>"
)
GOLD_LATEST_REMINDER_IDS = [
    0, 128828, 3476, 477, 22651, 4374, 1465, 13582, 22, 14, 4316, 344, 223,
    939, 24, 15, 3019, 15, 3425, 16, 128803, 6366, 128804, 128822,
]

GOLD_CONTEXT_TEXT = "<｜User｜>And 3+3?<｜Assistant｜></think>"
GOLD_CONTEXT_IDS = [128803, 4195, 223, 21, 13, 21, 33, 128804, 128822]


def test_latest_reminder_role_exact_ids() -> None:
    # The official latest_reminder role renders its own full-width marker.
    messages = [
        {"role": "latest_reminder", "content": "You are DeepSeek-V4, today is 2026-08-02."},
        {"role": "user", "content": "hi"},
    ]
    text, ids = enc.encode_message_ids(messages, thinking_mode="chat")
    assert text == GOLD_LATEST_REMINDER_TEXT
    assert ids == GOLD_LATEST_REMINDER_IDS
    assert enc.LATEST_REMINDER_SP in text


def test_context_parameter_multi_turn() -> None:
    # Official multi-turn context: the encoded prefix is passed as ``context``
    # and must NOT be re-rendered or re-add BOS.
    context = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]
    messages = [{"role": "user", "content": "And 3+3?"}]
    text, ids = enc.encode_message_ids(messages, thinking_mode="chat", context=context)
    assert text == GOLD_CONTEXT_TEXT
    assert ids == GOLD_CONTEXT_IDS
    assert not text.startswith(enc.BOS_TEXT)


def test_official_task_special_tokens_pinned() -> None:
    tasks = enc.official_task_special_tokens()
    assert set(tasks) == {"action", "query", "authority", "domain", "title", "read_url"}
    assert tasks["action"] == "<｜action｜>"
