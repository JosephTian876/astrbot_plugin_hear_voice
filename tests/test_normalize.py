"""Unit tests for model name normalization, rich 6-field parsing, and error filters."""

from __future__ import annotations

from hear_voice.normalize import (
    build_stt_instruction,
    clean_transcript,
    is_instruction_hallucination,
    is_provider_error_text,
    normalize_model_name,
    parse_rich_output,
)


def test_normalize_model_name_preserves_prefixes_by_default():
    assert normalize_model_name("［抗截断］gemini-3.8-flash", enabled=False) == "［抗截断］gemini-3.8-flash"
    assert normalize_model_name("[AN]gemini-4-hydrogen-high", enabled=False) == "[AN]gemini-4-hydrogen-high"
    assert normalize_model_name("models/gemini-2.5-flash", enabled=False) == "models/gemini-2.5-flash"
    assert normalize_model_name("", enabled=False) == "gemini-2.5-flash"


def test_normalize_model_name_strips_when_enabled():
    assert normalize_model_name("[满血D]gemini-2.5-flash", enabled=True) == "gemini-2.5-flash"
    assert normalize_model_name("vendor/models/gemini-2.5-flash", enabled=True) == "gemini-2.5-flash"
    assert normalize_model_name("［抗截断］gemini-3.8-flash", enabled=True) == "［抗截断］gemini-3.8-flash"


def test_build_stt_instruction_modes():
    rich_prompt = build_stt_instruction("rich", user_text="冬马")
    for label in ("原话转写", "语气/情绪", "说话人/音色", "语速/停顿", "环境音", "大意总结", "冬马"):
        assert label in rich_prompt

    simple_prompt = build_stt_instruction("simple")
    assert "纯文本原话" in simple_prompt
    assert "6 项" not in simple_prompt

    custom = build_stt_instruction("rich", custom_instruction="CUSTOM PROMPT")
    assert custom.startswith("CUSTOM PROMPT")


def test_parse_rich_output_extracts_all_six_fields():
    raw = (
        "1) 原话转写：今天下雪了，记得多穿点衣服呀。\n"
        "2) 语气/情绪：温柔关切、带点撒娇\n"
        "3) 说话人/音色：单人年轻女声，声线清亮，普通话\n"
        "4) 语速/停顿：语速偏慢，尾音轻柔拉长\n"
        "5) 环境音：室内安静，偶有轻微衣物摩擦声\n"
        "6) 大意总结：提醒对方天冷注意保暖。"
    )
    res = parse_rich_output(raw, output_mode="rich")
    assert res.transcript == "今天下雪了，记得多穿点衣服呀。"
    assert "温柔关切" in res.emotion
    assert "年轻女声" in res.speaker
    assert "语速偏慢" in res.speed
    assert "室内安静" in res.ambient
    assert "注意保暖" in res.summary
    assert res.is_silent is False


def test_parse_rich_output_handles_legacy_labels_and_silence():
    legacy = (
        "1) 原话转写：【用户未说话】\n"
        "2) 语言：【不适用】\n"
        "3) 语气/情绪：【不适用】\n"
        "4) 环境音：持续键盘敲击声。\n"
        "5) 说话人数：【不适用】\n"
        "6) 大意总结：只有打字声音。"
    )
    res = parse_rich_output(legacy, output_mode="rich")
    assert res.is_silent is True
    assert "未检测到说话声" in res.transcript
    assert "只有打字声音" in res.transcript


def test_clean_transcript_collapses_newlines_and_truncates():
    text = "line1\n\n\n\nline2   "
    assert clean_transcript(text, enabled=True, max_chars=100) == "line1\n\nline2"
    truncated = clean_transcript("a" * 50, enabled=True, max_chars=10)
    assert truncated == "a" * 10 + "..."
    assert clean_transcript(text, enabled=False, max_chars=5) == text.strip()


def test_provider_error_and_hallucination_detection():
    assert is_provider_error_text("Gemini 3 Pro is no longer available. Please switch to a supported model.")
    assert is_provider_error_text('{"error": {"code": 400, "message": "Unsupported MIME type"}}')
    assert is_provider_error_text("Quota exceeded for this project")
    assert not is_provider_error_text("1) 原话转写：我们去喝奶茶吧")

    instr = build_stt_instruction("rich")
    assert is_instruction_hallucination(instr, instr)
    assert not is_instruction_hallucination("1) 原话转写：晚上吃火锅", instr)
