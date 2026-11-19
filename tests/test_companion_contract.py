"""Integration contract tests with astrbot_plugin_private_companion and memory_companion."""

from __future__ import annotations

import json
import types
from typing import Any

import pytest
from astrbot.api.message_components import At, Plain, Record
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

from hear_voice.config import HearVoiceConfig
from hear_voice.engine import HearVoiceEngine
from hear_voice.hook_pipeline import (
    EXTRA_COMPANION_TRANSCRIPTS,
    EXTRA_HEAR_VOICE_CACHE_ONLY,
    EXTRA_HEAR_VOICE_DONE,
    EXTRA_HEAR_VOICE_ENGINE_USED,
    EXTRA_HEAR_VOICE_FALLBACK_USED,
    EXTRA_HEAR_VOICE_HIJACKED,
    EXTRA_HEAR_VOICE_IS_GROUP,
    EXTRA_HEAR_VOICE_NON_HIJACKING,
    EXTRA_HEAR_VOICE_NOTIFY_ONLY_SENT,
    EXTRA_HEAR_VOICE_RAW_TEXT,
    EXTRA_HEAR_VOICE_REPLY_REASON,
    EXTRA_HEAR_VOICE_SHOULD_REPLY,
    EXTRA_HEAR_VOICE_STATUS,
    EXTRA_HEAR_VOICE_TRANSCRIPT,
    EXTRA_STT_CACHE_ONLY,
    EXTRA_STT_DONE,
    EXTRA_STT_HIJACKED,
    EXTRA_STT_IS_GROUP,
    EXTRA_STT_NON_HIJACKING,
    EXTRA_STT_NOTIFY_ONLY_SENT,
    EXTRA_STT_RAW_TEXT,
    EXTRA_STT_REPLY_REASON,
    EXTRA_STT_SHOULD_REPLY,
    EXTRA_STT_STATUS,
    EXTRA_STT_TRANSCRIPT,
    HearVoicePipeline,
)
from tests.conftest import FakeChatProvider, FakeContext, FakeEvent, make_data_uri_wav


class _RichStubClient:
    async def transcribe(self, audio, *, instruction: str, endpoint) -> str:
        return (
            "1) 原话转写：下班一起去超市买菜吧\n"
            "2) 语气/情绪：轻松期待\n"
            "3) 说话人/音色：单人男声，普通话\n"
            "4) 语速/停顿：自然流畅\n"
            "5) 环境音：微弱地铁播报声\n"
            "6) 大意总结：约对方下班一起买菜。"
        )

    async def normalize_transcript_with_model(self, transcript: str, *, endpoint) -> str:
        return transcript

    async def close(self) -> None:
        return None

    @staticmethod
    def safe_endpoint_summary(endpoint) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_private_companion_position_and_memory_companion_clean_capture():
    """Verify dual _hear_voice_* + _gemini_stt_* aliases + astrbot_stt_transcripts contract."""
    ctx = FakeContext({"AlmoonsGemini-1": FakeChatProvider()})
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "output_mode": "rich",
            "temp_cleanup_interval_sec": 0,
        }
    )
    pipe = HearVoicePipeline(ctx, cfg, client=_RichStubClient())

    at_bot = At(qq="99999")
    rec = Record(file=make_data_uri_wav())
    raw_msg = {
        "message": [
            {"type": "at", "data": {"qq": "99999"}},
            {"type": "record", "data": {"file": rec.file}},
        ]
    }
    event = FakeEvent(
        messages=[at_bot, rec],
        message_str="",
        group_id="228397848",
        sender_id="302578963",
        sender_name="Joseph",
        raw_message=raw_msg,
    )

    async for _ in pipe.process_message_event(event):
        pass

    # 1. private_companion position contract: Record stays at index 1!
    assert event.message_obj.message[0] is at_bot
    assert event.message_obj.message[1] is rec
    assert isinstance(event.message_obj.message[2], Plain)

    # 2. memory_companion capture_runtime reads Plain components + identity reads message_str
    memory_body = "".join(
        p.text for p in event.message_obj.message if type(p).__name__ == "Plain"
    )
    assert memory_body == "[语音消息]: 下班一起去超市买菜吧"
    assert event.get_message_str() == "[语音消息]: 下班一起去超市买菜吧"
    assert "语气/情绪" not in memory_body
    assert "环境音" not in memory_body

    # 3. Both _hear_voice_* and legacy _gemini_stt_* keys exported identically
    assert event.requested_llm is False
    assert event.is_stopped() is False
    for k_hv, k_leg, exp in (
        (EXTRA_HEAR_VOICE_DONE, EXTRA_STT_DONE, True),
        (EXTRA_HEAR_VOICE_STATUS, EXTRA_STT_STATUS, "ok"),
        (EXTRA_HEAR_VOICE_IS_GROUP, EXTRA_STT_IS_GROUP, True),
        (EXTRA_HEAR_VOICE_SHOULD_REPLY, EXTRA_STT_SHOULD_REPLY, True),
        (EXTRA_HEAR_VOICE_CACHE_ONLY, EXTRA_STT_CACHE_ONLY, False),
        (EXTRA_HEAR_VOICE_REPLY_REASON, EXTRA_STT_REPLY_REASON, "group_passthrough"),
        (EXTRA_HEAR_VOICE_NON_HIJACKING, EXTRA_STT_NON_HIJACKING, True),
        (EXTRA_HEAR_VOICE_HIJACKED, EXTRA_STT_HIJACKED, False),
        (EXTRA_HEAR_VOICE_TRANSCRIPT, EXTRA_STT_TRANSCRIPT, "下班一起去超市买菜吧"),
    ):
        assert event.get_extra(k_hv) == exp
        assert event.get_extra(k_leg) == exp

    assert "语气/情绪" in event.get_extra(EXTRA_HEAR_VOICE_RAW_TEXT)
    assert "语气/情绪" in event.get_extra(EXTRA_STT_RAW_TEXT)
    assert event.get_extra(EXTRA_HEAR_VOICE_ENGINE_USED) == "gemini"
    assert event.get_extra(EXTRA_HEAR_VOICE_FALLBACK_USED) is False

    entries = event.get_extra(EXTRA_COMPANION_TRANSCRIPTS)
    assert isinstance(entries, list) and len(entries) == 1
    assert entries[0]["source"] == "direct"
    assert entries[0]["transcript"] == "下班一起去超市买菜吧"
    assert entries[0]["emotion"] == "轻松期待"
    assert "微弱地铁播报声" in entries[0]["ambient"]

    # 4. Stage 2 protects non-audio providers & marks rich injection _no_save=True
    req = ProviderRequest(
        prompt="<attachment>",
        audio_urls=["C:/temp/decoded.wav"],
        extra_user_content_parts=[TextPart(text="[Audio Attachment: path C:/temp/decoded.wav]")],
    )
    await pipe.process_llm_request(event, req)
    assert req.audio_urls == []
    assert req.prompt == "[语音消息]: 下班一起去超市买菜吧"
    assert not any("Audio Attachment" in p.text for p in req.extra_user_content_parts)
    assert any(p._no_save and "轻松期待" in p.text for p in req.extra_user_content_parts)
    await pipe.shutdown()


def _legacy_hijack_detected(event: FakeEvent) -> bool:
    """Reference downstream companion check that avoids misinterpreting non-hijacking STT."""
    if event.get_extra(EXTRA_STT_NON_HIJACKING) is True or event.get_extra(EXTRA_STT_HIJACKED) is False:
        return False
    if getattr(event, "_has_send_oper", False):
        return True
    return event.get_extra(EXTRA_STT_SHOULD_REPLY) is True and not event.get_extra(EXTRA_STT_CACHE_ONLY)


class _FailingStubClient:
    last_error = "HTTP 400: INVALID_ARGUMENT: The attachment format is not supported"

    async def transcribe(self, audio, *, instruction: str, endpoint) -> str:
        return ""

    async def normalize_transcript_with_model(self, transcript: str, *, endpoint) -> str:
        return transcript

    async def close(self) -> None:
        return None

    @staticmethod
    def safe_endpoint_summary(endpoint) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_companion_notify_pass_and_default_pass_failure_handoff_contract():
    """Verify both default pass and notify_pass keep downstream private_companion unblocked."""
    ctx_primary = FakeContext({"AlmoonsGemini-1": FakeChatProvider()})

    # 1) Default on_stt_fail="pass": silent, no send_message, non-hijacking flags set
    cfg_pass = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "enable_fallback": False,
            "temp_cleanup_interval_sec": 0,
        }
    )
    pipe_pass = HearVoicePipeline(ctx_primary, cfg_pass, client=_FailingStubClient())
    ev_pass = FakeEvent(messages=[Record(file=make_data_uri_wav())], sender_id="302578963")
    yielded_pass = [item async for item in pipe_pass.process_message_event(ev_pass)]
    assert yielded_pass == []
    assert ev_pass.is_stopped() is False
    assert ev_pass._has_send_oper is False
    assert ctx_primary.sent_messages == []
    assert ev_pass.get_extra(EXTRA_HEAR_VOICE_STATUS) == "failed"
    assert ev_pass.get_extra(EXTRA_STT_STATUS) == "failed"
    assert ev_pass.get_extra(EXTRA_HEAR_VOICE_NON_HIJACKING) is True
    assert ev_pass.get_extra(EXTRA_STT_NON_HIJACKING) is True
    assert ev_pass.get_extra(EXTRA_HEAR_VOICE_HIJACKED) is False
    assert ev_pass.get_extra(EXTRA_STT_HIJACKED) is False
    assert _legacy_hijack_detected(ev_pass) is False
    await pipe_pass.shutdown()

    # 2) notify_pass primary path
    cfg_np = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "enable_fallback": False,
            "on_stt_fail": "notify_pass",
            "temp_cleanup_interval_sec": 0,
        }
    )
    pipe_np = HearVoicePipeline(ctx_primary, cfg_np, client=_FailingStubClient())
    ev_np = FakeEvent(messages=[Record(file=make_data_uri_wav())], sender_id="302578963")
    yielded_np = [item async for item in pipe_np.process_message_event(ev_np)]
    assert yielded_np == []
    assert ev_np.is_stopped() is False
    assert ev_np._has_send_oper is False
    assert len(ctx_primary.sent_messages) == 1
    assert ev_np.get_extra(EXTRA_HEAR_VOICE_NOTIFY_ONLY_SENT) is True
    assert ev_np.get_extra(EXTRA_STT_NOTIFY_ONLY_SENT) is True
    assert _legacy_hijack_detected(ev_np) is False
    await pipe_np.shutdown()

    # 3) notify_pass fallback path when context.send_message is unavailable
    ctx_fallback = types.SimpleNamespace(get_provider_by_id=lambda pid: FakeChatProvider())
    pipe_fallback = HearVoicePipeline(ctx_fallback, cfg_np, client=_FailingStubClient())
    ev_fallback = FakeEvent(messages=[Record(file=make_data_uri_wav())], sender_id="302578963")
    yielded_fb = [item async for item in pipe_fallback.process_message_event(ev_fallback)]
    assert yielded_fb == []
    assert ev_fallback.is_stopped() is False
    assert ev_fallback._has_send_oper is True
    assert ev_fallback.get_extra(EXTRA_HEAR_VOICE_NOTIFY_ONLY_SENT) is True
    assert ev_fallback.get_extra(EXTRA_STT_NOTIFY_ONLY_SENT) is True
    assert _legacy_hijack_detected(ev_fallback) is False
    await pipe_fallback.shutdown()


@pytest.mark.asyncio
async def test_companion_non_hijacking_contract_with_deepseek_fallback_and_cache_only():
    """Verify private, group passthrough, cache_only, and DeepSeek fallback maintain companion contract."""
    ctx = FakeContext(
        {
            "AlmoonsGemini-1": FakeChatProvider(),
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-9999",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
        }
    )

    class _Resp:
        def __init__(self, status: int, payload: dict[str, Any]):
            self.status = status
            self._raw = json.dumps(payload, ensure_ascii=False)

        async def text(self) -> str:
            return self._raw

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Sess:
        closed = False

        def post(self, url: str, **kwargs):
            if "generateContent" in url:
                return _Resp(400, {"error": {"message": "bad format"}})
            return _Resp(200, {"request_id": "ds-ok", "text": "晚上一起吃饭", "duration_ms": 500})

        async def close(self) -> None:
            self.closed = True

    for group_id, prob, exp_group, exp_should_reply, exp_cache_only, exp_reason in (
        ("", 1.0, False, True, False, "private_voice"),
        ("228397848", 1.0, True, True, False, "group_passthrough"),
        ("228397848", 0.0, True, False, True, "group_probability_miss"),
    ):
        cfg = HearVoiceConfig.from_mapping(
            {
                "provider_id": "AlmoonsGemini-1",
                "fallback_provider_id": "deepseek/deepseek-flash",
                "output_mode": "rich",
                "group_voice_reply_probability": prob,
                "temp_cleanup_interval_sec": 0,
            }
        )
        engine = HearVoiceEngine(ctx, cfg)
        engine._transport._session = _Sess()
        pipe = HearVoicePipeline(ctx, cfg, engine=engine)

        rec = Record(file=make_data_uri_wav())
        event = FakeEvent(
            messages=[rec],
            message_str="",
            group_id=group_id,
            sender_id="302578963",
            sender_name="Joseph",
        )
        async for _ in pipe.process_message_event(event):
            pass

        assert event.requested_llm is False
        assert event.is_stopped() is False
        assert event.get_extra(EXTRA_HEAR_VOICE_IS_GROUP) is exp_group
        assert event.get_extra(EXTRA_STT_IS_GROUP) is exp_group
        assert event.get_extra(EXTRA_HEAR_VOICE_SHOULD_REPLY) is exp_should_reply
        assert event.get_extra(EXTRA_STT_SHOULD_REPLY) is exp_should_reply
        assert event.get_extra(EXTRA_HEAR_VOICE_CACHE_ONLY) is exp_cache_only
        assert event.get_extra(EXTRA_STT_CACHE_ONLY) is exp_cache_only
        assert event.get_extra(EXTRA_HEAR_VOICE_REPLY_REASON) == exp_reason
        assert event.get_extra(EXTRA_HEAR_VOICE_ENGINE_USED) == "deepseek_asr"
        assert event.get_extra(EXTRA_HEAR_VOICE_FALLBACK_USED) is True
        assert _legacy_hijack_detected(event) is False

        if exp_cache_only:
            assert event.should_call_llm_calls == [True]
        else:
            assert event.should_call_llm_calls == []

        assert event.get_extra(EXTRA_HEAR_VOICE_TRANSCRIPT) == "晚上一起吃饭"
        assert event.get_extra(EXTRA_STT_TRANSCRIPT) == "晚上一起吃饭"
        entries = event.get_extra(EXTRA_COMPANION_TRANSCRIPTS)
        assert entries[0]["transcript"] == "晚上一起吃饭"
        assert entries[0]["engine"] == "deepseek_asr"
        assert entries[0]["fallback_used"] is True
        await pipe.shutdown()
