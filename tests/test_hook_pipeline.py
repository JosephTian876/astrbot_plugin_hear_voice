"""Unit tests for the two-stage non-hijacking multi-engine HearVoice pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Plain, Record, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

import main as plugin_main
from hear_voice.config import HearVoiceConfig
from hear_voice.engine import HearVoiceEngine
from hear_voice.hook_pipeline import (
    EXTRA_COMPANION_TRANSCRIPTS,
    EXTRA_HEAR_VOICE_AUDIO_CLEANED,
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
    EXTRA_STT_AUDIO_CLEANED,
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

RICH_SAMPLE = (
    "1) 原话转写：别生气啦，晚上一起去吃拉面嘛。\n"
    "2) 语气/情绪：软声哄人、带着撒娇和试探\n"
    "3) 说话人/音色：单人年轻女声，声线柔和，普通话\n"
    "4) 语速/停顿：前半句语速略快，句尾轻轻拉长\n"
    "5) 环境音：室内较安静，有轻微键盘敲击声\n"
    "6) 大意总结：温柔哄对方消气并约晚上吃拉面。"
)


class _StubClient:
    def __init__(self, response_text: str = RICH_SAMPLE, *, delay: float = 0.0, fail: bool = False):
        self.response_text = response_text
        self.delay = delay
        self.fail = fail
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self.closed = False
        self.last_error = ""

    async def transcribe(self, audio, *, instruction: str, endpoint) -> str:
        self.calls += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            if self.fail:
                self.last_error = self.last_error or "HTTP 400: INVALID_ARGUMENT: The attachment format is not supported"
                return ""
            return self.response_text
        finally:
            self.active -= 1

    async def normalize_transcript_with_model(self, transcript: str, *, endpoint) -> str:
        return transcript.rstrip("。") + "！"

    async def close(self) -> None:
        self.closed = True

    @staticmethod
    def safe_endpoint_summary(endpoint) -> str:
        return "stub-endpoint"


def _make_ctx() -> FakeContext:
    return FakeContext(
        {
            "AlmoonsGemini-1": FakeChatProvider(),
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-secret-8888",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
        }
    )


async def _collect_stage1(pipeline: HearVoicePipeline, event) -> list[Any]:
    results = []
    async for item in pipeline.process_message_event(event):
        results.append(item)
    return results


def test_plugin_hook_priorities_are_300000():
    msg_hooks = getattr(plugin_main.HearVoicePlugin.on_voice_message, "_astrbot_hooks", [])
    llm_hooks = getattr(plugin_main.HearVoicePlugin.on_llm_request, "_astrbot_hooks", [])
    assert msg_hooks[0][0] == "event_message_type"
    assert msg_hooks[0][2]["priority"] == 300000
    assert llm_hooks[0][0] == "on_llm_request"
    assert llm_hooks[0][2]["priority"] == 300000


@pytest.mark.asyncio
async def test_rich_mode_two_stage_non_hijacking_dual_extras_and_record_preserved():
    ctx = _make_ctx()
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "output_mode": "rich",
            "temp_cleanup_interval_sec": 0,
        }
    )
    client = _StubClient(RICH_SAMPLE)
    pipe = HearVoicePipeline(ctx, cfg, client=client)

    rec = Record(file=make_data_uri_wav())
    event = FakeEvent(messages=[rec], message_str="", group_id="55555", sender_id="10001")

    yielded = await _collect_stage1(pipe, event)
    assert yielded == []
    assert event.requested_llm is False
    assert event.is_stopped() is False
    assert event.message_obj.message[0] is rec
    assert isinstance(event.message_obj.message[1], Plain)
    assert event.message_obj.message[1].text == "[语音消息]: 别生气啦，晚上一起去吃拉面嘛。"
    assert event.message_str == "[语音消息]: 别生气啦，晚上一起去吃拉面嘛。"
    assert event.message_obj.message_str == "[语音消息]: 别生气啦，晚上一起去吃拉面嘛。"

    # Canonical _hear_voice_* AND alias _gemini_stt_* keys both exported
    assert event.get_extra(EXTRA_HEAR_VOICE_TRANSCRIPT) == "别生气啦，晚上一起去吃拉面嘛。"
    assert event.get_extra(EXTRA_STT_TRANSCRIPT) == "别生气啦，晚上一起去吃拉面嘛。"
    assert event.get_extra(EXTRA_HEAR_VOICE_RAW_TEXT) == RICH_SAMPLE
    assert event.get_extra(EXTRA_STT_RAW_TEXT) == RICH_SAMPLE
    assert "语气/情绪" not in event.get_extra(EXTRA_HEAR_VOICE_TRANSCRIPT)
    assert event.get_extra(EXTRA_HEAR_VOICE_DONE) is True
    assert event.get_extra(EXTRA_STT_DONE) is True
    assert event.get_extra(EXTRA_HEAR_VOICE_STATUS) == "ok"
    assert event.get_extra(EXTRA_STT_STATUS) == "ok"
    assert event.get_extra(EXTRA_HEAR_VOICE_IS_GROUP) is True
    assert event.get_extra(EXTRA_STT_IS_GROUP) is True
    assert event.get_extra(EXTRA_HEAR_VOICE_SHOULD_REPLY) is True
    assert event.get_extra(EXTRA_STT_SHOULD_REPLY) is True
    assert event.get_extra(EXTRA_HEAR_VOICE_CACHE_ONLY) is False
    assert event.get_extra(EXTRA_STT_CACHE_ONLY) is False
    assert event.get_extra(EXTRA_HEAR_VOICE_REPLY_REASON) == "group_passthrough"
    assert event.get_extra(EXTRA_STT_REPLY_REASON) == "group_passthrough"
    assert event.get_extra(EXTRA_HEAR_VOICE_NON_HIJACKING) is True
    assert event.get_extra(EXTRA_STT_NON_HIJACKING) is True
    assert event.get_extra(EXTRA_HEAR_VOICE_HIJACKED) is False
    assert event.get_extra(EXTRA_STT_HIJACKED) is False
    assert event.get_extra(EXTRA_HEAR_VOICE_ENGINE_USED) == "gemini"
    assert event.get_extra(EXTRA_HEAR_VOICE_FALLBACK_USED) is False

    req = ProviderRequest(
        prompt="<attachment>",
        audio_urls=["C:/temp/voice.wav"],
        extra_user_content_parts=[
            TextPart(text="[Audio Attachment: path C:/temp/voice.wav]"),
            TextPart(text="[Audio Attachment in quoted message: path C:/temp/q.wav]"),
            TextPart(text="[Voice unavailable]"),
            TextPart(text="keep me"),
        ],
        contexts=[
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAA"}},
                    {"type": "text", "text": "[Audio Attachment: path C:/temp/old.wav]"},
                    {"type": "text", "text": "旧文本"},
                ],
            }
        ],
    )
    await pipe.process_llm_request(event, req)

    assert req.audio_urls == []
    assert event.get_extra(EXTRA_HEAR_VOICE_AUDIO_CLEANED) is True
    assert event.get_extra(EXTRA_STT_AUDIO_CLEANED) is True
    assert req.prompt == "[语音消息]: 别生气啦，晚上一起去吃拉面嘛。"
    texts = [getattr(p, "text", "") for p in req.extra_user_content_parts]
    assert "keep me" in texts
    assert not any("Audio Attachment" in t or "Voice unavailable" in t for t in texts)
    injected = [p for p in req.extra_user_content_parts if "系统临时语音声学与上下文感知" in p.text]
    assert len(injected) == 1
    assert injected[0]._no_save is True
    assert "软声哄人" in injected[0].text
    assert req.contexts[0]["content"] == [{"type": "text", "text": "旧文本"}]
    await pipe.shutdown()


@pytest.mark.asyncio
async def test_pure_stt_deepseek_fallback_under_rich_mode_emits_no_template_or_empty_temp_part():
    ctx = _make_ctx()
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "fallback_provider_id": "deepseek/deepseek-flash",
            "output_mode": "rich",
            "retry_times": 0,
            "temp_cleanup_interval_sec": 0,
        }
    )

    class _MockResp:
        def __init__(self, status: int, payload: dict[str, Any]):
            self.status = status
            self._raw = json.dumps(payload, ensure_ascii=False)

        async def text(self) -> str:
            return self._raw

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _FailoverSess:
        def __init__(self) -> None:
            self.closed = False

        def post(self, url: str, **kwargs):
            if "generateContent" in url:
                return _MockResp(503, {"error": {"message": "gemini down"}})
            if "/v1/tools/asr" in url:
                return _MockResp(200, {"request_id": "ds-1", "text": "今晚吃寿喜烧", "duration_ms": 900})
            raise AssertionError(url)

        async def close(self) -> None:
            self.closed = True

    engine = HearVoiceEngine(ctx, cfg)
    engine._transport._session = _FailoverSess()
    pipe = HearVoicePipeline(ctx, cfg, engine=engine)

    rec = Record(file=make_data_uri_wav())
    ev = FakeEvent(messages=[rec], group_id="666")
    await _collect_stage1(pipe, ev)

    assert ev.get_extra(EXTRA_HEAR_VOICE_ENGINE_USED) == "deepseek_asr"
    assert ev.get_extra(EXTRA_HEAR_VOICE_FALLBACK_USED) is True
    assert ev.get_extra(EXTRA_HEAR_VOICE_TRANSCRIPT) == "今晚吃寿喜烧"
    assert ev.get_extra(EXTRA_HEAR_VOICE_RAW_TEXT) == "今晚吃寿喜烧"
    assert ev.get_extra(EXTRA_STT_TRANSCRIPT) == "今晚吃寿喜烧"
    assert ev.get_extra(EXTRA_STT_RAW_TEXT) == "今晚吃寿喜烧"
    assert ev.message_str == "[语音消息]: 今晚吃寿喜烧"
    entry = ev.get_extra(EXTRA_COMPANION_TRANSCRIPTS)[0]
    assert entry["engine"] == "deepseek_asr"
    assert entry["fallback_used"] is True
    assert entry["mode"] == "simple"
    assert entry["rich_text"] == ""

    # Stage 2 cleans audio attachments & sets prompt, WITHOUT injecting any bogus temp TextPart
    req = ProviderRequest(
        prompt="<attachment>",
        audio_urls=["/tmp/voice.wav"],
        extra_user_content_parts=[TextPart(text="[Audio Attachment: path /tmp/voice.wav]")],
        contexts=[],
    )
    await pipe.process_llm_request(ev, req)
    assert req.audio_urls == []
    assert req.prompt == "[语音消息]: 今晚吃寿喜烧"
    assert req.extra_user_content_parts == []
    assert req.contexts == []
    await pipe.shutdown()


@pytest.mark.asyncio
async def test_default_on_stt_fail_pass_is_silent_zero_qq_messages_and_logs_redacted_warning(caplog):
    ctx = _make_ctx()
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "enable_fallback": False,
            "temp_cleanup_interval_sec": 0,
        }
    )
    assert cfg.on_stt_fail == "pass"

    client = _StubClient(fail=True)
    client.last_error = "HTTP 401: invalid key test-secret-key-1234"
    pipe = HearVoicePipeline(ctx, cfg, client=client)

    ev = FakeEvent(messages=[Record(file=make_data_uri_wav())], group_id="777")
    with caplog.at_level(logging.WARNING):
        yielded = await _collect_stage1(pipe, ev)

    assert yielded == []
    assert ctx.sent_messages == []
    assert ev.sent_chains == []
    assert ev._has_send_oper is False
    assert ev.is_stopped() is False
    assert ev.get_extra(EXTRA_HEAR_VOICE_STATUS) == "failed"
    assert ev.get_extra(EXTRA_STT_STATUS) == "failed"
    assert ev.get_extra(EXTRA_HEAR_VOICE_NON_HIJACKING) is True
    assert ev.get_extra(EXTRA_STT_NON_HIJACKING) is True
    assert ev.get_extra(EXTRA_HEAR_VOICE_HIJACKED) is False
    assert ev.get_extra(EXTRA_STT_HIJACKED) is False
    assert ev.get_extra(EXTRA_HEAR_VOICE_NOTIFY_ONLY_SENT) is None
    assert "[HearVoice]" in caplog.text
    assert "401" in caplog.text
    assert "test-secret-key-1234" not in caplog.text
    await pipe.shutdown()


@pytest.mark.asyncio
async def test_simple_mode_and_text_normalize_and_quoted_voice():
    ctx = _make_ctx()
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "output_mode": "simple",
            "enable_text_normalize": True,
            "temp_cleanup_interval_sec": 0,
        }
    )
    client = _StubClient("你好呀。")
    pipe = HearVoicePipeline(ctx, cfg, client=client)

    qrec = Record(file=make_data_uri_wav())
    reply = Reply(id="77", chain=[qrec])
    drec = Record(file=make_data_uri_wav())
    event = FakeEvent(messages=[reply, Plain("和纱听下"), drec], message_str="和纱听下")

    await _collect_stage1(pipe, event)
    assert event.requested_llm is False
    assert event.is_stopped() is False
    assert event.message_obj.message[0] is reply
    assert event.message_obj.message[2] is drec
    assert event.get_extra(EXTRA_HEAR_VOICE_TRANSCRIPT) == "你好呀！\n你好呀！"
    assert event.get_extra(EXTRA_STT_TRANSCRIPT) == "你好呀！\n你好呀！"
    assert event.get_extra(EXTRA_HEAR_VOICE_IS_GROUP) is False
    assert event.get_extra(EXTRA_HEAR_VOICE_SHOULD_REPLY) is True
    assert event.get_extra(EXTRA_HEAR_VOICE_CACHE_ONLY) is False
    assert event.get_extra(EXTRA_HEAR_VOICE_REPLY_REASON) == "private_voice"
    assert event.get_extra(EXTRA_HEAR_VOICE_NON_HIJACKING) is True
    assert event.get_extra(EXTRA_HEAR_VOICE_HIJACKED) is False
    assert "[语音消息]: 你好呀！" in event.message_str
    assert "[引用语音消息]: 你好呀！" in event.message_str
    assert event.message_str.startswith("和纱听下\n")
    await pipe.shutdown()


@pytest.mark.asyncio
async def test_whitelists_and_private_group_toggles():
    ctx = _make_ctx()
    client = _StubClient("测试")

    pipe_no_priv = HearVoicePipeline(
        ctx,
        HearVoiceConfig.from_mapping({"provider_id": "AlmoonsGemini-1", "enable_private": False}),
        client=client,
    )
    ev_priv = FakeEvent(messages=[Record(file=make_data_uri_wav())], group_id="")
    await _collect_stage1(pipe_no_priv, ev_priv)
    assert client.calls == 0

    pipe_wl = HearVoicePipeline(
        ctx,
        HearVoiceConfig.from_mapping(
            {
                "provider_id": "AlmoonsGemini-1",
                "group_whitelist": ["111"],
                "user_whitelist": ["u1"],
                "temp_cleanup_interval_sec": 0,
            }
        ),
        client=client,
    )
    await _collect_stage1(
        pipe_wl,
        FakeEvent(messages=[Record(file=make_data_uri_wav())], group_id="222", sender_id="u1"),
    )
    await _collect_stage1(
        pipe_wl,
        FakeEvent(messages=[Record(file=make_data_uri_wav())], group_id="111", sender_id="u2"),
    )
    assert client.calls == 0

    ev_ok = FakeEvent(messages=[Record(file=make_data_uri_wav())], group_id="111", sender_id="u1")
    await _collect_stage1(pipe_wl, ev_ok)
    assert client.calls == 1
    assert ev_ok.get_extra(EXTRA_HEAR_VOICE_DONE) is True
    assert ev_ok.get_extra(EXTRA_STT_DONE) is True
    await pipe_no_priv.shutdown()
    await pipe_wl.shutdown()


@pytest.mark.asyncio
async def test_skip_already_transcribed_and_non_voice_and_commands():
    ctx = _make_ctx()
    client = _StubClient("不会再调用")
    pipe = HearVoicePipeline(
        ctx,
        HearVoiceConfig.from_mapping({"provider_id": "AlmoonsGemini-1", "temp_cleanup_interval_sec": 0}),
        client=client,
    )

    ev_text = FakeEvent(messages=[Plain("普通文本")], message_str="普通文本")
    await _collect_stage1(pipe, ev_text)
    assert client.calls == 0

    ev_done = FakeEvent(messages=[Record(file=make_data_uri_wav())])
    ev_done.set_extra(EXTRA_COMPANION_TRANSCRIPTS, [{"transcript": "已有"}])
    await _collect_stage1(pipe, ev_done)
    assert client.calls == 0

    ev_done_hv = FakeEvent(messages=[Record(file=make_data_uri_wav())])
    ev_done_hv.set_extra(EXTRA_HEAR_VOICE_DONE, True)
    await _collect_stage1(pipe, ev_done_hv)
    assert client.calls == 0

    from astrbot.api.event import filter as stub_filter

    class CommandFilter:
        pass

    CommandFilter.__module__ = "astrbot.core.star.filter.command"
    ev_cmd = FakeEvent(messages=[Record(file=make_data_uri_wav())])
    ev_cmd.set_extra(
        "activated_handlers",
        [types.SimpleNamespace(handler_module_path="other_plugin.main", event_filters=[CommandFilter()])],
    )
    await _collect_stage1(pipe, ev_cmd)
    assert client.calls == 0
    assert stub_filter is not None
    await pipe.shutdown()


@pytest.mark.asyncio
async def test_concurrency_semaphore_caps_parallel_requests():
    ctx = _make_ctx()
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "stt_max_concurrency": 2,
            "output_mode": "simple",
            "temp_cleanup_interval_sec": 0,
        }
    )
    client = _StubClient("并发测试", delay=0.05)
    pipe = HearVoicePipeline(ctx, cfg, client=client)

    events = [FakeEvent(messages=[Record(file=make_data_uri_wav())], sender_id=str(i)) for i in range(6)]
    await asyncio.gather(*(_collect_stage1(pipe, ev) for ev in events))

    assert client.calls == 6
    assert client.peak_active <= 2
    await pipe.shutdown()


@pytest.mark.asyncio
async def test_failure_policies_pass_block_notify_notify_pass(caplog):
    for policy, expect_stopped, expect_yield, expect_sent in (
        ("pass", False, False, False),
        ("block", True, False, False),
        ("notify", True, True, False),
        ("notify_pass", False, False, True),
    ):
        ctx = _make_ctx()
        client = _StubClient(fail=True)
        pipe = HearVoicePipeline(
            ctx,
            HearVoiceConfig.from_mapping(
                {
                    "provider_id": "AlmoonsGemini-1",
                    "enable_fallback": False,
                    "on_stt_fail": policy,
                    "temp_cleanup_interval_sec": 0,
                }
            ),
            client=client,
        )
        ev = FakeEvent(messages=[Record(file=make_data_uri_wav())])
        with caplog.at_level(logging.WARNING):
            yielded = await _collect_stage1(pipe, ev)
        assert ev.is_stopped() is expect_stopped, policy
        assert bool(yielded) is expect_yield, policy
        assert ev.get_extra(EXTRA_HEAR_VOICE_STATUS) == "failed", policy
        assert ev.get_extra(EXTRA_STT_STATUS) == "failed", policy
        assert bool(ctx.sent_messages) is expect_sent, policy
        assert ev._has_send_oper is False, policy
        if policy in ("pass", "notify_pass"):
            assert ev.get_extra(EXTRA_HEAR_VOICE_NON_HIJACKING) is True, policy
            assert ev.get_extra(EXTRA_HEAR_VOICE_HIJACKED) is False, policy
            assert ev.get_extra(EXTRA_STT_NON_HIJACKING) is True, policy
            assert ev.get_extra(EXTRA_STT_HIJACKED) is False, policy
        if policy == "notify_pass":
            assert ev.get_extra(EXTRA_HEAR_VOICE_NOTIFY_ONLY_SENT) is True
            assert ev.get_extra(EXTRA_STT_NOTIFY_ONLY_SENT) is True
            session, chain = ctx.sent_messages[0]
            assert session == ev.unified_msg_origin
            assert "语音识别失败" in chain.chain[0].text
        assert "HTTP 400" in caplog.text
        caplog.clear()
        await pipe.shutdown()


@pytest.mark.asyncio
async def test_notify_pass_falls_back_to_event_send_when_send_message_unavailable():
    ctx_no_send = types.SimpleNamespace(
        get_provider_by_id=lambda pid: FakeChatProvider(),
    )
    pipe1 = HearVoicePipeline(
        ctx_no_send,
        HearVoiceConfig.from_mapping(
            {
                "provider_id": "AlmoonsGemini-1",
                "enable_fallback": False,
                "on_stt_fail": "notify_pass",
                "temp_cleanup_interval_sec": 0,
            }
        ),
        client=_StubClient(fail=True),
    )
    ev1 = FakeEvent(messages=[Record(file=make_data_uri_wav())])
    yielded1 = await _collect_stage1(pipe1, ev1)
    assert yielded1 == []
    assert ev1.is_stopped() is False
    assert ev1._has_send_oper is True
    assert len(ev1.sent_chains) == 1
    assert "语音识别失败" in ev1.sent_chains[0].chain[0].text
    assert ev1.get_extra(EXTRA_HEAR_VOICE_STATUS) == "failed"
    assert ev1.get_extra(EXTRA_HEAR_VOICE_NON_HIJACKING) is True
    assert ev1.get_extra(EXTRA_HEAR_VOICE_HIJACKED) is False
    assert ev1.get_extra(EXTRA_HEAR_VOICE_NOTIFY_ONLY_SENT) is True
    assert ev1.get_extra(EXTRA_STT_NOTIFY_ONLY_SENT) is True
    await pipe1.shutdown()

    ctx_false = _make_ctx()
    ctx_false.send_message = AsyncMock(return_value=False)  # type: ignore[method-assign]
    pipe2 = HearVoicePipeline(
        ctx_false,
        HearVoiceConfig.from_mapping(
            {
                "provider_id": "AlmoonsGemini-1",
                "enable_fallback": False,
                "on_stt_fail": "notify_pass",
                "temp_cleanup_interval_sec": 0,
            }
        ),
        client=_StubClient(fail=True),
    )
    ev2 = FakeEvent(messages=[Record(file=make_data_uri_wav())])
    yielded2 = await _collect_stage1(pipe2, ev2)
    assert yielded2 == []
    assert ev2.is_stopped() is False
    assert ev2._has_send_oper is True
    assert ev2.get_extra(EXTRA_HEAR_VOICE_NON_HIJACKING) is True
    assert ev2.get_extra(EXTRA_HEAR_VOICE_HIJACKED) is False
    assert ev2.get_extra(EXTRA_HEAR_VOICE_NOTIFY_ONLY_SENT) is True
    await pipe2.shutdown()


@pytest.mark.asyncio
async def test_group_probability_zero_caches_context_without_stopping_event():
    ctx = _make_ctx()
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "output_mode": "simple",
            "group_voice_reply_probability": 0.0,
            "temp_cleanup_interval_sec": 0,
        }
    )
    pipe = HearVoicePipeline(ctx, cfg, client=_StubClient("路人发言"))
    ev = FakeEvent(messages=[Record(file=make_data_uri_wav())], group_id="888")

    await _collect_stage1(pipe, ev)
    assert ev.requested_llm is False
    assert ev.is_stopped() is False
    assert ev.get_extra(EXTRA_HEAR_VOICE_DONE) is True
    assert ev.get_extra(EXTRA_HEAR_VOICE_SHOULD_REPLY) is False
    assert ev.get_extra(EXTRA_HEAR_VOICE_CACHE_ONLY) is True
    assert ev.get_extra(EXTRA_STT_SHOULD_REPLY) is False
    assert ev.get_extra(EXTRA_STT_CACHE_ONLY) is True
    assert ev.get_extra(EXTRA_HEAR_VOICE_REPLY_REASON) == "group_probability_miss"
    assert ev.get_extra(EXTRA_HEAR_VOICE_NON_HIJACKING) is True
    assert ev.get_extra(EXTRA_HEAR_VOICE_HIJACKED) is False
    assert ev.should_call_llm_calls == [True]
    assert ev.get_extra(EXTRA_HEAR_VOICE_TRANSCRIPT) == "路人发言"
    assert ev.get_extra(EXTRA_COMPANION_TRANSCRIPTS)[0]["transcript"] == "路人发言"
    await pipe.shutdown()


@pytest.mark.asyncio
async def test_stage2_contexts_fallback_when_extra_parts_missing():
    ctx = _make_ctx()
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "output_mode": "rich",
            "temp_cleanup_interval_sec": 0,
        }
    )
    pipe = HearVoicePipeline(ctx, cfg, client=_StubClient(RICH_SAMPLE))
    ev = FakeEvent(messages=[Record(file=make_data_uri_wav())])
    await _collect_stage1(pipe, ev)

    req_minimal = types.SimpleNamespace(
        prompt="<attachment>",
        audio_urls=["/tmp/voice.wav"],
        contexts=[{"role": "user", "content": "[Audio Attachment: path /tmp/voice.wav]"}],
    )
    await pipe.process_llm_request(ev, req_minimal)
    assert req_minimal.audio_urls == []
    assert req_minimal.prompt == "[语音消息]: 别生气啦，晚上一起去吃拉面嘛。"
    assert req_minimal.contexts[0]["content"] == "[语音消息]"
    assert req_minimal.contexts[-1]["_no_save"] is True
    assert "软声哄人" in req_minimal.contexts[-1]["content"]
    await pipe.shutdown()


@pytest.mark.asyncio
async def test_plugin_shell_lifecycle_and_terminate():
    ctx = _make_ctx()
    plugin = plugin_main.HearVoicePlugin(
        ctx,
        {
            "provider_id": "AlmoonsGemini-1",
            "fallback_provider_id": "deepseek/deepseek-flash",
            "register_stt_provider": True,
            "stt_provider_id": "hear_voice_stt_shell",
            "temp_cleanup_interval_sec": 0,
        },
    )
    assert "hear_voice_stt_shell" in ctx.provider_manager.inst_map
    await plugin.terminate()
    assert "hear_voice_stt_shell" not in ctx.provider_manager.inst_map


@pytest.mark.asyncio
async def test_production_info_summary_logged_redacted_when_debug_mode_false(caplog):
    ctx = _make_ctx()
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "output_mode": "simple",
            "debug_mode": False,
            "handle_quoted_voice": True,
            "temp_cleanup_interval_sec": 0,
        }
    )
    secret_transcript = "绝密口令九五二七不要写进日志"
    pipe = HearVoicePipeline(ctx, cfg, client=_StubClient(secret_transcript))

    # Direct voice with debug_mode=False
    ev_direct = FakeEvent(messages=[Record(file=make_data_uri_wav())])
    with caplog.at_level(logging.INFO):
        await _collect_stage1(pipe, ev_direct)

    info_lines = [r.message for r in caplog.records if r.levelno == logging.INFO and "语音转写完成" in r.message]
    assert len(info_lines) == 1
    line = info_lines[0]
    assert "[HearVoice] 语音转写完成" in line
    assert "engine=gemini" in line
    assert "fallback=false" in line
    assert "source=direct" in line
    assert "count=1" in line
    assert f"chars={len(secret_transcript)}" in line
    assert "elapsed_ms=" in line
    # Strictly redacted: no raw transcript content, no secrets, no file paths.
    assert secret_transcript not in line
    assert "sk-ds-secret-8888" not in line
    assert "data:audio" not in line

    # Quoted voice also emits concise redacted INFO summary with source=quoted
    caplog.clear()
    quoted_reply = Reply(id="q-99", chain=[Record(file=make_data_uri_wav())])
    ev_quoted = FakeEvent(messages=[quoted_reply, Plain("他说了啥")])
    with caplog.at_level(logging.INFO):
        await _collect_stage1(pipe, ev_quoted)

    quoted_lines = [r.message for r in caplog.records if r.levelno == logging.INFO and "语音转写完成" in r.message]
    assert len(quoted_lines) == 1
    qline = quoted_lines[0]
    assert "source=quoted" in qline
    assert "count=1" in qline
    assert secret_transcript not in qline
    await pipe.shutdown()
