"""Unit tests for unified SttEndpointSpec, resolve_engine_kind, and Primary->Fallback engine."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any
from unittest.mock import patch

import pytest

from hear_voice.audio_resolver import ResolvedAudio
from hear_voice.config import HearVoiceConfig
from hear_voice.engine import (
    HearVoiceEngine,
    resolve_endpoints,
    resolve_engine_kind,
)
from tests.conftest import FakeChatProvider, FakeContext, make_wav_bytes


def test_resolve_engine_kind_auto_detection_and_explicit_overrides():
    assert (
        resolve_engine_kind(
            "auto",
            provider_type="openai_chat_completion",
            api_base="https://api.deepseek.com/v1",
            provider_id="deepseek/deepseek-flash",
            source_id="deepseek",
        )
        == "deepseek_asr"
    )
    assert (
        resolve_engine_kind(
            "auto",
            provider_type="openai_chat_completion",
            api_base="https://proxy.example.com/v1/tools/asr",
            provider_id="custom",
        )
        == "deepseek_asr"
    )
    assert (
        resolve_engine_kind(
            "auto",
            provider_type="openai_whisper_api",
            api_base="https://api.openai.com/v1",
        )
        == "openai_whisper"
    )
    assert (
        resolve_engine_kind(
            "auto",
            provider_type="googlegenai_chat_completion",
            api_base="https://api.starbridge.zone",
            provider_id="AlmoonsGemini-1",
        )
        == "gemini"
    )
    # Explicit engine overrides auto-detection hints
    assert (
        resolve_engine_kind(
            "gemini",
            api_base="https://api.deepseek.com/v1",
            provider_id="deepseek/deepseek-flash",
        )
        == "gemini"
    )
    assert resolve_engine_kind("deepseek_asr", api_base="https://custom.example.com") == "deepseek_asr"


@pytest.mark.asyncio
async def test_resolve_primary_and_fallback_endpoints_from_providers():
    ctx = FakeContext(
        {
            "AlmoonsGemini-1": FakeChatProvider(
                provider_id="AlmoonsGemini-1",
                provider_type="googlegenai_chat_completion",
                api_base="https://api.starbridge.zone",
                api_key="sk-gem-key-1111",
                model="［抗截断］gemini-3.8-flash",
            ),
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-key-2222",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
        }
    )
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "fallback_provider_id": "deepseek/deepseek-flash",
        }
    )
    primary, fallback = await resolve_endpoints(ctx, cfg)
    assert primary is not None
    assert primary.slot == "primary"
    assert primary.engine_kind == "gemini"
    assert primary.model == "［抗截断］gemini-3.8-flash"

    assert fallback is not None
    assert fallback.slot == "fallback"
    assert fallback.engine_kind == "deepseek_asr"
    assert fallback.api_base == "https://api.deepseek.com/v1/tools/asr"
    assert fallback.api_key == "sk-ds-key-2222"
    assert fallback.is_pure_stt is True


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


class _MultiRouteMockSession:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def post(self, url: str, *, headers=None, json=None, data=None, proxy=None, ssl=None):
        call = {"url": url, "headers": headers, "json": json, "data": data}
        self.calls.append(call)
        return self._handler(call)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_primary_gemini_fails_seamlessly_falls_back_to_deepseek_asr(caplog):
    ctx = FakeContext(
        {
            "AlmoonsGemini-1": FakeChatProvider(
                provider_id="AlmoonsGemini-1",
                api_base="https://api.starbridge.zone",
                api_key="sk-gem-secret-1111",
                model="gemini-3.8-flash",
            ),
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-secret-2222",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
        }
    )
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "fallback_provider_id": "deepseek/deepseek-flash",
            "output_mode": "rich",
            "retry_times": 1,
        }
    )
    engine = HearVoiceEngine(ctx, cfg)
    wav = make_wav_bytes(duration_ms=100, sample_rate=16000)
    audio = ResolvedAudio(
        base64_data=base64.b64encode(wav).decode("ascii"),
        mime_type="audio/wav",
        audio_format="wav",
        size_bytes=len(wav),
        raw_bytes=wav,
    )

    def _route(call: dict[str, Any]):
        if "generateContent" in call["url"]:
            return _MockResp(400, {"error": {"code": 400, "message": "audio disabled"}})
        if "/v1/tools/asr" in call["url"]:
            return _MockResp(200, {"request_id": "r1", "text": "明天早上八点见", "duration_ms": 100})
        raise AssertionError(f"Unexpected URL: {call['url']}")

    sess = _MultiRouteMockSession(_route)
    engine._transport._session = sess

    with caplog.at_level(logging.WARNING), patch("hear_voice.engine.asyncio.sleep", return_value=None):
        res, status = await engine.transcribe_audio(audio)

    assert status == "ok"
    assert res is not None
    assert res.engine_used == "deepseek_asr"
    assert res.fallback_used is True
    assert res.is_rich is False
    assert res.transcript == "明天早上八点见"
    assert res.raw_text == "明天早上八点见"
    assert "原话转写" not in res.raw_text
    assert len(sess.calls) == 2
    assert "主识别失败" in caplog.text
    assert "自动切换回退通道" in caplog.text
    assert "sk-gem-secret-1111" not in caplog.text
    assert "sk-ds-secret-2222" not in caplog.text
    await engine.close()


@pytest.mark.asyncio
async def test_primary_deepseek_fails_falls_back_to_gemini_rich_and_text_normalize():
    ctx = FakeContext(
        {
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-primary-1111",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
            "AlmoonsGemini-1": FakeChatProvider(
                provider_id="AlmoonsGemini-1",
                api_base="https://api.starbridge.zone",
                api_key="sk-gem-backup-2222",
                model="gemini-3.8-flash",
            ),
        }
    )
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "deepseek/deepseek-flash",
            "fallback_provider_id": "AlmoonsGemini-1",
            "output_mode": "rich",
            "retry_times": 0,
        }
    )
    engine = HearVoiceEngine(ctx, cfg)
    wav = make_wav_bytes(duration_ms=100, sample_rate=16000)
    audio = ResolvedAudio(
        base64_data=base64.b64encode(wav).decode("ascii"),
        mime_type="audio/wav",
        audio_format="wav",
        size_bytes=len(wav),
        raw_bytes=wav,
    )

    rich_reply = (
        "1) 原话转写：别忘了带钥匙\n"
        "2) 语气/情绪：温柔提醒\n"
        "3) 说话人/音色：女声\n"
        "4) 语速/停顿：适中\n"
        "5) 环境音：安静\n"
        "6) 大意总结：提醒带钥匙"
    )

    def _route(call: dict[str, Any]):
        if "/v1/tools/asr" in call["url"]:
            return _MockResp(503, {"error": {"message": "ds busy"}})
        if "generateContent" in call["url"]:
            return _MockResp(200, {"candidates": [{"content": {"parts": [{"text": rich_reply}]}}]})
        raise AssertionError(f"Unexpected URL: {call['url']}")

    sess = _MultiRouteMockSession(_route)
    engine._transport._session = sess

    res, status = await engine.transcribe_audio(audio)
    assert status == "ok"
    assert res is not None
    assert res.engine_used == "gemini"
    assert res.fallback_used is True
    assert res.is_rich is True
    assert res.transcript == "别忘了带钥匙"
    assert res.rich_result.emotion == "温柔提醒"
    await engine.close()


@pytest.mark.asyncio
async def test_deepseek_primary_with_enable_text_normalize_and_openai_whisper_engine():
    ctx = FakeContext(
        {
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-primary-3333",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
            "whisper-1": FakeChatProvider(
                provider_id="whisper-1",
                provider_type="openai_whisper_api",
                api_base="https://api.openai.com/v1",
                api_key="sk-wh-4444",
                model="whisper-1",
            ),
        }
    )
    wav = make_wav_bytes(duration_ms=100, sample_rate=16000)
    audio = ResolvedAudio(
        base64_data=base64.b64encode(wav).decode("ascii"),
        mime_type="audio/wav",
        audio_format="wav",
        size_bytes=len(wav),
        raw_bytes=wav,
    )

    # 1) DeepSeek primary + enable_text_normalize=True uses provider's chat completion model
    cfg_ds = HearVoiceConfig.from_mapping(
        {
            "provider_id": "deepseek/deepseek-flash",
            "enable_fallback": False,
            "enable_text_normalize": True,
        }
    )
    engine_ds = HearVoiceEngine(ctx, cfg_ds)

    def _ds_route(call: dict[str, Any]):
        if "/v1/tools/asr" in call["url"]:
            return _MockResp(200, {"request_id": "r1", "text": "明天一起吃饭吧", "duration_ms": 100})
        if "/chat/completions" in call["url"]:
            return _MockResp(200, {"choices": [{"message": {"content": "明天一起吃饭吧！"}}]})
        raise AssertionError(f"Unexpected URL: {call['url']}")

    engine_ds._transport._session = _MultiRouteMockSession(_ds_route)
    res_ds, _ = await engine_ds.transcribe_audio(audio)
    assert res_ds is not None
    assert res_ds.engine_used == "deepseek_asr"
    assert res_ds.transcript == "明天一起吃饭吧！"
    assert res_ds.is_rich is False
    await engine_ds.close()

    # 2) OpenAI Whisper primary
    cfg_wh = HearVoiceConfig.from_mapping(
        {
            "provider_id": "whisper-1",
            "enable_fallback": False,
            "output_mode": "rich",
        }
    )
    engine_wh = HearVoiceEngine(ctx, cfg_wh)

    def _wh_route(call: dict[str, Any]):
        assert "/v1/audio/transcriptions" in call["url"]
        return _MockResp(200, {"text": "你好 Whisper"})

    engine_wh._transport._session = _MultiRouteMockSession(_wh_route)
    res_wh, _ = await engine_wh.transcribe_audio(audio)
    assert res_wh is not None
    assert res_wh.engine_used == "openai_whisper"
    assert res_wh.transcript == "你好 Whisper"
    assert res_wh.is_rich is False
    await engine_wh.close()


@pytest.mark.asyncio
async def test_merged_provider_source_and_disable_fallback_and_fallback_only():
    ctx = FakeContext(
        {
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="",
                api_key="sk-ds-merged-1111",
                model="deepseek-flash",
            ),
            "AlmoonsGemini-1": FakeChatProvider(
                provider_id="AlmoonsGemini-1",
                api_base="https://api.starbridge.zone",
                api_key="sk-gem-1111",
                model="gemini-3.8-flash",
            ),
        }
    )
    ctx.provider_manager.get_provider_config_by_id = lambda pid, merged=False: (
        {
            "id": pid,
            "type": "openai_chat_completion",
            "provider_source_id": "deepseek",
            "api_base": "https://api.deepseek.com/v1",
        }
        if pid == "deepseek/deepseek-flash" and merged
        else None
    )

    # 1) Merged provider_source api_base resolves deepseek_asr cleanly
    cfg_merged = HearVoiceConfig.from_mapping({"provider_id": "deepseek/deepseek-flash"})
    p1, _ = await resolve_endpoints(ctx, cfg_merged)
    assert p1 is not None
    assert p1.engine_kind == "deepseek_asr"
    assert p1.api_base == "https://api.deepseek.com/v1/tools/asr"

    # 2) enable_fallback=False ignores configured fallback slot
    cfg_no_fb = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "enable_fallback": False,
            "fallback_provider_id": "deepseek/deepseek-flash",
        }
    )
    p2, fb2 = await resolve_endpoints(ctx, cfg_no_fb)
    assert p2 is not None and fb2 is None

    # 3) Primary unconfigured + Fallback configured -> routes straight to Fallback
    cfg_fb_only = HearVoiceConfig.from_mapping(
        {
            "provider_id": "",
            "fallback_provider_id": "deepseek/deepseek-flash",
        }
    )
    engine_fb = HearVoiceEngine(ctx, cfg_fb_only)
    engine_fb._transport._session = _MultiRouteMockSession(
        lambda call: _MockResp(200, {"request_id": "fb1", "text": "仅备通道生效", "duration_ms": 50})
    )
    wav = make_wav_bytes(duration_ms=80, sample_rate=16000)
    audio = ResolvedAudio(
        base64_data=base64.b64encode(wav).decode("ascii"),
        mime_type="audio/wav",
        audio_format="wav",
        size_bytes=len(wav),
        raw_bytes=wav,
    )
    res_fb, status_fb = await engine_fb.transcribe_audio(audio)
    assert status_fb == "ok"
    assert res_fb is not None
    assert res_fb.engine_used == "deepseek_asr"
    assert res_fb.fallback_used is True
    assert res_fb.transcript == "仅备通道生效"
    await engine_fb.close()


@pytest.mark.asyncio
async def test_gemini_400_unsupported_format_transcodes_wav_and_retries_once_successfully(caplog):
    ctx = FakeContext(
        {
            "AlmoonsGemini-1": FakeChatProvider(
                provider_id="AlmoonsGemini-1",
                api_base="https://api.starbridge.zone",
                api_key="sk-gem-retry-1111",
                model="gemini-3.8-flash",
            ),
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-retry-2222",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
        }
    )
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "fallback_provider_id": "deepseek/deepseek-flash",
            "output_mode": "simple",
            "retry_times": 0,
        }
    )
    engine = HearVoiceEngine(ctx, cfg)
    mp3_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x10" + (b"\x00" * 32)
    wav_bytes = make_wav_bytes(duration_ms=100, sample_rate=16000)
    audio = ResolvedAudio(
        base64_data=base64.b64encode(mp3_bytes).decode("ascii"),
        mime_type="audio/mpeg",
        audio_format="mp3",
        size_bytes=len(mp3_bytes),
        source_kind="quoted",
        raw_bytes=mp3_bytes,
    )

    transcode_calls: list[dict[str, Any]] = []

    async def _fake_ffmpeg_transcode(ffmpeg_path, raw_bytes, *, input_suffix, output_suffix, output_args, timeout_sec=None):
        transcode_calls.append(
            {
                "ffmpeg_path": ffmpeg_path,
                "input_suffix": input_suffix,
                "output_suffix": output_suffix,
                "output_args": output_args,
            }
        )
        return wav_bytes

    def _route(call: dict[str, Any]):
        if "generateContent" in call["url"]:
            mime = call["json"]["contents"][0]["parts"][0]["inline_data"]["mime_type"]
            if mime == "audio/mp3":
                return _MockResp(
                    400,
                    {
                        "error": {
                            "code": 400,
                            "message": "The attachment format is not supported",
                            "status": "INVALID_ARGUMENT",
                        }
                    },
                )
            if mime == "audio/wav":
                return _MockResp(
                    200,
                    {"candidates": [{"content": {"parts": [{"text": "转码 WAV 后主通道成功"}]}}]},
                )
        raise AssertionError(f"Unexpected call: {call['url']}")

    sess = _MultiRouteMockSession(_route)
    engine._transport._session = sess

    with (
        caplog.at_level(logging.WARNING),
        patch("hear_voice.audio_resolver._run_ffmpeg_transcode", side_effect=_fake_ffmpeg_transcode),
    ):
        res, status = await engine.transcribe_audio(audio, ffmpeg_path="/usr/bin/ffmpeg")

    assert status == "ok"
    assert res is not None
    assert res.engine_used == "gemini"
    assert res.fallback_used is False
    assert res.transcript == "转码 WAV 后主通道成功"
    assert len(sess.calls) == 2
    assert sess.calls[0]["json"]["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/mp3"
    assert sess.calls[1]["json"]["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/wav"
    assert len(transcode_calls) == 1
    assert transcode_calls[0]["output_suffix"] == ".wav"
    assert transcode_calls[0]["output_args"] == ("-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1")
    await engine.close()


@pytest.mark.asyncio
async def test_gemini_400_wav_retry_still_fails_then_falls_back_to_deepseek_and_skips_when_already_wav(caplog):
    ctx = FakeContext(
        {
            "AlmoonsGemini-1": FakeChatProvider(
                provider_id="AlmoonsGemini-1",
                api_base="https://api.starbridge.zone",
                api_key="sk-gem-fb-1111",
                model="gemini-3.8-flash",
            ),
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-fb-2222",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
        }
    )
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "fallback_provider_id": "deepseek/deepseek-flash",
            "output_mode": "simple",
            "retry_times": 0,
        }
    )
    engine = HearVoiceEngine(ctx, cfg)
    amr_bytes = b"#!AMR\n" + (b"\x00" * 24)
    wav_bytes = make_wav_bytes(duration_ms=100, sample_rate=16000)
    amr_audio = ResolvedAudio(
        base64_data=base64.b64encode(amr_bytes).decode("ascii"),
        mime_type="audio/amr",
        audio_format="amr",
        size_bytes=len(amr_bytes),
        source_kind="quoted",
        raw_bytes=amr_bytes,
    )

    transcode_count = 0

    async def _fake_ffmpeg_transcode(*_args, **_kwargs):
        nonlocal transcode_count
        transcode_count += 1
        return wav_bytes

    def _route(call: dict[str, Any]):
        if "generateContent" in call["url"]:
            return _MockResp(
                400,
                {
                    "error": {
                        "code": 400,
                        "message": "The attachment format is not supported",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )
        if "/v1/tools/asr" in call["url"]:
            return _MockResp(200, {"request_id": "fb-400", "text": "回退 DeepSeek 成功", "duration_ms": 100})
        raise AssertionError(f"Unexpected URL: {call['url']}")

    sess = _MultiRouteMockSession(_route)
    engine._transport._session = sess

    with (
        caplog.at_level(logging.WARNING),
        patch("hear_voice.audio_resolver._run_ffmpeg_transcode", side_effect=_fake_ffmpeg_transcode),
    ):
        res, status = await engine.transcribe_audio(amr_audio, ffmpeg_path="/usr/bin/ffmpeg")

    assert status == "ok"
    assert res is not None
    assert res.engine_used == "deepseek_asr"
    assert res.fallback_used is True
    assert res.transcript == "回退 DeepSeek 成功"
    # Gemini called twice (amr + single wav retry), then DeepSeek called once.
    gemini_calls = [c for c in sess.calls if "generateContent" in c["url"]]
    deepseek_calls = [c for c in sess.calls if "/v1/tools/asr" in c["url"]]
    assert len(gemini_calls) == 2
    assert gemini_calls[0]["json"]["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/amr"
    assert gemini_calls[1]["json"]["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/wav"
    assert len(deepseek_calls) == 1

    # Already-WAV input must NOT retry primary with ffmpeg; goes straight to fallback.
    sess.calls.clear()
    transcode_count = 0
    already_wav = ResolvedAudio(
        base64_data=base64.b64encode(wav_bytes).decode("ascii"),
        mime_type="audio/wav",
        audio_format="wav",
        size_bytes=len(wav_bytes),
        raw_bytes=wav_bytes,
    )
    with patch("hear_voice.audio_resolver._run_ffmpeg_transcode", side_effect=_fake_ffmpeg_transcode):
        res2, status2 = await engine.transcribe_audio(already_wav, ffmpeg_path="/usr/bin/ffmpeg")

    assert status2 == "ok"
    assert res2 is not None and res2.engine_used == "deepseek_asr" and res2.fallback_used is True
    assert transcode_count == 0
    assert len([c for c in sess.calls if "generateContent" in c["url"]]) == 1
    assert len([c for c in sess.calls if "/v1/tools/asr" in c["url"]]) == 1
    await engine.close()
