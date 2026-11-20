"""Unit tests for GeminiClient, endpoint resolution, thought filtering, and retries."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest

from hear_voice.audio_resolver import ResolvedAudio
from hear_voice.config import HearVoiceConfig
from hear_voice.engine import resolve_endpoint
from hear_voice.gemini_client import (
    GeminiClient,
    build_gemini_generate_url,
    build_openai_chat_url,
    extract_gemini_parts_text,
    extract_openai_message_text,
    is_unsupported_audio_format_400,
    normalize_gemini_audio_mime,
)
from tests.conftest import FakeChatProvider, FakeContext


@pytest.mark.asyncio
async def test_resolve_endpoint_from_astrbot_provider_preserves_model_and_proxy():
    provider = FakeChatProvider(
        provider_id="AlmoonsGemini-1",
        provider_type="googlegenai_chat_completion",
        api_base="https://api.starbridge.zone",
        api_key="sk-provider-secret-9999",
        model="［抗截断］gemini-3.8-flash",
        proxy="http://127.0.0.1:12450",
    )
    ctx = FakeContext({"AlmoonsGemini-1": provider})
    cfg = HearVoiceConfig.from_mapping({"provider_id": "AlmoonsGemini-1"})
    ep = await resolve_endpoint(ctx, cfg)
    assert ep is not None
    assert ep.engine_kind == "gemini"
    assert ep.api_base == "https://api.starbridge.zone"
    assert ep.api_key == "sk-provider-secret-9999"
    assert ep.model == "［抗截断］gemini-3.8-flash"
    assert ep.api_format == "gemini_native"
    assert ep.auth_mode == "x_goog_api_key"
    assert ep.proxy == "http://127.0.0.1:12450"

    summary = GeminiClient.safe_endpoint_summary(ep)
    assert "sk-provider-secret-9999" not in summary
    assert "provider:AlmoonsGemini-1" in summary


@pytest.mark.asyncio
async def test_resolve_endpoint_openai_provider():
    provider = FakeChatProvider(
        provider_id="openai-relay",
        provider_type="openai_chat_completion",
        api_base="https://relay.example.com/v1",
        api_key="sk-openai-1234",
        model="gemini-2.5-flash",
    )
    ctx = FakeContext({"openai-relay": provider})
    cfg = HearVoiceConfig.from_mapping({"provider_id": "openai-relay"})
    ep = await resolve_endpoint(ctx, cfg)
    assert ep is not None
    assert ep.engine_kind == "gemini"
    assert ep.api_format == "openai_compat"
    assert ep.auth_mode == "bearer"


def test_url_builders():
    assert (
        build_gemini_generate_url("https://api.starbridge.zone/v1", "gemini-2.5-flash")
        == "https://api.starbridge.zone/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert (
        build_gemini_generate_url("https://generativelanguage.googleapis.com/v1beta", "models/gemini-2.5-flash")
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert build_openai_chat_url("https://relay.example.com") == "https://relay.example.com/v1/chat/completions"
    assert build_openai_chat_url("https://relay.example.com/v1") == "https://relay.example.com/v1/chat/completions"
    assert (
        build_openai_chat_url("https://relay.example.com/v1/chat/completions")
        == "https://relay.example.com/v1/chat/completions"
    )


def test_extract_gemini_parts_filters_thought_blocks():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "Let me analyze the speaker's audio carefully..."},
                        {"text": "1) 原话转写：明天见"},
                        {"thought": True, "text": "Double check punctuation"},
                        {"text": "2) 语气/情绪：期待"},
                    ]
                }
            }
        ]
    }
    extracted = extract_gemini_parts_text(payload)
    assert "Let me analyze" not in extracted
    assert "Double check" not in extracted
    assert "1) 原话转写：明天见" in extracted
    assert "2) 语气/情绪：期待" in extracted


def test_extract_openai_message_text_handles_multimodal_list():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "thinking", "text": "secret thought"},
                        {"type": "text", "text": "你好世界"},
                    ]
                }
            }
        ]
    }
    assert extract_openai_message_text(payload) == "你好世界"


class _MockPostResp:
    def __init__(self, status: int, payload: dict[str, Any]):
        self.status = status
        self._raw = json.dumps(payload, ensure_ascii=False)

    async def text(self) -> str:
        return self._raw

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _MockTextResp:
    def __init__(self, status: int, raw: str):
        self.status = status
        self._raw = raw

    async def text(self) -> str:
        return self._raw

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _MockSession:
    def __init__(self, responses: list[Any], recorded: list[dict[str, Any]]) -> None:
        self._responses = responses
        self._recorded = recorded
        self.closed = False

    def post(self, url: str, *, headers=None, json=None, data=None, proxy=None, ssl=None):
        self._recorded.append(
            {"url": url, "headers": headers, "json": json, "data": data, "proxy": proxy, "ssl": ssl}
        )
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_client_gemini_native_retries_and_filters_thoughts():
    cfg = HearVoiceConfig.from_mapping(
        {
            "api_base": "https://api.starbridge.zone",
            "api_key": "sk-secret-key-0001",
            "model": "gemini-3.8-flash",
            "retry_times": 1,
        }
    )
    ep = await resolve_endpoint(None, cfg)
    assert ep is not None

    recorded: list[dict[str, Any]] = []
    responses = [
        _MockPostResp(503, {"error": "temporary"}),
        _MockPostResp(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"thought": True, "text": "HIDDEN_THOUGHT"},
                                {"text": "你好呀，和纱"},
                            ]
                        }
                    }
                ]
            },
        ),
    ]
    client = GeminiClient(cfg)
    client._session = _MockSession(responses, recorded)

    with patch("hear_voice.engine.asyncio.sleep", return_value=None):
        audio = ResolvedAudio(
            base64_data="AAAA", mime_type="audio/wav", audio_format="wav", size_bytes=3
        )
        out = await client.transcribe(audio, instruction="transcribe", endpoint=ep)

    assert out == "你好呀，和纱"
    assert "HIDDEN_THOUGHT" not in out
    assert len(recorded) == 2
    assert recorded[0]["headers"]["x-goog-api-key"] == "sk-secret-key-0001"
    assert recorded[0]["json"]["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/wav"
    await client.close()


@pytest.mark.asyncio
async def test_client_openai_compat_uses_input_audio_with_format_and_normalize():
    cfg = HearVoiceConfig.from_mapping(
        {
            "api_base": "https://api.starbridge.zone/v1",
            "api_key": "sk-openai-0002",
            "model": "gemini-3.8-flash",
            "retry_times": 1,
        }
    )
    ep = await resolve_endpoint(None, cfg)
    assert ep is not None

    recorded: list[dict[str, Any]] = []
    responses = [
        aiohttp.ClientError("transient glitch"),
        _MockPostResp(200, {"choices": [{"message": {"content": "今天天气不错"}}]}),
        _MockPostResp(200, {"choices": [{"message": {"content": "今天天气不错。"}}]}),
    ]
    client = GeminiClient(cfg)
    client._session = _MockSession(responses, recorded)

    with patch("hear_voice.engine.asyncio.sleep", return_value=None):
        audio = ResolvedAudio(
            base64_data="BBBB", mime_type="audio/mpeg", audio_format="mp3", size_bytes=3
        )
        raw = await client.transcribe(audio, instruction="transcribe", endpoint=ep)
        norm = await client.normalize_transcript_with_model(raw, endpoint=ep)

    assert raw == "今天天气不错"
    assert norm == "今天天气不错。"
    assert recorded[1]["headers"]["Authorization"] == "Bearer sk-openai-0002"
    input_audio = recorded[1]["json"]["messages"][0]["content"][0]["input_audio"]
    assert input_audio == {"data": "BBBB", "format": "mp3"}
    await client.close()


@pytest.mark.asyncio
async def test_client_logs_non_retryable_4xx_without_retrying_or_leaking_secrets(caplog):
    secret_key = "sk-secret-key-4001"
    cfg = HearVoiceConfig.from_mapping(
        {
            "api_base": "https://api.starbridge.zone",
            "api_key": secret_key,
            "model": "gemini-3.8-flash",
            "retry_times": 2,
        }
    )
    ep = await resolve_endpoint(None, cfg)
    assert ep is not None

    recorded: list[dict[str, Any]] = []
    responses = [
        _MockPostResp(
            400,
            {
                "error": {
                    "code": 400,
                    "message": "The attachment format is not supported",
                    "status": "INVALID_ARGUMENT",
                }
            },
        ),
        _MockPostResp(200, {"candidates": []}),
    ]
    client = GeminiClient(cfg)
    client._session = _MockSession(responses, recorded)

    audio = ResolvedAudio(
        base64_data="CCCC", mime_type="audio/wav", audio_format="wav", size_bytes=3
    )
    with caplog.at_level(logging.WARNING), patch("hear_voice.engine.asyncio.sleep", return_value=None):
        out = await client.transcribe(audio, instruction="transcribe", endpoint=ep)

    assert out == ""
    assert len(recorded) == 1
    assert "400" in client.last_error
    assert "INVALID_ARGUMENT" in client.last_error
    assert "The attachment format is not supported" in client.last_error

    warnings = "\n".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "400" in warnings
    assert "The attachment format is not supported" in warnings
    assert secret_key not in warnings
    assert secret_key not in client.last_error
    assert "x-goog-api-key" not in warnings
    await client.close()


@pytest.mark.asyncio
async def test_client_redacts_secrets_and_retries_429_5xx_and_200_errors(caplog):
    secret_key = "sk-secret-key-4019"
    leaked_query_key = "AIza" + "FakeSecretToken123456789"
    leaked_bearer = "bearer-tok-987654321"
    cfg = HearVoiceConfig.from_mapping(
        {
            "api_base": "https://relay.example.com/v1",
            "api_key": secret_key,
            "model": "gemini-3.8-flash",
            "retry_times": 1,
        }
    )
    ep = await resolve_endpoint(None, cfg)
    assert ep is not None

    audio = ResolvedAudio(
        base64_data="DDDD", mime_type="audio/mpeg", audio_format="mp3", size_bytes=3
    )

    rec_401: list[dict[str, Any]] = []
    client = GeminiClient(cfg)
    client._session = _MockSession(
        [
            _MockTextResp(
                401,
                json.dumps(
                    {
                        "error": {
                            "message": (
                                f"Bad key {secret_key}, Bearer {leaked_bearer}, "
                                f"url=https://relay.example.com/v1?key={leaked_query_key} "
                                + ("x" * 300)
                            )
                        }
                    }
                ),
            )
        ],
        rec_401,
    )
    with caplog.at_level(logging.WARNING), patch("hear_voice.engine.asyncio.sleep", return_value=None):
        out_401 = await client.transcribe(audio, instruction="transcribe", endpoint=ep)
    assert out_401 == ""
    assert len(rec_401) == 1
    assert secret_key not in client.last_error
    assert leaked_bearer not in client.last_error
    assert leaked_query_key not in client.last_error
    assert client.last_error.endswith("...")
    assert "401" in caplog.text
    assert secret_key not in caplog.text
    assert leaked_bearer not in caplog.text
    assert leaked_query_key not in caplog.text

    caplog.clear()
    rec_retry: list[dict[str, Any]] = []
    client._session = _MockSession(
        [
            _MockPostResp(429, {"error": {"message": "rate limited"}}),
            _MockPostResp(500, {"error": {"message": "upstream crash"}}),
        ],
        rec_retry,
    )
    with caplog.at_level(logging.WARNING), patch("hear_voice.engine.asyncio.sleep", return_value=None):
        out_retry = await client.transcribe(audio, instruction="transcribe", endpoint=ep)
    assert out_retry == ""
    assert len(rec_retry) == 2
    assert "429" in caplog.text
    assert "500" in caplog.text
    assert "HTTP 500" in client.last_error

    caplog.clear()
    ep_gemini = await resolve_endpoint(
        None,
        HearVoiceConfig.from_mapping(
            {"api_base": "https://api.starbridge.zone", "api_key": secret_key, "retry_times": 0}
        ),
    )
    assert ep_gemini is not None
    client._session = _MockSession(
        [
            _MockPostResp(200, {"error": {"code": 400, "message": "wrapped error"}}),
            _MockPostResp(200, {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}),
        ],
        [],
    )
    with caplog.at_level(logging.WARNING):
        assert await client.transcribe(audio, instruction="transcribe", endpoint=ep_gemini) == ""
        assert "wrapped error" in client.last_error
        assert await client.transcribe(audio, instruction="transcribe", endpoint=ep_gemini) == ""
        assert "SAFETY" in client.last_error
    assert "wrapped error" in caplog.text
    assert "SAFETY" in caplog.text
    await client.close()


def test_normalize_gemini_audio_mime_and_400_format_detector():
    assert normalize_gemini_audio_mime("audio/mpeg") == "audio/mp3"
    assert normalize_gemini_audio_mime("audio/x-mpeg") == "audio/mp3"
    assert normalize_gemini_audio_mime("AUDIO/MPEG; charset=binary") == "audio/mp3"
    assert normalize_gemini_audio_mime("audio/mp3") == "audio/mp3"
    assert normalize_gemini_audio_mime("audio/wav") == "audio/wav"
    assert normalize_gemini_audio_mime("audio/ogg") == "audio/ogg"
    assert normalize_gemini_audio_mime("", "mp3") == "audio/mp3"
    assert normalize_gemini_audio_mime("", "wav") == "audio/wav"

    assert is_unsupported_audio_format_400(
        "HTTP 400: INVALID_ARGUMENT: The attachment format is not supported"
    )
    assert is_unsupported_audio_format_400(
        'HTTP 400: {"error":{"code":400,"message":"Unsupported MIME type: audio/amr"}}'
    )
    assert is_unsupported_audio_format_400("HTTP 400: unsupported audio format")
    assert not is_unsupported_audio_format_400("HTTP 400: API key not valid")
    assert not is_unsupported_audio_format_400("HTTP 503: attachment format is not supported")
    assert not is_unsupported_audio_format_400(
        "HTTP 200 error payload: attachment format is not supported"
    )


@pytest.mark.asyncio
async def test_client_gemini_native_normalizes_audio_mpeg_to_audio_mp3():
    cfg = HearVoiceConfig.from_mapping(
        {
            "api_base": "https://api.starbridge.zone",
            "api_key": "sk-gem-mime-0001",
            "model": "gemini-3.8-flash",
            "retry_times": 0,
        }
    )
    ep = await resolve_endpoint(None, cfg)
    assert ep is not None
    assert ep.api_format == "gemini_native"

    recorded: list[dict[str, Any]] = []
    responses = [
        _MockPostResp(
            200,
            {"candidates": [{"content": {"parts": [{"text": "引用语音正常识别"}]}}]},
        )
    ]
    client = GeminiClient(cfg)
    client._session = _MockSession(responses, recorded)

    audio = ResolvedAudio(
        base64_data="SUQz",
        mime_type="audio/mpeg",
        audio_format="mp3",
        size_bytes=3,
        source_kind="quoted",
    )
    out = await client.transcribe(audio, instruction="transcribe", endpoint=ep)
    assert out == "引用语音正常识别"
    assert len(recorded) == 1
    inline = recorded[0]["json"]["contents"][0]["parts"][0]["inline_data"]
    assert inline["mime_type"] == "audio/mp3"
    assert inline["data"] == "SUQz"
    await client.close()
