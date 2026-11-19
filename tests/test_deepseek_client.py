"""Unit tests for DeepSeek official ASR (/v1/tools/asr) client and URL normalization."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest

from hear_voice.audio_resolver import ResolvedAudio
from hear_voice.config import HearVoiceConfig
from hear_voice.deepseek_client import DeepSeekAsrClient, normalize_deepseek_asr_url
from hear_voice.engine import resolve_endpoint
from tests.conftest import make_ogg_opus_bytes, make_wav_bytes


def test_normalize_deepseek_asr_url_variants():
    assert normalize_deepseek_asr_url("") == "https://api.deepseek.com/v1/tools/asr"
    assert normalize_deepseek_asr_url("https://api.deepseek.com") == "https://api.deepseek.com/v1/tools/asr"
    assert normalize_deepseek_asr_url("https://api.deepseek.com/") == "https://api.deepseek.com/v1/tools/asr"
    assert normalize_deepseek_asr_url("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1/tools/asr"
    assert normalize_deepseek_asr_url("https://api.deepseek.com/v1/") == "https://api.deepseek.com/v1/tools/asr"
    assert (
        normalize_deepseek_asr_url("https://api.deepseek.com/v1/tools/asr")
        == "https://api.deepseek.com/v1/tools/asr"
    )
    assert (
        normalize_deepseek_asr_url("https://api.deepseek.com/v1/chat/completions")
        == "https://api.deepseek.com/v1/tools/asr"
    )
    assert (
        normalize_deepseek_asr_url("https://custom-proxy.example.com/ds/tools/asr")
        == "https://custom-proxy.example.com/ds/tools/asr"
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
async def test_deepseek_asr_multipart_upload_wav_and_json_parse():
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_engine": "deepseek_asr",
            "api_base": "https://api.deepseek.com/v1",
            "api_key": "sk-ds-secret-1111",
            "retry_times": 1,
        }
    )
    ep = await resolve_endpoint(None, cfg)
    assert ep is not None
    assert ep.engine_kind == "deepseek_asr"
    assert ep.api_base == "https://api.deepseek.com/v1/tools/asr"

    wav = make_wav_bytes(duration_ms=100, sample_rate=16000)
    audio = ResolvedAudio(
        base64_data=base64.b64encode(wav).decode("ascii"),
        mime_type="audio/wav",
        audio_format="wav",
        size_bytes=len(wav),
        raw_bytes=wav,
    )

    recorded: list[dict[str, Any]] = []
    client = DeepSeekAsrClient(cfg)
    client._session = _MockSession(
        [
            _MockResp(
                200,
                {
                    "request_id": "req_ds_123",
                    "text": "你好，今天天气真不错。",
                    "duration_ms": 100,
                },
            )
        ],
        recorded,
    )

    out = await client.transcribe(audio, endpoint=ep)
    assert out == "你好，今天天气真不错。"
    assert len(recorded) == 1
    req = recorded[0]
    assert req["url"] == "https://api.deepseek.com/v1/tools/asr"
    assert req["headers"]["Authorization"] == "Bearer sk-ds-secret-1111"
    assert req["json"] is None
    form = req["data"]
    assert isinstance(form, aiohttp.FormData)
    # Verify exact single 'file' multipart field (no 'model' or 'language' fields)
    fields = form._fields
    assert len(fields) == 1
    opts, fheaders, value = fields[0]
    assert opts["name"] == "file"
    assert opts["filename"] == "speech.wav"
    assert fheaders["Content-Type"] == "audio/wav"
    assert value == wav
    await client.close()


@pytest.mark.asyncio
async def test_deepseek_asr_retries_with_fresh_formdata_and_redacts_401(caplog):
    secret = "sk-ds-secret-9999"
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_engine": "deepseek_asr",
            "api_key": secret,
            "retry_times": 1,
        }
    )
    ep = await resolve_endpoint(None, cfg)
    assert ep is not None
    assert ep.api_base == "https://api.deepseek.com/v1/tools/asr"

    ogg = make_ogg_opus_bytes(duration_sec=2.0)
    audio = ResolvedAudio(
        base64_data=base64.b64encode(ogg).decode("ascii"),
        mime_type="audio/ogg",
        audio_format="ogg",
        size_bytes=len(ogg),
        raw_bytes=ogg,
    )

    # 1) 429 retry then 200 -> sends two distinct FormData instances
    recorded_retry: list[dict[str, Any]] = []
    client = DeepSeekAsrClient(cfg)
    client._session = _MockSession(
        [
            _MockResp(429, {"error": {"message": "rate limit"}}),
            _MockResp(200, {"request_id": "req_2", "text": "测试重试成功", "duration_ms": 2000}),
        ],
        recorded_retry,
    )
    with patch("hear_voice.engine.asyncio.sleep", return_value=None):
        out = await client.transcribe(audio, endpoint=ep)
    assert out == "测试重试成功"
    assert len(recorded_retry) == 2
    assert recorded_retry[0]["data"] is not recorded_retry[1]["data"]
    assert recorded_retry[1]["data"]._fields[0][0]["filename"] == "speech.ogg"

    # 2) Non-retryable 401 -> zero retries & secret redacted from logs & last_error
    caplog.clear()
    recorded_401: list[dict[str, Any]] = []
    client._session = _MockSession(
        [_MockResp(401, {"error": {"message": f"invalid key {secret}"}})],
        recorded_401,
    )
    with caplog.at_level(logging.WARNING):
        out_401 = await client.transcribe(audio, endpoint=ep)
    assert out_401 == ""
    assert len(recorded_401) == 1
    assert "401" in client.last_error
    assert secret not in client.last_error
    assert secret not in caplog.text
    await client.close()


@pytest.mark.asyncio
async def test_deepseek_asr_rejects_over_120s_audio_cleanly(caplog):
    cfg = HearVoiceConfig.from_mapping({"provider_engine": "deepseek_asr", "api_key": "sk-ds-1234"})
    ep = await resolve_endpoint(None, cfg)
    assert ep is not None

    long_wav = make_wav_bytes(duration_ms=122_000, sample_rate=8000)
    audio = ResolvedAudio(
        base64_data=base64.b64encode(long_wav).decode("ascii"),
        mime_type="audio/wav",
        audio_format="wav",
        size_bytes=len(long_wav),
        raw_bytes=long_wav,
    )
    client = DeepSeekAsrClient(cfg)
    recorded: list[dict[str, Any]] = []
    client._session = _MockSession([], recorded)

    with caplog.at_level(logging.WARNING):
        out = await client.transcribe(audio, endpoint=ep)
    assert out == ""
    assert recorded == []
    assert "120s" in client.last_error
    assert "时长超限" in caplog.text
    await client.close()
