"""Unit tests for optional AstrBot hear_voice_stt STTProvider registration and failover."""

from __future__ import annotations

import json
from typing import Any

import pytest
from astrbot.core.provider.register import provider_cls_map

from hear_voice.config import HearVoiceConfig
from hear_voice.engine import HearVoiceEngine
from hear_voice.stt_provider import (
    ADAPTER_TYPE_NAME,
    HearVoiceSTTProvider,
    ensure_hear_voice_stt_adapter_registered,
    mount_runtime_stt_provider,
    unmount_runtime_stt_provider,
)
from tests.conftest import FakeChatProvider, FakeContext, make_wav_bytes


def test_adapter_registration_is_idempotent_across_reloads():
    assert ensure_hear_voice_stt_adapter_registered() is True
    assert ADAPTER_TYPE_NAME == "hear_voice_stt"
    assert ADAPTER_TYPE_NAME in provider_cls_map
    assert ensure_hear_voice_stt_adapter_registered() is True
    assert provider_cls_map[ADAPTER_TYPE_NAME].cls_type is HearVoiceSTTProvider


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


class _MockFailoverSession:
    def __init__(self) -> None:
        self.closed = False

    def post(self, url: str, **kwargs):
        if "generateContent" in url:
            return _MockResp(403, {"error": {"message": "forbidden"}})
        if "/v1/tools/asr" in url:
            return _MockResp(200, {"request_id": "ds-stt-1", "text": "早上好，今天也要加油", "duration_ms": 80})
        raise AssertionError(url)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_mount_unmount_and_get_text_delegates_through_primary_fallback_engine(tmp_path):
    wav_path = tmp_path / "hello.wav"
    wav_path.write_bytes(make_wav_bytes(sample_rate=16000))

    ctx = FakeContext(
        {
            "AlmoonsGemini-1": FakeChatProvider(),
            "deepseek/deepseek-flash": FakeChatProvider(
                provider_id="deepseek/deepseek-flash",
                provider_type="openai_chat_completion",
                api_base="https://api.deepseek.com/v1",
                api_key="sk-ds-stt-1234",
                model="deepseek-flash",
                provider_source_id="deepseek",
            ),
        }
    )
    cfg = HearVoiceConfig.from_mapping(
        {
            "provider_id": "AlmoonsGemini-1",
            "fallback_provider_id": "deepseek/deepseek-flash",
            "register_stt_provider": True,
            "stt_provider_id": "hear_voice_stt_test",
            "local_audio_allowed_dirs": [str(tmp_path)],
        }
    )

    engine = HearVoiceEngine(ctx, cfg)
    engine._transport._session = _MockFailoverSession()

    mounted = mount_runtime_stt_provider(ctx, cfg, engine=engine)
    assert mounted is not None
    assert ctx.provider_manager.inst_map["hear_voice_stt_test"] is mounted
    assert mounted in ctx.provider_manager.stt_provider_insts

    text = await mounted.get_text(str(wav_path))
    assert text == "早上好，今天也要加油"

    await unmount_runtime_stt_provider(ctx, "hear_voice_stt_test", mounted)
    assert "hear_voice_stt_test" not in ctx.provider_manager.inst_map
    assert mounted not in ctx.provider_manager.stt_provider_insts
    await engine.close()
