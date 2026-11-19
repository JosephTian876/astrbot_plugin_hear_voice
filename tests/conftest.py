"""Shared pytest fixtures and offline AstrBot doubles."""

from __future__ import annotations

import base64
import io
import struct
import sys
import types
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUBS_ROOT = Path(__file__).resolve().parent / "stubs"

for path_entry in (str(STUBS_ROOT), str(REPO_ROOT)):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)


def make_wav_bytes(
    duration_ms: int = 80,
    sample_rate: int = 24000,
    *,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Generate minimal valid PCM WAV bytes."""
    n_samples = max(1, int(sample_rate * duration_ms / 1000))
    frame_bytes = b"\x00" * (sample_width * channels)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(frame_bytes * n_samples)
    return buf.getvalue()


def make_ogg_opus_bytes(duration_sec: float = 1.5, *, pre_skip: int = 312) -> bytes:
    """Generate minimal OggS + OpusHead + ending granule header for duration inspection tests."""
    head_payload = b"OpusHead" + struct.pack("<BBHIHB", 1, 1, pre_skip, 48000, 0, 0)
    first_page = (
        b"OggS\x00\x02"
        + struct.pack("<qIIIB", 0, 1, 0, 0, 1)
        + bytes([len(head_payload)])
        + head_payload
    )
    granule = int(duration_sec * 48000) + pre_skip
    audio_payload = b"\x00" * 16
    last_page = (
        b"OggS\x00\x04"
        + struct.pack("<qIIIB", granule, 1, 1, 0, 1)
        + bytes([len(audio_payload)])
        + audio_payload
    )
    return first_page + last_page


def make_mp3_bytes() -> bytes:
    """Generate minimal ID3-prefixed dummy MP3 payload for unit tests."""
    return b"ID3\x04\x00\x00\x00\x00\x00\x00" + (b"\xff\xfb\x90\x00" + b"\x00" * 32)


def make_data_uri_wav(duration_ms: int = 80, sample_rate: int = 24000) -> str:
    b64 = base64.b64encode(make_wav_bytes(duration_ms=duration_ms, sample_rate=sample_rate)).decode(
        "ascii"
    )
    return f"data:audio/wav;base64,{b64}"


@dataclass
class FakePlainResult:
    text: str
    stopped: bool = False

    def stop_event(self) -> "FakePlainResult":
        self.stopped = True
        return self


@dataclass
class FakeSender:
    user_id: str = "10001"
    nickname: str = "冬马"


@dataclass
class FakeMessageObj:
    message: list[Any] = field(default_factory=list)
    message_str: str = ""
    group_id: str = ""
    self_id: str = "99999"
    sender: FakeSender = field(default_factory=FakeSender)
    message_type: str = "FriendMessage"
    raw_message: dict[str, Any] | None = None
    message_id: str = "msg-1"


class FakeEvent:
    """Minimal faithful AstrMessageEvent test double."""

    def __init__(
        self,
        *,
        messages: list[Any] | None = None,
        message_str: str = "",
        group_id: str = "",
        sender_id: str = "10001",
        sender_name: str = "测试用户",
        self_id: str = "99999",
        platform_name: str = "aiocqhttp",
        raw_message: dict[str, Any] | None = None,
        bot: Any = None,
    ) -> None:
        self.message_str = message_str
        msg_type = "GroupMessage" if group_id else "FriendMessage"
        self.message_obj = FakeMessageObj(
            message=list(messages or []),
            message_str=message_str,
            group_id=group_id,
            self_id=self_id,
            sender=FakeSender(user_id=sender_id, nickname=sender_name),
            message_type=msg_type,
            raw_message=raw_message,
        )
        self._platform_name = platform_name
        target = group_id or sender_id
        self.unified_msg_origin = f"{platform_name}:{msg_type}:{target}"
        self.extras: dict[str, Any] = {}
        self.bot = bot
        self.call_llm = False
        self.should_call_llm_calls: list[bool] = []
        self.stopped = False
        self.requested_llm = False
        self._has_send_oper = False
        self.sent_chains: list[Any] = []

    def get_messages(self) -> list[Any]:
        return self.message_obj.message

    def get_message_str(self) -> str:
        return self.message_str

    def get_group_id(self) -> str:
        return self.message_obj.group_id

    def get_sender_id(self) -> str:
        return self.message_obj.sender.user_id

    def get_sender_name(self) -> str:
        return self.message_obj.sender.nickname

    def get_self_id(self) -> str:
        return self.message_obj.self_id

    def get_platform_name(self) -> str:
        return self._platform_name

    def set_extra(self, key: str, value: Any) -> None:
        self.extras[key] = value

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)

    def stop_event(self) -> None:
        self.stopped = True

    def is_stopped(self) -> bool:
        return self.stopped

    def should_call_llm(self, val: bool) -> None:
        self.call_llm = val
        self.should_call_llm_calls.append(val)

    def plain_result(self, text: str) -> FakePlainResult:
        return FakePlainResult(text=text)

    async def send(self, message: Any) -> None:
        self.sent_chains.append(message)
        self._has_send_oper = True

    def request_llm(self, **kwargs: Any) -> Any:  # pragma: no cover - must never be called
        self.requested_llm = True
        raise AssertionError("Non-hijacking STT plugin must never call request_llm")


class FakeChatProvider:
    def __init__(
        self,
        *,
        provider_id: str = "AlmoonsGemini-1",
        provider_type: str = "googlegenai_chat_completion",
        api_base: str = "https://api.starbridge.zone",
        api_key: str = "test-secret-key-1234",
        model: str = "［抗截断］gemini-3.8-flash",
        proxy: str = "",
        provider_source_id: str = "",
    ) -> None:
        self.provider_config: dict[str, Any] = {
            "id": provider_id,
            "type": provider_type,
            "api_base": api_base,
            "key": [api_key],
            "model": model,
            "proxy": proxy,
        }
        if provider_source_id:
            self.provider_config["provider_source_id"] = provider_source_id
        self._key = api_key
        self._model = model

    def get_current_key(self) -> str:
        return self._key

    def get_model(self) -> str:
        return self._model


class FakeContext:
    def __init__(self, providers: dict[str, Any] | None = None) -> None:
        self._providers = dict(providers or {})
        self.provider_manager = types.SimpleNamespace(
            inst_map=dict(self._providers),
            stt_provider_insts=[],
            curr_stt_provider_inst=None,
            provider_settings={},
        )
        self.sent_messages: list[tuple[Any, Any]] = []

    def get_provider_by_id(self, provider_id: str) -> Any | None:
        return self._providers.get(provider_id) or self.provider_manager.inst_map.get(provider_id)

    async def send_message(self, session: Any, message_chain: Any) -> bool:
        self.sent_messages.append((session, message_chain))
        return True


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()


@pytest.fixture
def mp3_bytes() -> bytes:
    return make_mp3_bytes()
