"""Unit tests for audio extraction, pysilk decoding, DeepSeek adaptation, and security guards."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
import types
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from astrbot.api.message_components import Plain, Record, Reply

from hear_voice.audio_resolver import (
    DEEPSEEK_MAX_AUDIO_BYTES,
    ResolvedAudio,
    StaticResolver,
    cleanup_temp_files,
    detect_audio_format_bytes,
    download_remote_audio_bytes,
    encode_audio_bytes,
    extract_voice_sources,
    is_private_or_reserved_ip,
    is_safe_local_audio_path,
    prepare_audio_for_deepseek,
    prepare_remote_target,
    remap_local_path,
    resolve_voice_audio,
)
from hear_voice.config import HearVoiceConfig
from tests.conftest import FakeEvent, make_mp3_bytes, make_ogg_opus_bytes, make_wav_bytes


def test_extract_voice_sources_direct_and_quoted_without_mutating_event():
    r1 = Record(file="direct1.amr")
    r2 = Record(file="quoted1.amr")
    r3 = Record(file="direct2.amr")
    reply = Reply(id="100", chain=[Plain("旧文字"), r2])
    event = FakeEvent(messages=[reply, Plain("看看这个"), r1, r3])
    original_snapshot = list(event.message_obj.message)

    sources = extract_voice_sources(event, handle_quoted_voice=True, max_count=5)
    assert [(s.source_kind, s.component) for s in sources] == [
        ("direct", r1),
        ("direct", r3),
        ("quoted", r2),
    ]
    assert event.message_obj.message == original_snapshot

    capped = extract_voice_sources(event, handle_quoted_voice=True, max_count=2)
    assert len(capped) == 2

    no_quote = extract_voice_sources(event, handle_quoted_voice=False, max_count=5)
    assert [(s.source_kind, s.component) for s in no_quote] == [
        ("direct", r1),
        ("direct", r3),
    ]


def test_detect_audio_format_bytes():
    assert detect_audio_format_bytes(b"\x02#!SILK_V3abc") == "silk"
    assert detect_audio_format_bytes(b"#!SILK_V3abc") == "silk"
    assert detect_audio_format_bytes(b"#!AMR\nxxxx") == "amr"
    assert detect_audio_format_bytes(b"\xff\xf1\x50\x80\x02\x00\xfc") == "aac"
    assert detect_audio_format_bytes(b"\xff\xfb\x90\x00") == "mp3"
    assert detect_audio_format_bytes(make_mp3_bytes()) == "mp3"
    assert detect_audio_format_bytes(make_wav_bytes()) == "wav"
    assert detect_audio_format_bytes(b"OggS\x00\x02") == "ogg"
    assert detect_audio_format_bytes(b"fLaC\x00\x00") == "flac"


@pytest.mark.asyncio
async def test_real_pysilk_tencent_prefix_decode_to_wav_and_deepseek_ready():
    import pysilk

    pcm_samples = b"\x00\x00" * 480
    silk_buf = io.BytesIO()
    pysilk.encode(io.BytesIO(pcm_samples), silk_buf, 24000, 24000, tencent=True)
    silk_bytes = silk_buf.getvalue()
    assert silk_bytes.startswith(b"\x02#!SILK")

    cfg = HearVoiceConfig.from_mapping({})
    resolved = await encode_audio_bytes(silk_bytes, suffix=".amr", cfg=cfg, ffmpeg_path="")
    assert resolved is not None
    assert resolved.mime_type == "audio/wav"
    assert resolved.audio_format == "wav"
    decoded_wav = base64.b64decode(resolved.base64_data)
    with wave.open(io.BytesIO(decoded_wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 24000

    # Decoded 24kHz 16-bit PCM WAV uploads directly to DeepSeek ASR without ffmpeg
    prepared, err = await prepare_audio_for_deepseek(resolved, ffmpeg_path="")
    assert err == ""
    assert prepared is not None
    assert (prepared.filename, prepared.content_type) == ("speech.wav", "audio/wav")
    assert prepared.data == decoded_wav


@pytest.mark.asyncio
async def test_prepare_audio_for_deepseek_wav_ogg_transcode_and_limits(caplog):
    # 1. Valid Ogg Opus <= 8 MiB and <= 120s passes directly without ffmpeg
    ogg = make_ogg_opus_bytes(duration_sec=5.0)
    ogg_audio = ResolvedAudio(
        base64_data=base64.b64encode(ogg).decode("ascii"),
        mime_type="audio/ogg",
        audio_format="ogg",
        size_bytes=len(ogg),
        raw_bytes=ogg,
    )
    prep_ogg, err_ogg = await prepare_audio_for_deepseek(ogg_audio, ffmpeg_path="")
    assert err_ogg == ""
    assert prep_ogg is not None
    assert (prep_ogg.filename, prep_ogg.content_type) == ("speech.ogg", "audio/ogg")

    # 2. Duration > 120s WAV and Ogg Opus rejected cleanly with warning
    long_wav = make_wav_bytes(duration_ms=125_000, sample_rate=8000)
    assert len(long_wav) < DEEPSEEK_MAX_AUDIO_BYTES
    long_wav_audio = ResolvedAudio(
        base64_data=base64.b64encode(long_wav).decode("ascii"),
        mime_type="audio/wav",
        audio_format="wav",
        size_bytes=len(long_wav),
        raw_bytes=long_wav,
    )
    with caplog.at_level(logging.WARNING):
        prep_long_wav, err_long_wav = await prepare_audio_for_deepseek(long_wav_audio, ffmpeg_path="ffmpeg")
    assert prep_long_wav is None
    assert "120s" in err_long_wav
    assert "时长超限" in caplog.text

    caplog.clear()
    long_ogg = make_ogg_opus_bytes(duration_sec=125.0)
    long_ogg_audio = ResolvedAudio(
        base64_data=base64.b64encode(long_ogg).decode("ascii"),
        mime_type="audio/ogg",
        audio_format="ogg",
        size_bytes=len(long_ogg),
        raw_bytes=long_ogg,
    )
    with caplog.at_level(logging.WARNING):
        prep_long_ogg, err_long_ogg = await prepare_audio_for_deepseek(long_ogg_audio, ffmpeg_path="ffmpeg")
    assert prep_long_ogg is None
    assert "120s" in err_long_ogg
    assert "时长超限" in caplog.text

    # 3. MP3 transcoded to 16kHz mono pcm_s16le WAV; if WAV > 8 MiB, falls back to libopus Ogg
    mp3 = make_mp3_bytes()
    mp3_audio = ResolvedAudio(
        base64_data=base64.b64encode(mp3).decode("ascii"),
        mime_type="audio/mpeg",
        audio_format="mp3",
        size_bytes=len(mp3),
        raw_bytes=mp3,
    )
    wav_16k = make_wav_bytes(duration_ms=100, sample_rate=16000)

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0

        async def communicate(self):
            return b"", b""

    recorded_cmds: list[tuple[Any, ...]] = []

    async def _fake_exec_wav_ok(*cmd, **kwargs):
        recorded_cmds.append(cmd)
        Path(cmd[-1]).write_bytes(wav_16k)
        return _FakeProc()

    with patch("hear_voice.audio_resolver.asyncio.create_subprocess_exec", side_effect=_fake_exec_wav_ok):
        prep_mp3, err_mp3 = await prepare_audio_for_deepseek(mp3_audio, ffmpeg_path="ffmpeg")
    assert err_mp3 == ""
    assert prep_mp3 is not None
    assert (prep_mp3.filename, prep_mp3.content_type) == ("speech.wav", "audio/wav")
    assert "pcm_s16le" in recorded_cmds[0]
    assert "16000" in recorded_cmds[0]

    # When WAV transcode exceeds 8 MiB, fall back to libopus Ogg <= 8 MiB
    oversized_wav = b"RIFF" + b"\x00" * 40 + b"\x00" * (DEEPSEEK_MAX_AUDIO_BYTES + 128)
    small_ogg = make_ogg_opus_bytes(duration_sec=10.0)
    calls: list[tuple[Any, ...]] = []

    async def _fake_exec_wav_too_big_then_opus(*cmd, **kwargs):
        calls.append(cmd)
        out_path = Path(cmd[-1])
        if out_path.suffix == ".wav":
            out_path.write_bytes(oversized_wav)
        else:
            out_path.write_bytes(small_ogg)
        return _FakeProc()

    with patch(
        "hear_voice.audio_resolver.asyncio.create_subprocess_exec",
        side_effect=_fake_exec_wav_too_big_then_opus,
    ):
        prep_opus, err_opus = await prepare_audio_for_deepseek(mp3_audio, ffmpeg_path="ffmpeg")
    assert err_opus == ""
    assert prep_opus is not None
    assert (prep_opus.filename, prep_opus.content_type) == ("speech.ogg", "audio/ogg")
    assert len(calls) == 2
    assert "libopus" in calls[1]


@pytest.mark.asyncio
async def test_resolve_inline_base64_and_size_limit():
    cfg = HearVoiceConfig.from_mapping({"max_audio_mb": 1})
    wav_b64 = base64.b64encode(make_wav_bytes()).decode("ascii")
    event = FakeEvent(messages=[Record(file=f"base64://{wav_b64}")])
    sources = extract_voice_sources(event)
    resolved = await resolve_voice_audio(event, sources[0], cfg)
    assert resolved is not None
    assert resolved.mime_type == "audio/wav"

    huge_b64 = base64.b64encode(b"x" * (1024 * 1024 + 500)).decode("ascii")
    event_big = FakeEvent(messages=[Record(file=f"base64://{huge_b64}")])
    sources_big = extract_voice_sources(event_big)
    assert await resolve_voice_audio(event_big, sources_big[0], cfg) is None


def test_local_path_guard_and_remap(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    good_file = allowed / "voice.wav"
    good_file.write_bytes(make_wav_bytes())
    bad_ext = allowed / "passwd.txt"
    bad_ext.write_text("nope", encoding="utf-8")
    outside_file = outside / "private.wav"
    outside_file.write_bytes(make_wav_bytes())

    cfg = HearVoiceConfig.from_mapping(
        {
            "strict_local_path_check": True,
            "local_audio_allowed_dirs": [str(allowed)],
            "path_remap_from": "/app/.config/QQ",
            "path_remap_to": str(allowed),
        }
    )
    assert is_safe_local_audio_path(str(good_file), cfg) is True
    assert is_safe_local_audio_path(str(bad_ext), cfg) is False
    assert is_safe_local_audio_path(str(outside_file), cfg) is False
    assert is_safe_local_audio_path(str(allowed / ".." / "secret" / "private.wav"), cfg) is False

    remapped = remap_local_path("/app/.config/QQ/ptt/voice.wav", cfg)
    assert remapped.replace("\\", "/").endswith("/ptt/voice.wav")
    assert remapped.startswith(str(allowed))


@pytest.mark.asyncio
async def test_resolve_file_uri_and_local_wav(tmp_path):
    allowed = tmp_path / "voice_dir"
    allowed.mkdir()
    wav_path = allowed / "sample.wav"
    wav_path.write_bytes(make_wav_bytes())

    cfg = HearVoiceConfig.from_mapping(
        {"local_audio_allowed_dirs": [str(allowed)], "voice_file_wait_sec": 0}
    )
    event = FakeEvent(messages=[Record(file="", url=wav_path.as_uri())])
    sources = extract_voice_sources(event)
    resolved = await resolve_voice_audio(event, sources[0], cfg)
    assert resolved is not None
    assert resolved.mime_type == "audio/wav"


def test_is_private_or_reserved_ip():
    assert is_private_or_reserved_ip("127.0.0.1") is True
    assert is_private_or_reserved_ip("10.0.0.5") is True
    assert is_private_or_reserved_ip("172.16.0.1") is True
    assert is_private_or_reserved_ip("192.168.1.10") is True
    assert is_private_or_reserved_ip("169.254.169.254") is True
    assert is_private_or_reserved_ip("::1") is True
    assert is_private_or_reserved_ip("::ffff:127.0.0.1") is True
    assert is_private_or_reserved_ip("not-an-ip") is True
    assert is_private_or_reserved_ip("8.8.8.8") is False


@pytest.mark.asyncio
async def test_ssrf_guards_and_static_resolver():
    cfg_disabled = HearVoiceConfig.from_mapping({"allow_remote_audio_url": False})
    ok, _ = await prepare_remote_target("https://cdn.example.com/a.mp3", cfg_disabled)
    assert ok is False

    cfg = HearVoiceConfig.from_mapping(
        {
            "allow_remote_audio_url": True,
            "block_private_network": True,
            "remote_audio_domain_whitelist": ["cdn.example.com", ".qq.com"],
        }
    )

    for bad_url in (
        "file:///etc/passwd",
        "http://localhost/a.mp3",
        "http://127.0.0.1/a.mp3",
        "https://169.254.169.254/latest/meta-data",
        "https://evil.com/a.mp3",
    ):
        allowed, _ = await prepare_remote_target(bad_url, cfg)
        assert allowed is False, bad_url

    with patch("hear_voice.audio_resolver.resolve_host_ips", AsyncMock(return_value=["10.0.0.2"])):
        allowed, _ = await prepare_remote_target("https://cdn.example.com/a.mp3", cfg)
        assert allowed is False

    with patch("hear_voice.audio_resolver.resolve_host_ips", AsyncMock(return_value=["93.184.216.34"])):
        allowed, target = await prepare_remote_target("https://sub.qq.com/a.mp3", cfg)
        assert allowed is True
        assert target is not None and target["ips"] == ["93.184.216.34"]

    resolver = StaticResolver({"sub.qq.com": ["93.184.216.34"]})
    rows = await resolver.resolve("sub.qq.com", 443)
    assert rows[0]["host"] == "93.184.216.34"
    await resolver.close()


class _FakeResp:
    def __init__(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None):
        self.status = status
        self._body = body
        self.headers = headers or {}
        self.content = self

    async def iter_chunked(self, size: int):
        for i in range(0, len(self._body), size):
            yield self._body[i : i + size]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeClientSession:
    def __init__(self, responses: list[_FakeResp], *args, **kwargs) -> None:
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url: str, **kwargs):
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_redirect_to_private_ip_is_blocked():
    cfg = HearVoiceConfig.from_mapping(
        {
            "allow_remote_audio_url": True,
            "block_private_network": True,
            "remote_audio_domain_whitelist": ["cdn.example.com"],
        }
    )
    redirect_resp = _FakeResp(302, headers={"Location": "http://127.0.0.1/secret.mp3"})

    with patch("hear_voice.audio_resolver.resolve_host_ips", AsyncMock(return_value=["93.184.216.34"])), patch(
        "hear_voice.audio_resolver.aiohttp.ClientSession",
        lambda *a, **kw: _FakeClientSession([redirect_resp]),
    ):
        data, _ = await download_remote_audio_bytes("https://cdn.example.com/a.mp3", cfg)
        assert data is None


@pytest.mark.asyncio
async def test_onebot_get_record_and_get_file_fallbacks():
    mp3 = make_mp3_bytes()
    mp3_b64 = base64.b64encode(mp3).decode("ascii")

    bot1 = types.SimpleNamespace(
        api=types.SimpleNamespace(
            call_action=AsyncMock(return_value={"file": "voice.mp3", "base64": mp3_b64})
        )
    )
    ev1 = FakeEvent(messages=[Record(file="missing_token.amr")], bot=bot1)
    cfg = HearVoiceConfig.from_mapping({"voice_file_wait_sec": 0})
    res1 = await resolve_voice_audio(ev1, extract_voice_sources(ev1)[0], cfg)
    assert res1 is not None and res1.mime_type == "audio/mpeg"

    async def _call_action(action, **kwargs):
        if action == "get_record":
            raise RuntimeError("get_record unsupported")
        return {"data": {"base64": mp3_b64}}

    bot2 = types.SimpleNamespace(api=types.SimpleNamespace(call_action=_call_action))
    ev2 = FakeEvent(messages=[Record(file="missing_token2.amr")], bot=bot2)
    res2 = await resolve_voice_audio(ev2, extract_voice_sources(ev2)[0], cfg)
    assert res2 is not None and res2.mime_type == "audio/mpeg"

    bot3 = types.SimpleNamespace(
        api=types.SimpleNamespace(
            call_action=AsyncMock(return_value={"data": {"url": "http://127.0.0.1:5700/rec.mp3"}})
        )
    )
    ev3 = FakeEvent(messages=[Record(file="tok3.amr")], bot=bot3)
    cfg_disallow_loop = HearVoiceConfig.from_mapping(
        {"voice_file_wait_sec": 0, "allow_napcat_local_record_url": False}
    )
    assert await resolve_voice_audio(ev3, extract_voice_sources(ev3)[0], cfg_disallow_loop) is None

    with patch(
        "hear_voice.audio_resolver.aiohttp.ClientSession",
        lambda *a, **kw: _FakeClientSession([_FakeResp(200, mp3)]),
    ):
        res3 = await resolve_voice_audio(ev3, extract_voice_sources(ev3)[0], cfg)
        assert res3 is not None and res3.mime_type == "audio/mpeg"


def test_cleanup_temp_files_only_removes_hvstt_prefix():
    tmp_dir = tempfile.gettempdir()
    own_path = Path(tmp_dir) / f"hvstt_test_{os.getpid()}.tmp"
    other_path = Path(tmp_dir) / f"unrelated_test_{os.getpid()}.tmp"
    own_path.write_bytes(b"123")
    other_path.write_bytes(b"456")
    try:
        removed = cleanup_temp_files(older_than_sec=0)
        assert removed >= 1
        assert not own_path.exists()
        assert other_path.exists()
    finally:
        if own_path.exists():
            own_path.unlink()
        if other_path.exists():
            other_path.unlink()


@pytest.mark.asyncio
async def test_ffmpeg_async_transcoding_and_timeout_kills_process():
    cfg = HearVoiceConfig.from_mapping({})
    mp3_bytes = make_mp3_bytes()

    class _FakeProc:
        def __init__(self, *, timeout: bool = False):
            self.returncode = 0
            self._timeout = timeout
            self.killed = False

        async def communicate(self):
            if self._timeout:
                await asyncio.sleep(1.0)
            return b"", b""

        def kill(self):
            self.killed = True

        async def wait(self):
            return -9

    created_kwargs: list[dict] = []

    async def _fake_exec(*cmd, **kwargs):
        created_kwargs.append(kwargs)
        out_path = cmd[-1]
        Path(out_path).write_bytes(mp3_bytes)
        return _FakeProc(timeout=False)

    with patch("hear_voice.audio_resolver.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        res = await encode_audio_bytes(b"#!AMR\n12345678", suffix=".amr", cfg=cfg, ffmpeg_path="ffmpeg")
        assert res is not None
        assert res.mime_type == "audio/mpeg"
        if os.name == "nt":
            assert "creationflags" in created_kwargs[0]

    timeout_proc = _FakeProc(timeout=True)

    async def _fake_timeout_exec(*cmd, **kwargs):
        return timeout_proc

    with patch("hear_voice.audio_resolver.asyncio.create_subprocess_exec", side_effect=_fake_timeout_exec), patch(
        "hear_voice.audio_resolver._SUBPROCESS_TIMEOUT_SEC", 0.01
    ):
        res_fail = await encode_audio_bytes(b"#!AMR\n12345678", suffix=".amr", cfg=cfg, ffmpeg_path="ffmpeg")
        assert res_fail is None
        assert timeout_proc.killed is True
