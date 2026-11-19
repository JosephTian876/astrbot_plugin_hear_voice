"""Async audio source extraction, OneBot fallbacks, SSRF/path guards, and format adaptation."""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urljoin, urlparse
from urllib.request import url2pathname

import aiohttp
from aiohttp.abc import AbstractResolver

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover - standalone import fallback
    import logging

    logger = logging.getLogger("astrbot_plugin_hear_voice")

from .config import HearVoiceConfig, redact_url

_ALLOWED_LOCAL_EXTS = frozenset(
    {".mp3", ".wav", ".amr", ".silk", ".pcm", ".ogg", ".opus", ".aac", ".m4a", ".flac", ".bin"}
)
_TEMP_PREFIX = "hvstt_"
_MAX_REDIRECT_HOPS = 3
_SUBPROCESS_TIMEOUT_SEC = 30.0

DEEPSEEK_MAX_AUDIO_BYTES = 8 * 1024 * 1024
DEEPSEEK_MAX_DURATION_SEC = 120.0
DEEPSEEK_SUPPORTED_WAV_RATES = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})


@dataclass(frozen=True)
class VoiceSource:
    """One voice component discovered on an incoming event."""

    index: int
    component: Any
    source_kind: Literal["direct", "quoted"]


@dataclass(frozen=True)
class ResolvedAudio:
    """Normalized audio payload ready for Gemini, DeepSeek ASR, or OpenAI STT."""

    base64_data: str
    mime_type: str
    audio_format: str
    size_bytes: int
    source_kind: Literal["direct", "quoted"] = "direct"
    raw_bytes: bytes | None = None
    source_raw_bytes: bytes | None = None
    source_format: str = ""

    def get_bytes(self) -> bytes:
        """Return raw audio bytes for this payload."""
        if self.raw_bytes is not None:
            return self.raw_bytes
        return base64.b64decode(self.base64_data)


@dataclass(frozen=True)
class WavAudioInfo:
    """Header metadata parsed from an uncompressed PCM RIFF WAV buffer."""

    channels: int
    sample_width: int
    sample_rate: int
    n_frames: int
    duration_sec: float

    @property
    def is_deepseek_compatible(self) -> bool:
        """Return True if WAV matches DeepSeek 16-bit PCM sample rate requirements."""
        return (
            self.channels >= 1
            and self.sample_width == 2
            and self.sample_rate in DEEPSEEK_SUPPORTED_WAV_RATES
        )


@dataclass(frozen=True)
class PreparedDeepSeekAudio:
    """Audio payload adapted to DeepSeek ASR (/v1/tools/asr) container and limit rules."""

    data: bytes
    filename: str
    content_type: str
    duration_sec: float | None = None


class StaticResolver(AbstractResolver):
    """Pin DNS lookups to pre-validated public IP addresses to mitigate DNS rebinding."""

    def __init__(self, host_ip_map: dict[str, list[str]]) -> None:
        self._host_ip_map = {k.lower(): list(v) for k, v in host_ip_map.items()}

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_UNSPEC):
        ips = self._host_ip_map.get((host or "").lower(), [])
        if not ips:
            raise OSError(f"resolver: no validated IP for host {host}")
        results = []
        for ip in ips:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            results.append(
                {
                    "hostname": host,
                    "host": ip,
                    "port": port,
                    "family": fam,
                    "proto": socket.IPPROTO_TCP,
                    "flags": socket.AI_NUMERICHOST,
                }
            )
        return results

    async def close(self) -> None:
        return


def _file_uri_to_local_path(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "file":
        return uri
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    if netloc and netloc.lower() != "localhost":
        if len(netloc) == 2 and netloc[1] == ":" and netloc[0].isalpha():
            return os.path.normpath(url2pathname(f"{netloc}{path}"))
        return os.path.normpath(url2pathname(f"//{netloc}{path}"))
    decoded = url2pathname(path)
    if len(decoded) >= 4 and decoded[0] in ("/", "\\") and decoded[2] == ":" and decoded[1].isalpha():
        decoded = decoded[1:]
    elif os.name != "nt" and decoded.startswith("//"):
        decoded = "/" + decoded.lstrip("/")
    return os.path.normpath(decoded)


def is_private_or_reserved_ip(ip_str: str) -> bool:
    """Return True if IP is private, loopback, link-local, multicast, reserved, or unspecified."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    if getattr(ip, "ipv4_mapped", None) is not None:
        return is_private_or_reserved_ip(str(ip.ipv4_mapped))
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if h == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(h)
        if getattr(ip, "ipv4_mapped", None) is not None:
            return ip.ipv4_mapped.is_loopback
        return ip.is_loopback
    except ValueError:
        return False


def host_allowed_by_whitelist(host: str, whitelist: list[str]) -> bool:
    if not whitelist:
        return True
    normalized = (host or "").lower().strip()
    if not normalized:
        return False
    for entry in whitelist:
        rule = entry.lower().strip()
        if not rule:
            continue
        if rule.startswith("."):
            if normalized.endswith(rule):
                return True
        elif normalized == rule or normalized.endswith("." + rule):
            return True
    return False


async def resolve_host_ips(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    ips: list[str] = []
    for info in infos:
        if info and len(info) >= 5 and info[4]:
            ip = str(info[4][0])
            if ip not in ips:
                ips.append(ip)
    return ips


async def prepare_remote_target(url: str, cfg: HearVoiceConfig) -> tuple[bool, dict[str, Any] | None]:
    """Validate a remote HTTP(S) audio URL against SSRF and domain whitelist rules."""
    if not cfg.allow_remote_audio_url:
        return False, None

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        return False, None

    host = (parsed.hostname or "").lower().strip()
    if not host or (cfg.block_private_network and host == "localhost"):
        return False, None

    if not host_allowed_by_whitelist(host, cfg.remote_audio_domain_whitelist):
        return False, None

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)

    try:
        ipaddress.ip_address(host)
        if cfg.block_private_network and is_private_or_reserved_ip(host):
            return False, None
        return True, {"parsed": parsed, "host": host, "port": port, "ips": [host]}
    except ValueError:
        pass

    try:
        ips = await resolve_host_ips(host, port)
        if not ips:
            return False, None
        if cfg.block_private_network and any(is_private_or_reserved_ip(ip) for ip in ips):
            return False, None
        return True, {"parsed": parsed, "host": host, "port": port, "ips": ips}
    except Exception:
        return False, None


def _temp_dirs_set() -> set[str]:
    dirs = {os.path.realpath(tempfile.gettempdir())}
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

        p = str(get_astrbot_temp_path() or "").strip()
        if p:
            dirs.add(os.path.realpath(p))
    except Exception:
        pass
    return dirs


def is_safe_local_audio_path(path: str, cfg: HearVoiceConfig) -> bool:
    """Validate that a local path is a regular allowed audio file inside whitelisted dirs."""
    if not path:
        return False
    try:
        rp = os.path.realpath(path)
    except OSError:
        return False
    if not os.path.isfile(rp):
        return False

    bn = os.path.basename(rp)
    if bn.startswith(_TEMP_PREFIX):
        for tdir in _temp_dirs_set():
            try:
                if os.path.commonpath([rp, tdir]) == tdir:
                    return True
            except ValueError:
                continue

    ext = os.path.splitext(rp)[1].lower()
    if ext not in _ALLOWED_LOCAL_EXTS:
        return False

    if not cfg.strict_local_path_check:
        return True

    for base in cfg.local_audio_allowed_dirs:
        try:
            if os.path.commonpath([rp, base]) == base:
                return True
        except ValueError:
            continue
    return False


def remap_local_path(path: str, cfg: HearVoiceConfig) -> str:
    """Map NapCat container paths to host-accessible paths via manual or auto rules."""
    if not path:
        return path
    norm = path.replace("\\", "/")

    if cfg.path_remap_from and cfg.path_remap_to:
        src = cfg.path_remap_from.rstrip("/\\").replace("\\", "/")
        dst = cfg.path_remap_to.rstrip("/\\")
        if norm.startswith(src + "/") or norm == src:
            return dst + norm[len(src) :]

    for raw_src, raw_dst in cfg.auto_remap_pairs:
        src = raw_src.replace("\\", "/")
        if norm.startswith(src + "/") or norm == src:
            return raw_dst + norm[len(src) :]

    return path


def find_ffmpeg(custom_path: str = "") -> str:
    """Locate ffmpeg executable path safely without running unvalidated binaries."""
    custom = (custom_path or "").strip()
    if custom:
        if os.path.isdir(custom):
            candidate = os.path.join(custom, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        else:
            candidate = custom
        bn = os.path.basename(candidate).lower()
        if bn in ("ffmpeg", "ffmpeg.exe") and os.path.isfile(candidate):
            if os.name == "nt" or os.access(candidate, os.X_OK):
                return candidate
        return ""

    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    found = shutil.which(exe)
    if found:
        return found
    return ""


def detect_audio_format_bytes(data: bytes, suffix: str = "") -> str:
    """Identify audio container/codec format from leading magic bytes, falling back to suffix."""
    header = data[:32]
    if b"SILK" in header:
        return "silk"
    if header.startswith(b"#!AMR"):
        return "amr"
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xF6) == 0xF0:
        return "aac"
    if header.startswith(b"ID3") or (
        len(header) >= 2
        and header[0] == 0xFF
        and (header[1] & 0xE0) == 0xE0
        and (header[1] & 0x06) != 0
    ):
        return "mp3"
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "wav"
    if header.startswith(b"OggS"):
        return "ogg"
    if header.startswith(b"fLaC"):
        return "flac"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "m4a"

    ext = suffix.lower().lstrip(".")
    if ext in {"mp3", "wav", "silk", "amr", "ogg", "opus", "aac", "m4a", "flac"}:
        return ext
    return "unknown"


def is_ogg_opus_bytes(data: bytes) -> bool:
    """Return True when ``data`` looks like an Ogg container carrying an Opus stream."""
    return len(data) >= 36 and data.startswith(b"OggS") and b"OpusHead" in data[:128]


def inspect_wav_audio(data: bytes) -> WavAudioInfo | None:
    """Inspect uncompressed PCM WAV header metadata, returning None if invalid or non-PCM."""
    if not (len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"):
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getcomptype() != "NONE":
                return None
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
        if channels <= 0 or sample_width <= 0 or sample_rate <= 0:
            return None
        duration_sec = n_frames / float(sample_rate)
        return WavAudioInfo(
            channels=channels,
            sample_width=sample_width,
            sample_rate=sample_rate,
            n_frames=n_frames,
            duration_sec=duration_sec,
        )
    except Exception:
        return None


def inspect_ogg_opus_duration(data: bytes) -> float | None:
    """Estimate Ogg Opus duration in seconds from the last Ogg page granule position."""
    if not is_ogg_opus_bytes(data):
        return None
    head_idx = data.find(b"OpusHead", 0, 128)
    if head_idx < 0 or head_idx + 12 > len(data):
        return None
    pre_skip = struct.unpack_from("<H", data, head_idx + 10)[0]

    pos = len(data)
    while pos > 0:
        idx = data.rfind(b"OggS\x00", 0, pos)
        if idx < 0:
            return None
        if idx + 14 <= len(data):
            granule = struct.unpack_from("<q", data, idx + 6)[0]
            if granule > 0:
                return max(0.0, (granule - pre_skip) / 48000.0)
        pos = idx
    return None


def _decode_silk_bytes_to_wav_sync(silk_bytes: bytes) -> bytes:
    """Decode Tencent/QQ SILK bytes (with optional leading \\x02 byte) into RIFF WAV bytes."""
    import pysilk

    payload = silk_bytes[1:] if silk_bytes.startswith(b"\x02#!SILK") else silk_bytes
    if not payload.startswith(b"#!SILK") and b"#!SILK" in payload[:16]:
        idx = payload.find(b"#!SILK")
        payload = payload[idx:]

    in_buf = io.BytesIO(payload)
    pcm_buf = io.BytesIO()
    pysilk.decode(in_buf, pcm_buf, 24000)
    pcm_data = pcm_buf.getvalue()
    if not pcm_data:
        raise ValueError("pysilk decoded empty PCM stream")

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_data)
    return wav_buf.getvalue()


async def _run_ffmpeg_transcode(
    ffmpeg_path: str,
    raw_bytes: bytes,
    *,
    input_suffix: str = ".bin",
    output_suffix: str,
    output_args: tuple[str, ...],
    timeout_sec: float | None = None,
) -> bytes:
    """Execute async ffmpeg transcode with strict temp-file cleanup and timeout kill."""
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is not available")
    effective_timeout = _SUBPROCESS_TIMEOUT_SEC if timeout_sec is None else timeout_sec

    in_fd, in_path = tempfile.mkstemp(prefix=_TEMP_PREFIX + "in_", suffix=input_suffix or ".bin")
    out_fd, out_path = tempfile.mkstemp(prefix=_TEMP_PREFIX + "out_", suffix=output_suffix)
    os.close(in_fd)
    os.close(out_fd)

    def _write_in() -> None:
        with open(in_path, "wb") as fh:
            fh.write(raw_bytes)

    proc: asyncio.subprocess.Process | None = None
    try:
        await asyncio.to_thread(_write_in)
        kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        proc = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            in_path,
            *output_args,
            out_path,
            **kwargs,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise RuntimeError("ffmpeg transcoding timed out") from exc

        if proc.returncode != 0:
            err_msg = (stderr or b"").decode(errors="ignore")[:200]
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {err_msg}")

        def _read_out() -> bytes:
            with open(out_path, "rb") as fh:
                return fh.read()

        data = await asyncio.to_thread(_read_out)
        if not data:
            raise RuntimeError(f"ffmpeg produced empty {output_suffix} output")
        return data
    finally:
        for fp in (in_path, out_path):
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except OSError:
                pass


async def _transcode_bytes_with_ffmpeg(
    ffmpeg_path: str,
    raw_bytes: bytes,
    *,
    input_suffix: str = ".bin",
    timeout_sec: float | None = None,
) -> bytes:
    """Transcode arbitrary audio bytes to 16kHz mono 64kbps MP3 asynchronously."""
    return await _run_ffmpeg_transcode(
        ffmpeg_path,
        raw_bytes,
        input_suffix=input_suffix,
        output_suffix=".mp3",
        output_args=("-c:a", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k"),
        timeout_sec=timeout_sec,
    )


async def prepare_audio_for_deepseek(
    audio: ResolvedAudio,
    *,
    ffmpeg_path: str = "",
    timeout_sec: float | None = None,
) -> tuple[PreparedDeepSeekAudio | None, str]:
    """Adapt resolved audio to DeepSeek ASR limits (16-bit PCM WAV or Ogg Opus, <= 8 MiB, <= 120s)."""
    try:
        primary_bytes = audio.get_bytes()
    except Exception as exc:
        msg = f"invalid audio bytes: {type(exc).__name__}"
        logger.warning(f"[HearVoice] DeepSeek ASR 音频解码失败: {msg}")
        return None, msg

    if not primary_bytes:
        return None, "empty audio payload"

    source_bytes = audio.source_raw_bytes
    wav_candidate = (
        primary_bytes
        if inspect_wav_audio(primary_bytes) is not None
        else (source_bytes if source_bytes and inspect_wav_audio(source_bytes) is not None else None)
    )
    if wav_candidate is not None:
        wav_info = inspect_wav_audio(wav_candidate)
        if wav_info is not None:
            if wav_info.duration_sec > DEEPSEEK_MAX_DURATION_SEC:
                msg = f"duration {wav_info.duration_sec:.1f}s > {int(DEEPSEEK_MAX_DURATION_SEC)}s limit"
                logger.warning(f"[HearVoice] DeepSeek ASR 音频时长超限 ({msg})，已跳过")
                return None, msg
            if wav_info.is_deepseek_compatible and len(wav_candidate) <= DEEPSEEK_MAX_AUDIO_BYTES:
                return (
                    PreparedDeepSeekAudio(
                        data=wav_candidate,
                        filename="speech.wav",
                        content_type="audio/wav",
                        duration_sec=wav_info.duration_sec,
                    ),
                    "",
                )

    ogg_candidate: bytes | None = None
    if source_bytes and is_ogg_opus_bytes(source_bytes):
        ogg_candidate = source_bytes
    elif is_ogg_opus_bytes(primary_bytes):
        ogg_candidate = primary_bytes

    if ogg_candidate is not None:
        ogg_dur = inspect_ogg_opus_duration(ogg_candidate)
        if ogg_dur is not None and ogg_dur > DEEPSEEK_MAX_DURATION_SEC:
            msg = f"duration {ogg_dur:.1f}s > {int(DEEPSEEK_MAX_DURATION_SEC)}s limit"
            logger.warning(f"[HearVoice] DeepSeek ASR Ogg Opus 时长超限 ({msg})，已跳过")
            return None, msg
        if len(ogg_candidate) <= DEEPSEEK_MAX_AUDIO_BYTES:
            return (
                PreparedDeepSeekAudio(
                    data=ogg_candidate,
                    filename="speech.ogg",
                    content_type="audio/ogg",
                    duration_sec=ogg_dur,
                ),
                "",
            )

    if not ffmpeg_path:
        msg = f"ffmpeg unavailable to transcode {audio.audio_format or 'audio'} for DeepSeek ASR"
        logger.warning(
            f"[HearVoice] DeepSeek ASR 需要 ffmpeg 将 {audio.audio_format or 'audio'} "
            "转码为 16-bit PCM WAV / Ogg Opus，但未检测到可用 ffmpeg"
        )
        return None, msg

    transcode_src = source_bytes if source_bytes else primary_bytes
    transcode_fmt = audio.source_format or audio.audio_format or detect_audio_format_bytes(transcode_src)
    in_suffix = f".{transcode_fmt}" if transcode_fmt and transcode_fmt != "unknown" else ".bin"

    try:
        wav_out = await _run_ffmpeg_transcode(
            ffmpeg_path,
            transcode_src,
            input_suffix=in_suffix,
            output_suffix=".wav",
            output_args=("-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1"),
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        msg = f"ffmpeg wav transcode failed: {exc}"
        logger.warning(f"[HearVoice] DeepSeek ASR WAV 转码失败: {msg}")
        return None, msg

    wav_out_info = inspect_wav_audio(wav_out)
    if wav_out_info is not None and wav_out_info.duration_sec > DEEPSEEK_MAX_DURATION_SEC:
        msg = f"duration {wav_out_info.duration_sec:.1f}s > {int(DEEPSEEK_MAX_DURATION_SEC)}s limit"
        logger.warning(f"[HearVoice] DeepSeek ASR 音频时长超限 ({msg})，已跳过")
        return None, msg

    if len(wav_out) <= DEEPSEEK_MAX_AUDIO_BYTES:
        return (
            PreparedDeepSeekAudio(
                data=wav_out,
                filename="speech.wav",
                content_type="audio/wav",
                duration_sec=wav_out_info.duration_sec if wav_out_info else None,
            ),
            "",
        )

    try:
        ogg_out = await _run_ffmpeg_transcode(
            ffmpeg_path,
            transcode_src,
            input_suffix=in_suffix,
            output_suffix=".ogg",
            output_args=("-c:a", "libopus", "-ar", "48000", "-ac", "1", "-b:a", "32k"),
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        msg = f"ffmpeg opus fallback failed: {exc}"
        logger.warning(f"[HearVoice] DeepSeek ASR Ogg Opus 回退转码失败: {msg}")
        return None, msg

    ogg_dur = inspect_ogg_opus_duration(ogg_out)
    if ogg_dur is not None and ogg_dur > DEEPSEEK_MAX_DURATION_SEC:
        msg = f"duration {ogg_dur:.1f}s > {int(DEEPSEEK_MAX_DURATION_SEC)}s limit"
        logger.warning(f"[HearVoice] DeepSeek ASR Ogg Opus 时长超限 ({msg})，已跳过")
        return None, msg

    if len(ogg_out) > DEEPSEEK_MAX_AUDIO_BYTES:
        msg = f"size {len(ogg_out)} bytes > {DEEPSEEK_MAX_AUDIO_BYTES} bytes limit"
        logger.warning(f"[HearVoice] DeepSeek ASR 转码后体积仍超过 8 MiB ({msg})，已跳过")
        return None, msg

    return (
        PreparedDeepSeekAudio(
            data=ogg_out,
            filename="speech.ogg",
            content_type="audio/ogg",
            duration_sec=ogg_dur,
        ),
        "",
    )


async def encode_audio_bytes(
    raw_bytes: bytes,
    *,
    suffix: str = "",
    cfg: HearVoiceConfig,
    ffmpeg_path: str = "",
    source_kind: Literal["direct", "quoted"] = "direct",
) -> ResolvedAudio | None:
    """Normalize raw audio bytes into base64 payload within configured size limits."""
    max_bytes = cfg.max_audio_mb * 1024 * 1024
    if not raw_bytes:
        return None
    if len(raw_bytes) > max_bytes:
        logger.warning(
            f"[HearVoice] 音频体积超限 ({len(raw_bytes)} bytes > {max_bytes} bytes)，已拒收"
        )
        return None

    fmt = detect_audio_format_bytes(raw_bytes, suffix=suffix)

    if fmt == "mp3":
        b64 = await asyncio.to_thread(lambda: base64.b64encode(raw_bytes).decode("ascii"))
        return ResolvedAudio(
            b64, "audio/mpeg", "mp3", len(raw_bytes), source_kind, raw_bytes=raw_bytes
        )

    if fmt == "wav":
        b64 = await asyncio.to_thread(lambda: base64.b64encode(raw_bytes).decode("ascii"))
        return ResolvedAudio(
            b64, "audio/wav", "wav", len(raw_bytes), source_kind, raw_bytes=raw_bytes
        )

    if fmt == "silk":
        try:
            wav_bytes = await asyncio.to_thread(_decode_silk_bytes_to_wav_sync, raw_bytes)
        except Exception as exc:
            logger.warning(f"[HearVoice] SILK 语音解码失败: {type(exc).__name__}")
            return None
        if len(wav_bytes) <= max_bytes:
            b64 = await asyncio.to_thread(lambda: base64.b64encode(wav_bytes).decode("ascii"))
            return ResolvedAudio(
                b64, "audio/wav", "wav", len(wav_bytes), source_kind, raw_bytes=wav_bytes
            )
        if ffmpeg_path:
            try:
                mp3_bytes = await _transcode_bytes_with_ffmpeg(ffmpeg_path, wav_bytes, input_suffix=".wav")
            except Exception as exc:
                logger.warning(f"[HearVoice] SILK 解码后 WAV 转码 MP3 失败: {type(exc).__name__}")
                return None
            if len(mp3_bytes) <= max_bytes:
                b64 = await asyncio.to_thread(lambda: base64.b64encode(mp3_bytes).decode("ascii"))
                return ResolvedAudio(
                    b64,
                    "audio/mpeg",
                    "mp3",
                    len(mp3_bytes),
                    source_kind,
                    raw_bytes=mp3_bytes,
                    source_raw_bytes=wav_bytes,
                    source_format="wav",
                )
        logger.warning("[HearVoice] SILK 解码后的 WAV 超出体积上限且缺少可用 ffmpeg")
        return None

    if fmt in {"ogg", "opus"} and is_ogg_opus_bytes(raw_bytes) and not ffmpeg_path:
        b64 = await asyncio.to_thread(lambda: base64.b64encode(raw_bytes).decode("ascii"))
        return ResolvedAudio(
            b64,
            "audio/ogg",
            "ogg",
            len(raw_bytes),
            source_kind,
            raw_bytes=raw_bytes,
            source_raw_bytes=raw_bytes,
            source_format="ogg",
        )

    if fmt in {"amr", "ogg", "opus", "aac", "m4a", "flac"}:
        if not ffmpeg_path:
            logger.warning(f"[HearVoice] 音频格式 {fmt} 需要 ffmpeg 转码，但未检测到可用 ffmpeg")
            return None
        try:
            mp3_bytes = await _transcode_bytes_with_ffmpeg(
                ffmpeg_path, raw_bytes, input_suffix=f".{fmt}"
            )
        except Exception as exc:
            logger.warning(f"[HearVoice] ffmpeg 转码 {fmt} 失败: {type(exc).__name__}: {exc}")
            return None
        if len(mp3_bytes) <= max_bytes:
            b64 = await asyncio.to_thread(lambda: base64.b64encode(mp3_bytes).decode("ascii"))
            return ResolvedAudio(
                b64,
                "audio/mpeg",
                "mp3",
                len(mp3_bytes),
                source_kind,
                raw_bytes=mp3_bytes,
                source_raw_bytes=raw_bytes,
                source_format=fmt,
            )
        logger.warning(f"[HearVoice] {fmt} 转码后体积仍超过 {max_bytes} bytes 上限")
        return None

    logger.warning(f"[HearVoice] 无法识别的音频格式 (suffix={suffix or 'none'}, bytes={len(raw_bytes)})")
    return None


async def _read_bounded_response_bytes(resp: aiohttp.ClientResponse, max_bytes: int) -> bytes | None:
    content_length = resp.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                return None
        except ValueError:
            pass
    buf = bytearray()
    async for chunk in resp.content.iter_chunked(65536):
        buf.extend(chunk)
        if len(buf) > max_bytes:
            return None
    return bytes(buf)


async def download_remote_audio_bytes(url: str, cfg: HearVoiceConfig) -> tuple[bytes | None, str]:
    """Download remote HTTP(S) audio safely with SSRF checks, redirect validation, and size caps."""
    max_bytes = cfg.max_audio_mb * 1024 * 1024
    current_url = url

    for _hop in range(_MAX_REDIRECT_HOPS + 1):
        safe, target = await prepare_remote_target(current_url, cfg)
        if not safe or not target:
            return None, ""

        host = target["host"]
        ips = target["ips"]
        resolver = StaticResolver({host: ips})
        connector = aiohttp.TCPConnector(resolver=resolver, ssl=cfg.verify_ssl, limit=4)
        timeout = aiohttp.ClientTimeout(total=cfg.timeout_sec)

        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as sess:
                async with sess.get(
                    current_url,
                    allow_redirects=False,
                    headers={"Host": host},
                ) as resp:
                    if 300 <= resp.status < 400:
                        loc = resp.headers.get("Location", "").strip()
                        if not loc or _hop >= _MAX_REDIRECT_HOPS:
                            return None, ""
                        current_url = urljoin(current_url, loc)
                        continue
                    if resp.status != 200:
                        return None, ""
                    data = await _read_bounded_response_bytes(resp, max_bytes)
                    if not data:
                        return None, ""
                    suffix = os.path.splitext(urlparse(current_url).path)[1]
                    return data, suffix
        except Exception:
            return None, ""
        finally:
            await resolver.close()

    return None, ""


async def download_trusted_loopback_url(url: str, cfg: HearVoiceConfig) -> tuple[bytes | None, str]:
    """Download a NapCat loopback callback URL returned directly by get_record/get_file."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        return None, ""
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return None, ""

    if cfg.block_private_network:
        if not is_loopback_host(host) or not cfg.allow_napcat_local_record_url:
            return None, ""

    max_bytes = cfg.max_audio_mb * 1024 * 1024
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_sec)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as sess:
            async with sess.get(url, allow_redirects=False) as resp:
                if resp.status != 200:
                    return None, ""
                data = await _read_bounded_response_bytes(resp, max_bytes)
                if not data:
                    return None, ""
                suffix = os.path.splitext(parsed.path)[1]
                return data, suffix
    except Exception:
        return None, ""


def _decode_inline_base64_ref(ref: str, max_bytes: int) -> tuple[bytes | None, str]:
    raw = (ref or "").strip()
    if raw.startswith("base64://"):
        payload = raw[len("base64://") :].strip()
        suffix = ""
    elif raw.lower().startswith("data:"):
        header, sep, payload = raw.partition(",")
        if not sep or ";base64" not in header.lower():
            return None, ""
        mime = header[5:].split(";", 1)[0].strip().lower()
        suffix = {
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/ogg": ".ogg",
            "audio/opus": ".opus",
            "audio/amr": ".amr",
            "audio/flac": ".flac",
            "audio/aac": ".aac",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
        }.get(mime, "")
    else:
        return None, ""

    if not payload or len(payload) > (max_bytes * 4) // 3 + 64:
        return None, ""
    try:
        decoded = base64.b64decode(payload, validate=False)
    except Exception:
        return None, ""
    if not decoded or len(decoded) > max_bytes:
        return None, ""
    return decoded, suffix


def _extract_record_tokens(record_comp: Any) -> list[str]:
    tokens: list[str] = []
    for key in ("file", "id", "file_id", "path", "url"):
        val = getattr(record_comp, key, None)
        if val and isinstance(val, str):
            s = val.strip()
            if s and not s.startswith(("base64://", "data:")) and s not in tokens:
                tokens.append(s)
    data = getattr(record_comp, "data", None)
    if isinstance(data, dict):
        for key in ("file", "id", "file_id", "path", "url"):
            val = data.get(key)
            if val and isinstance(val, str):
                s = val.strip()
                if s and not s.startswith(("base64://", "data:")) and s not in tokens:
                    tokens.append(s)
    return tokens


async def _call_onebot_fallback(event: Any, record_comp: Any, cfg: HearVoiceConfig) -> tuple[bytes | None, str]:
    """Try OneBot/NapCat get_record, then get_file if needed, returning (bytes, suffix)."""
    if not cfg.enable_get_record_fallback:
        return None, ""
    bot = getattr(event, "bot", None)
    api = getattr(bot, "api", None)
    call_action = getattr(api, "call_action", None)
    if not callable(call_action):
        return None, ""

    tokens = _extract_record_tokens(record_comp)
    if not tokens:
        return None, ""

    attempts: list[tuple[str, dict[str, Any]]] = []
    for token in tokens[:2]:
        attempts.append(("get_record", {"file": token, "out_format": "mp3"}))
        attempts.append(("get_file", {"file": token}))
        attempts.append(("get_file", {"file_id": token}))

    max_bytes = cfg.max_audio_mb * 1024 * 1024
    for action, params in attempts:
        try:
            result = await call_action(action, **params)
        except Exception:
            continue
        if not isinstance(result, dict):
            continue

        data_inner = result.get("data")
        if isinstance(data_inner, str) and data_inner.strip():
            remapped = remap_local_path(data_inner.strip(), cfg)
            if is_safe_local_audio_path(remapped, cfg):
                raw_bytes = await asyncio.to_thread(_read_file_bounded_sync, remapped, max_bytes)
                if raw_bytes:
                    return raw_bytes, os.path.splitext(remapped)[1]
            continue

        lookup = data_inner if isinstance(data_inner, dict) and data_inner else result

        b64 = str(lookup.get("base64", "") or "").strip()
        if b64 and len(b64) <= (max_bytes * 4) // 3 + 64:
            try:
                decoded = await asyncio.to_thread(base64.b64decode, b64)
                if decoded and len(decoded) <= max_bytes:
                    return decoded, ".mp3" if action == "get_record" else ""
            except Exception:
                pass

        for key in ("file", "path", "url", "file_path", "localPath", "local_path", "filename"):
            target = lookup.get(key)
            if not target or not isinstance(target, str):
                continue
            target = target.strip()
            if not target:
                continue
            if target.startswith(("http://", "https://")):
                host = urlparse(target).hostname or ""
                if is_loopback_host(host):
                    data_bytes, suffix = await download_trusted_loopback_url(target, cfg)
                else:
                    data_bytes, suffix = await download_remote_audio_bytes(target, cfg)
                if data_bytes:
                    return data_bytes, suffix
                continue
            if target.startswith("file://"):
                target = _file_uri_to_local_path(target)
            remapped = remap_local_path(target, cfg)
            if is_safe_local_audio_path(remapped, cfg):
                raw_bytes = await asyncio.to_thread(_read_file_bounded_sync, remapped, max_bytes)
                if raw_bytes:
                    return raw_bytes, os.path.splitext(remapped)[1]

    return None, ""


def _read_file_bounded_sync(path: str, max_bytes: int) -> bytes | None:
    try:
        size = os.path.getsize(path)
        if size <= 0 or size > max_bytes:
            return None
        with open(path, "rb") as fh:
            return fh.read(max_bytes + 1)
    except OSError:
        return None


def _candidate_refs(record_comp: Any) -> list[str]:
    refs: list[str] = []
    for key in ("path", "file", "url"):
        val = getattr(record_comp, key, None)
        if val and isinstance(val, str):
            s = val.strip().strip('"').strip("'")
            if s and s not in refs:
                refs.append(s)
    data = getattr(record_comp, "data", None)
    if isinstance(data, dict):
        for key in ("path", "file", "url"):
            val = data.get(key)
            if val and isinstance(val, str):
                s = val.strip().strip('"').strip("'")
                if s and s not in refs:
                    refs.append(s)
    return refs


def _looks_like_filesystem_path(ref: str) -> bool:
    if not ref or ref.startswith(("http://", "https://", "base64://", "data:")):
        return False
    if ref.startswith("file://"):
        return True
    if "/" in ref or "\\" in ref:
        return True
    return len(ref) >= 2 and ref[1] == ":"


async def _wait_for_local_path(path: str, wait_sec: int) -> bool:
    if await asyncio.to_thread(os.path.exists, path):
        return True
    if wait_sec <= 0:
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_sec
    while loop.time() < deadline:
        await asyncio.sleep(0.25)
        if await asyncio.to_thread(os.path.exists, path):
            return True
    return False


async def resolve_voice_audio(
    event: Any,
    voice: VoiceSource,
    cfg: HearVoiceConfig,
    *,
    ffmpeg_path: str = "",
) -> ResolvedAudio | None:
    """Resolve and normalize a single Record component into encoded audio bytes."""
    max_bytes = cfg.max_audio_mb * 1024 * 1024
    refs = _candidate_refs(voice.component)

    # 1. Inline base64:// or data: URI payloads
    for ref in refs:
        if ref.startswith(("base64://", "data:")):
            raw_bytes, suffix = await asyncio.to_thread(_decode_inline_base64_ref, ref, max_bytes)
            if raw_bytes:
                return await encode_audio_bytes(
                    raw_bytes, suffix=suffix, cfg=cfg, ffmpeg_path=ffmpeg_path, source_kind=voice.source_kind
                )

    # 2. Existing local file paths / file:// URIs (including PreProcessStage WAVs)
    for ref in refs:
        if ref.startswith(("http://", "https://", "base64://", "data:")):
            continue
        candidate = _file_uri_to_local_path(ref) if ref.startswith("file://") else ref
        remapped = remap_local_path(candidate, cfg)
        wait = cfg.voice_file_wait_sec if _looks_like_filesystem_path(ref) else 0
        if await _wait_for_local_path(remapped, wait):
            if await asyncio.to_thread(is_safe_local_audio_path, remapped, cfg):
                raw_bytes = await asyncio.to_thread(_read_file_bounded_sync, remapped, max_bytes)
                if raw_bytes and len(raw_bytes) <= max_bytes:
                    return await encode_audio_bytes(
                        raw_bytes,
                        suffix=os.path.splitext(remapped)[1],
                        cfg=cfg,
                        ffmpeg_path=ffmpeg_path,
                        source_kind=voice.source_kind,
                    )

    # 3. Remote HTTP(S) URLs (SSRF-guarded)
    for ref in refs:
        if ref.startswith(("http://", "https://")):
            raw_bytes, suffix = await download_remote_audio_bytes(ref, cfg)
            if raw_bytes:
                return await encode_audio_bytes(
                    raw_bytes, suffix=suffix, cfg=cfg, ffmpeg_path=ffmpeg_path, source_kind=voice.source_kind
                )

    # 4. OneBot/NapCat get_record / get_file fallback
    raw_bytes, suffix = await _call_onebot_fallback(event, voice.component, cfg)
    if raw_bytes:
        return await encode_audio_bytes(
            raw_bytes, suffix=suffix, cfg=cfg, ffmpeg_path=ffmpeg_path, source_kind=voice.source_kind
        )

    return None


async def resolve_audio_ref(
    audio_ref: str,
    cfg: HearVoiceConfig,
    *,
    ffmpeg_path: str = "",
) -> ResolvedAudio | None:
    """Resolve a standalone audio path/URL/base64 reference (used by STTProvider)."""
    stub = type("_RefHolder", (), {"path": audio_ref, "file": audio_ref, "url": audio_ref})()
    voice = VoiceSource(index=0, component=stub, source_kind="direct")
    return await resolve_voice_audio(None, voice, cfg, ffmpeg_path=ffmpeg_path)


def _component_type_name(comp: Any) -> str:
    cname = type(comp).__name__
    if cname in ("Record", "Reply", "Plain"):
        return cname
    ctype = getattr(comp, "type", None)
    val = getattr(ctype, "value", ctype)
    return str(val or cname)


def extract_voice_sources(
    event: Any,
    *,
    handle_quoted_voice: bool = True,
    max_count: int = 3,
) -> list[VoiceSource]:
    """Collect direct and quoted Record components in order without mutating the chain."""
    getter = getattr(event, "get_messages", None)
    messages: list[Any] = []
    if callable(getter):
        try:
            res = getter()
            if isinstance(res, (list, tuple)):
                messages = list(res)
        except Exception:
            messages = []
    if not messages:
        msg_obj = getattr(event, "message_obj", None)
        raw_chain = getattr(msg_obj, "message", None)
        if isinstance(raw_chain, (list, tuple)):
            messages = list(raw_chain)

    found: list[VoiceSource] = []
    seen_ids: set[int] = set()
    limit = max(1, max_count)

    # Direct voice segments take priority over quoted Reply.chain segments.
    for comp in messages:
        if len(found) >= limit:
            break
        if _component_type_name(comp) == "Record":
            cid = id(comp)
            if cid not in seen_ids:
                seen_ids.add(cid)
                found.append(VoiceSource(index=len(found), component=comp, source_kind="direct"))

    if handle_quoted_voice and len(found) < limit:
        for comp in messages:
            if len(found) >= limit:
                break
            if _component_type_name(comp) != "Reply":
                continue
            chain = getattr(comp, "chain", None)
            if isinstance(chain, (list, tuple)):
                for qcomp in chain:
                    if len(found) >= limit:
                        break
                    if _component_type_name(qcomp) == "Record":
                        qcid = id(qcomp)
                        if qcid not in seen_ids:
                            seen_ids.add(qcid)
                            found.append(
                                VoiceSource(index=len(found), component=qcomp, source_kind="quoted")
                            )
    return found


def cleanup_temp_files(*, older_than_sec: int = 0) -> int:
    """Delete plugin-owned hvstt_* temp files older than ``older_than_sec``."""
    now = time.time()
    removed = 0
    for d in _temp_dirs_set():
        if not os.path.isdir(d):
            continue
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if not name.startswith(_TEMP_PREFIX):
                continue
            path = os.path.join(d, name)
            if not os.path.isfile(path):
                continue
            try:
                if older_than_sec > 0 and (now - os.path.getmtime(path)) < older_than_sec:
                    continue
                os.remove(path)
                removed += 1
            except OSError:
                continue
    return removed


def describe_safe_ref(ref: str) -> str:
    """Return a log-safe description of a URL or file reference."""
    if not ref:
        return ""
    if ref.startswith(("http://", "https://")):
        return redact_url(ref)
    if ref.startswith(("base64://", "data:")):
        return f"inline<{len(ref)} chars>"
    return os.path.basename(ref)
