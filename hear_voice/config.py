"""Configuration schema, validation, path allowlist discovery, and credential redaction."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_DEEPSEEK_ASR_URL = "https://api.deepseek.com/v1/tools/asr"
DEFAULT_OPENAI_WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_WHISPER_MODEL = "whisper-1"

NAPCAT_KNOWN_SOURCES: tuple[str, ...] = (
    "/app/.config/QQ",
    "/app/QQ",
    "/opt/QQ",
)

NAPCAT_KNOWN_DESTS: tuple[str, ...] = (
    "/root/astrbot/ntqq",
    "/root/.config/QQ",
    "/home/user/.config/QQ",
    "/var/lib/QQ",
    "/data/QQ",
    "/app/.config/QQ",
)

NAPCAT_COMMON_ROOTS: tuple[str, ...] = (
    "/app/.config/QQ",
    "/root/astrbot/ntqq",
    "/root/.config/QQ",
    "/home/user/.config/QQ",
    "/var/lib/QQ",
    "/data/QQ",
    "/opt/QQ",
)

VALID_ENGINES = frozenset({"auto", "gemini", "deepseek_asr", "openai_whisper"})
VALID_API_FORMATS = frozenset({"auto", "gemini_native", "openai_compat"})
VALID_AUTH_MODES = frozenset({"auto", "x_goog_api_key", "bearer"})
VALID_OUTPUT_MODES = frozenset({"simple", "rich"})
VALID_FAIL_POLICIES = frozenset({"pass", "block", "notify", "notify_pass"})

_BEARER_TOKEN_RE = re.compile(r"(?i)(Bearer\s+)[^\s,;\"']+")
_HEADER_SECRET_RE = re.compile(
    r"(?i)((?:x-goog-api-key|authorization|api[-_]?key|access[-_]?token|token|secret)"
    r"\s*[:=]\s*[\"']?)[^\s,;\"'}]+"
)
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:key|api_key|token|access_token|secret|auth|sig|signature)=)[^&\s\"'>]+"
)
_PREFIXED_KEY_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9._-]{6,}|AIza[0-9A-Za-z\-_]{10,}|gh[pousr]_[A-Za-z0-9_]{10,})"
)


def redact_secret(value: str | None) -> str:
    """Return a masked representation of a secret without exposing credentials."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= 8:
        return "***"
    return f"***{raw[-4:]}"


def redact_url(url: str | None) -> str:
    """Strip query parameters, fragments, and userinfo from a URL before logging."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        if not parsed.scheme:
            return raw.split("?", 1)[0]
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}" if ":" not in netloc else f"[{netloc}]:{parsed.port}"
        return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
    except Exception:
        return raw.split("?", 1)[0]


def sanitize_error_message(
    raw: str,
    *,
    secrets: tuple[str, ...] = (),
    max_len: int = 240,
) -> str:
    """Extract, redact credentials from, and truncate an upstream error payload."""
    summary = (raw or "").strip()
    if summary:
        try:
            parsed = json.loads(summary)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                parts = [
                    str(err.get(k) or "").strip()
                    for k in ("code", "status", "type")
                    if err.get(k) not in (None, "")
                ]
                msg = str(err.get("message") or "").strip()
                prefix = " ".join(parts)
                if prefix and msg:
                    summary = f"{prefix}: {msg}"
                elif msg or prefix:
                    summary = msg or prefix
            elif err not in (None, ""):
                summary = str(err)
            else:
                for key in ("message", "detail", "msg"):
                    if parsed.get(key) not in (None, ""):
                        summary = str(parsed[key])
                        break

    compact = re.sub(r"\s+", " ", summary).strip()
    for secret in secrets:
        s = (secret or "").strip()
        if len(s) >= 6 and s in compact:
            compact = compact.replace(s, redact_secret(s) or "***")
    compact = _BEARER_TOKEN_RE.sub(r"\1***", compact)
    compact = _HEADER_SECRET_RE.sub(r"\1***", compact)
    compact = _QUERY_SECRET_RE.sub(r"\1***", compact)
    compact = _PREFIXED_KEY_RE.sub("***", compact)

    if max_len > 0 and len(compact) > max_len:
        return compact[:max_len].rstrip() + "..."
    return compact or "<empty>"


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_enum(
    value: Any,
    valid: frozenset[str],
    default: str,
    *,
    name: str,
    notices: list[str],
) -> str:
    parsed = str(default if value is None or value == "" else value).strip().lower()
    if parsed not in valid:
        notices.append(f"{name}={parsed!r} 无效，已回退 {default}")
        return default
    return parsed


def _coerce_int(
    value: Any,
    default: int,
    *,
    min_value: int,
    max_value: int,
    name: str,
    notices: list[str],
) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError("bool is not an int")
        parsed = int(value)
    except (TypeError, ValueError):
        notices.append(f"配置项 {name} 值无效，已回退默认 {default}")
        return default
    if parsed < min_value:
        notices.append(f"配置项 {name}={parsed} 小于下限 {min_value}，已收敛为 {min_value}")
        return min_value
    if parsed > max_value:
        notices.append(f"配置项 {name}={parsed} 大于上限 {max_value}，已收敛为 {max_value}")
        return max_value
    return parsed


def _coerce_float(
    value: Any,
    default: float,
    *,
    min_value: float,
    max_value: float,
    name: str,
    notices: list[str],
) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError("bool is not a float")
        parsed = float(value)
    except (TypeError, ValueError):
        notices.append(f"配置项 {name} 值无效，已回退默认 {default}")
        return default
    if parsed < min_value:
        notices.append(f"配置项 {name}={parsed} 小于下限 {min_value}，已收敛为 {min_value}")
        return min_value
    if parsed > max_value:
        notices.append(f"配置项 {name}={parsed} 大于上限 {max_value}，已收敛为 {max_value}")
        return max_value
    return parsed


def _coerce_str_list(value: Any, *, lowercase: bool = False) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if lowercase:
            text = text.lower()
        if text and text not in out:
            out.append(text)
    return out


def _resolve_astrbot_temp_dir() -> str | None:
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

        path = str(get_astrbot_temp_path() or "").strip()
        if path:
            return os.path.realpath(path)
    except Exception:
        return None
    return None


def _normalize_allowed_dirs(raw_dirs: list[str]) -> list[str]:
    candidates: list[str] = list(raw_dirs or [])
    candidates.append(os.path.abspath("data"))
    candidates.append(os.path.abspath("data/temp"))
    candidates.append(tempfile.gettempdir())
    astr_temp = _resolve_astrbot_temp_dir()
    if astr_temp:
        candidates.append(astr_temp)

    out: list[str] = []
    for entry in candidates:
        text = str(entry or "").strip()
        if not text:
            continue
        try:
            rp = os.path.realpath(text)
        except OSError:
            continue
        if rp not in out:
            out.append(rp)

    for root in NAPCAT_COMMON_ROOTS:
        try:
            rp = os.path.realpath(root)
            if os.path.isdir(rp) and rp not in out:
                out.append(rp)
        except OSError:
            continue
    return out


def build_auto_remap_pairs(path_remap_from: str, path_remap_to: str) -> list[tuple[str, str]]:
    """Build automatic NapCat -> AstrBot path remap pairs when manual remap is unset."""
    if path_remap_from and path_remap_to:
        return []
    pairs: list[tuple[str, str]] = []
    for src in NAPCAT_KNOWN_SOURCES:
        for dst in NAPCAT_KNOWN_DESTS:
            if src == dst:
                continue
            try:
                if os.path.isdir(os.path.realpath(dst)):
                    pair = (src.rstrip("/\\"), dst.rstrip("/\\"))
                    if pair not in pairs:
                        pairs.append(pair)
            except OSError:
                continue
    return pairs


@dataclass(frozen=True)
class SlotRawConfig:
    """Normalized raw configuration values for either the primary or fallback STT slot."""

    slot: Literal["primary", "fallback"]
    engine: str
    provider_id: str
    api_format: str
    api_base: str
    api_key: str
    model: str
    auth_mode: str
    proxy: str


def _parse_slot_fields(
    src: dict[str, Any],
    *,
    slot: Literal["primary", "fallback"],
    notices: list[str],
) -> SlotRawConfig:
    """Parse primary or fallback endpoint config uniformly with legacy alias support."""
    prefix = "" if slot == "primary" else "fallback_"
    engine_key = "provider_engine" if slot == "primary" else "fallback_engine"
    format_key = f"{prefix}api_format"
    legacy_format_key = f"{prefix}api_mode"
    base_key = f"{prefix}api_base"
    legacy_base_key = f"{prefix}api_url"

    raw_format = src.get(format_key, src.get(legacy_format_key, "auto"))
    raw_base = src.get(base_key, src.get(legacy_base_key, ""))

    return SlotRawConfig(
        slot=slot,
        engine=_coerce_enum(
            src.get(engine_key, "auto"),
            VALID_ENGINES,
            "auto",
            name=engine_key,
            notices=notices,
        ),
        provider_id=str(src.get(f"{prefix}provider_id", "") or "").strip(),
        api_format=_coerce_enum(
            raw_format,
            VALID_API_FORMATS,
            "auto",
            name=format_key,
            notices=notices,
        ),
        api_base=str(raw_base or "").strip(),
        api_key=str(src.get(f"{prefix}api_key", "") or "").strip(),
        model=str(src.get(f"{prefix}model", "") or "").strip(),
        auth_mode=_coerce_enum(
            src.get(f"{prefix}auth_mode", "auto"),
            VALID_AUTH_MODES,
            "auto",
            name=f"{prefix}auth_mode",
            notices=notices,
        ),
        proxy=str(src.get(f"{prefix}proxy", "") or "").strip(),
    )


@dataclass
class HearVoiceConfig:
    """Validated runtime configuration for the non-hijacking multi-engine HearVoice plugin."""

    enable_voice: bool = True

    # Primary slot
    provider_engine: str = "auto"
    provider_id: str = ""
    api_format: str = "auto"
    auth_mode: str = "auto"
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    proxy: str = ""
    verify_ssl: bool = True

    # Fallback slot
    enable_fallback: bool = True
    fallback_engine: str = "auto"
    fallback_provider_id: str = ""
    fallback_api_format: str = "auto"
    fallback_auth_mode: str = "auto"
    fallback_api_base: str = ""
    fallback_api_key: str = ""
    fallback_model: str = ""
    fallback_proxy: str = ""

    # Output & normalization
    output_mode: str = "rich"
    enable_model_normalize: bool = False
    enable_transcript_clean: bool = True
    enable_text_normalize: bool = False
    max_transcript_chars: int = 2000
    transcript_prefix: str = "[语音消息]: "
    quoted_transcript_prefix: str = "[引用语音消息]: "
    voice_instruction: str = ""

    # Scope & gate
    enable_private: bool = True
    enable_group_voice: bool = True
    group_whitelist: list[str] = field(default_factory=list)
    user_whitelist: list[str] = field(default_factory=list)
    group_voice_reply_probability: float = 1.0
    handle_quoted_voice: bool = True
    max_voice_count: int = 3
    skip_if_already_transcribed: bool = True
    skip_commands: bool = True

    # Stage 2 & failure behavior
    attach_voice_marker: bool = True
    attach_speaker_meta: bool = True
    inject_rich_context_on_llm_request: bool = True
    cleanup_audio_attachments: bool = True
    group_voice_export_context: bool = True
    on_stt_fail: str = "pass"
    show_transcript: bool = False

    # Audio & HTTP resilience
    stt_max_concurrency: int = 3
    max_audio_mb: int = 20
    timeout_sec: int = 120
    retry_times: int = 2
    voice_file_wait_sec: int = 5
    ffmpeg_path: str = ""
    enable_get_record_fallback: bool = True
    allow_napcat_local_record_url: bool = True
    path_remap_from: str = ""
    path_remap_to: str = ""

    # SSRF & local file security
    allow_remote_audio_url: bool = False
    remote_audio_domain_whitelist: list[str] = field(default_factory=list)
    block_private_network: bool = True
    strict_local_path_check: bool = True
    local_audio_allowed_dirs: list[str] = field(default_factory=list)

    # Temp cleanup
    enable_temp_cleanup: bool = True
    temp_cleanup_on_start: bool = True
    temp_cleanup_interval_sec: int = 1800
    temp_cleanup_max_age_sec: int = 300
    temp_cleanup_on_terminate: bool = True

    # Optional STT provider adapter & debug
    register_stt_provider: bool = False
    stt_provider_id: str = "hear_voice_stt"
    debug_mode: bool = False

    auto_remap_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def api_mode(self) -> str:
        """Backward-compatible alias for ``api_format``."""
        return self.api_format

    @property
    def api_url(self) -> str:
        """Backward-compatible alias for ``api_base``."""
        return self.api_base

    def get_slot_raw(self, slot: Literal["primary", "fallback"]) -> SlotRawConfig:
        """Return normalized raw slot fields for either ``primary`` or ``fallback``."""
        if slot == "primary":
            return SlotRawConfig(
                slot="primary",
                engine=self.provider_engine,
                provider_id=self.provider_id,
                api_format=self.api_format,
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.model,
                auth_mode=self.auth_mode,
                proxy=self.proxy,
            )
        return SlotRawConfig(
            slot="fallback",
            engine=self.fallback_engine,
            provider_id=self.fallback_provider_id,
            api_format=self.fallback_api_format,
            api_base=self.fallback_api_base,
            api_key=self.fallback_api_key,
            model=self.fallback_model,
            auth_mode=self.fallback_auth_mode,
            proxy=self.fallback_proxy or self.proxy,
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "HearVoiceConfig":
        cfg, _ = cls.from_mapping_detailed(raw)
        return cfg

    @classmethod
    def from_mapping_detailed(cls, raw: dict[str, Any] | None) -> tuple["HearVoiceConfig", list[str]]:
        src = dict(raw or {})
        notices: list[str] = []

        if "group_whitelist" not in src and "group_voice_whitelist" in src:
            src["group_whitelist"] = src["group_voice_whitelist"]
        if "enable_private" not in src and "enable_private_voice" in src:
            src["enable_private"] = src["enable_private_voice"]

        primary_slot = _parse_slot_fields(src, slot="primary", notices=notices)
        fallback_slot = _parse_slot_fields(src, slot="fallback", notices=notices)

        output_mode = _coerce_enum(
            src.get("output_mode", "rich"),
            VALID_OUTPUT_MODES,
            "rich",
            name="output_mode",
            notices=notices,
        )
        on_stt_fail = _coerce_enum(
            src.get("on_stt_fail", "pass"),
            VALID_FAIL_POLICIES,
            "pass",
            name="on_stt_fail",
            notices=notices,
        )

        path_remap_from = str(src.get("path_remap_from", "") or "").strip()
        path_remap_to = str(src.get("path_remap_to", "") or "").strip()

        raw_allowed_dirs = _coerce_str_list(src.get("local_audio_allowed_dirs", ["data", "data/temp"]))
        allowed_dirs = _normalize_allowed_dirs(raw_allowed_dirs)

        transcript_prefix = src.get("transcript_prefix", "[语音消息]: ")
        if not isinstance(transcript_prefix, str) or not transcript_prefix.strip():
            transcript_prefix = "[语音消息]: "
        quoted_prefix = src.get("quoted_transcript_prefix", "[引用语音消息]: ")
        if not isinstance(quoted_prefix, str) or not quoted_prefix.strip():
            quoted_prefix = "[引用语音消息]: "

        stt_provider_id = (
            str(src.get("stt_provider_id", "hear_voice_stt") or "hear_voice_stt").strip()
            or "hear_voice_stt"
        )

        cfg = cls(
            enable_voice=_coerce_bool(src.get("enable_voice", True), True),
            provider_engine=primary_slot.engine,
            provider_id=primary_slot.provider_id,
            api_format=primary_slot.api_format,
            auth_mode=primary_slot.auth_mode,
            api_base=primary_slot.api_base,
            api_key=primary_slot.api_key,
            model=primary_slot.model,
            proxy=primary_slot.proxy,
            verify_ssl=_coerce_bool(src.get("verify_ssl", True), True),
            enable_fallback=_coerce_bool(src.get("enable_fallback", True), True),
            fallback_engine=fallback_slot.engine,
            fallback_provider_id=fallback_slot.provider_id,
            fallback_api_format=fallback_slot.api_format,
            fallback_auth_mode=fallback_slot.auth_mode,
            fallback_api_base=fallback_slot.api_base,
            fallback_api_key=fallback_slot.api_key,
            fallback_model=fallback_slot.model,
            fallback_proxy=fallback_slot.proxy,
            output_mode=output_mode,
            enable_model_normalize=_coerce_bool(src.get("enable_model_normalize", False), False),
            enable_transcript_clean=_coerce_bool(src.get("enable_transcript_clean", True), True),
            enable_text_normalize=_coerce_bool(src.get("enable_text_normalize", False), False),
            max_transcript_chars=_coerce_int(
                src.get("max_transcript_chars", 2000),
                2000,
                min_value=50,
                max_value=20000,
                name="max_transcript_chars",
                notices=notices,
            ),
            transcript_prefix=transcript_prefix,
            quoted_transcript_prefix=quoted_prefix,
            voice_instruction=str(src.get("voice_instruction", "") or "").strip(),
            enable_private=_coerce_bool(src.get("enable_private", True), True),
            enable_group_voice=_coerce_bool(src.get("enable_group_voice", True), True),
            group_whitelist=_coerce_str_list(src.get("group_whitelist", [])),
            user_whitelist=_coerce_str_list(src.get("user_whitelist", [])),
            group_voice_reply_probability=_coerce_float(
                src.get("group_voice_reply_probability", 1.0),
                1.0,
                min_value=0.0,
                max_value=1.0,
                name="group_voice_reply_probability",
                notices=notices,
            ),
            handle_quoted_voice=_coerce_bool(src.get("handle_quoted_voice", True), True),
            max_voice_count=_coerce_int(
                src.get("max_voice_count", 3),
                3,
                min_value=1,
                max_value=10,
                name="max_voice_count",
                notices=notices,
            ),
            skip_if_already_transcribed=_coerce_bool(src.get("skip_if_already_transcribed", True), True),
            skip_commands=_coerce_bool(src.get("skip_commands", True), True),
            attach_voice_marker=_coerce_bool(src.get("attach_voice_marker", True), True),
            attach_speaker_meta=_coerce_bool(src.get("attach_speaker_meta", True), True),
            inject_rich_context_on_llm_request=_coerce_bool(
                src.get("inject_rich_context_on_llm_request", True), True
            ),
            cleanup_audio_attachments=_coerce_bool(src.get("cleanup_audio_attachments", True), True),
            group_voice_export_context=_coerce_bool(src.get("group_voice_export_context", True), True),
            on_stt_fail=on_stt_fail,
            show_transcript=_coerce_bool(src.get("show_transcript", False), False),
            stt_max_concurrency=_coerce_int(
                src.get("stt_max_concurrency", 3),
                3,
                min_value=1,
                max_value=10,
                name="stt_max_concurrency",
                notices=notices,
            ),
            max_audio_mb=_coerce_int(
                src.get("max_audio_mb", 20),
                20,
                min_value=1,
                max_value=200,
                name="max_audio_mb",
                notices=notices,
            ),
            timeout_sec=_coerce_int(
                src.get("timeout_sec", 120),
                120,
                min_value=5,
                max_value=600,
                name="timeout_sec",
                notices=notices,
            ),
            retry_times=_coerce_int(
                src.get("retry_times", 2),
                2,
                min_value=0,
                max_value=5,
                name="retry_times",
                notices=notices,
            ),
            voice_file_wait_sec=_coerce_int(
                src.get("voice_file_wait_sec", 5),
                5,
                min_value=0,
                max_value=60,
                name="voice_file_wait_sec",
                notices=notices,
            ),
            ffmpeg_path=str(src.get("ffmpeg_path", "") or "").strip(),
            enable_get_record_fallback=_coerce_bool(src.get("enable_get_record_fallback", True), True),
            allow_napcat_local_record_url=_coerce_bool(
                src.get("allow_napcat_local_record_url", True), True
            ),
            path_remap_from=path_remap_from,
            path_remap_to=path_remap_to,
            allow_remote_audio_url=_coerce_bool(src.get("allow_remote_audio_url", False), False),
            remote_audio_domain_whitelist=_coerce_str_list(
                src.get("remote_audio_domain_whitelist", []), lowercase=True
            ),
            block_private_network=_coerce_bool(src.get("block_private_network", True), True),
            strict_local_path_check=_coerce_bool(src.get("strict_local_path_check", True), True),
            local_audio_allowed_dirs=allowed_dirs,
            enable_temp_cleanup=_coerce_bool(src.get("enable_temp_cleanup", True), True),
            temp_cleanup_on_start=_coerce_bool(src.get("temp_cleanup_on_start", True), True),
            temp_cleanup_interval_sec=_coerce_int(
                src.get("temp_cleanup_interval_sec", 1800),
                1800,
                min_value=0,
                max_value=86400,
                name="temp_cleanup_interval_sec",
                notices=notices,
            ),
            temp_cleanup_max_age_sec=_coerce_int(
                src.get("temp_cleanup_max_age_sec", 300),
                300,
                min_value=0,
                max_value=86400,
                name="temp_cleanup_max_age_sec",
                notices=notices,
            ),
            temp_cleanup_on_terminate=_coerce_bool(src.get("temp_cleanup_on_terminate", True), True),
            register_stt_provider=_coerce_bool(src.get("register_stt_provider", False), False),
            stt_provider_id=stt_provider_id,
            debug_mode=_coerce_bool(src.get("debug_mode", False), False),
            auto_remap_pairs=build_auto_remap_pairs(path_remap_from, path_remap_to),
        )
        return cfg, notices
