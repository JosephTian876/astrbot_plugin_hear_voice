"""Unified multi-engine STT orchestration, shared HTTP transport/retries, and failover."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import aiohttp

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover - standalone import fallback
    import logging

    logger = logging.getLogger("astrbot_plugin_hear_voice")

from .audio_resolver import ResolvedAudio, transcode_audio_to_wav_resolved
from .config import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_WHISPER_URL,
    DEFAULT_WHISPER_MODEL,
    HearVoiceConfig,
    SlotRawConfig,
    redact_url,
    sanitize_error_message,
)
from .deepseek_client import DeepSeekAsrClient, normalize_deepseek_asr_url
from .gemini_client import GeminiClient, is_unsupported_audio_format_400
from .normalize import (
    RichSTTResult,
    build_stt_instruction,
    clean_transcript,
    is_instruction_hallucination,
    is_provider_error_text,
    normalize_model_name,
    parse_rich_output,
)

__all__ = [
    "DeepSeekAsrClient",
    "GeminiClient",
    "HearVoiceEngine",
    "OpenAIWhisperClient",
    "SttEndpointSpec",
    "SttHttpTransport",
    "TranscriptionResult",
    "normalize_deepseek_asr_url",
    "normalize_openai_whisper_url",
    "resolve_endpoint",
    "resolve_endpoints",
    "resolve_engine_kind",
    "resolve_slot_endpoint",
]


def _is_retryable_status(status: int) -> bool:
    """Return True only for HTTP 429 rate limits and 5xx server failures."""
    return status == 429 or status >= 500


def normalize_openai_whisper_url(api_base: str | None) -> str:
    """Normalize an OpenAI-compatible base URL to ``/v1/audio/transcriptions``."""
    raw = (api_base or "").strip().rstrip("/")
    if not raw:
        return DEFAULT_OPENAI_WHISPER_URL
    parsed = urlparse(raw)
    if parsed.path.lower().endswith("/audio/transcriptions"):
        return raw
    lower = raw.lower()
    if lower.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")].rstrip("/")
        lower = raw.lower()
    if lower.endswith("/v1"):
        return f"{raw}/audio/transcriptions"
    return f"{raw}/v1/audio/transcriptions"


def resolve_engine_kind(
    configured_engine: str = "auto",
    *,
    provider_type: str = "",
    api_base: str = "",
    provider_id: str = "",
    source_id: str = "",
) -> Literal["gemini", "deepseek_asr", "openai_whisper"]:
    """Determine concrete STT engine from explicit choice or provider/URL clues."""
    eng = (configured_engine or "auto").strip().lower()
    if eng in ("gemini", "deepseek_asr", "openai_whisper"):
        return eng  # type: ignore[return-value]

    ptype = (provider_type or "").strip().lower()
    if "whisper" in ptype:
        return "openai_whisper"

    base_lower = (api_base or "").strip().lower().rstrip("/")
    pid_lower = (provider_id or "").strip().lower()
    sid_lower = (source_id or "").strip().lower()
    path_lower = urlparse(base_lower).path if base_lower else ""

    if (
        "deepseek.com" in base_lower
        or base_lower.endswith("/tools/asr")
        or "/tools/asr" in path_lower
        or "deepseek" in pid_lower
        or "deepseek" in sid_lower
    ):
        return "deepseek_asr"

    return "gemini"


def _infer_gemini_api_format(configured_format: str, provider_type: str, api_base: str) -> str:
    if configured_format in ("gemini_native", "openai_compat"):
        return configured_format
    ptype = (provider_type or "").lower()
    if "googlegenai" in ptype or "gemini" in ptype:
        return "gemini_native"
    if "openai" in ptype:
        return "openai_compat"
    base_lower = (api_base or "").rstrip("/").lower()
    if base_lower.endswith(("/chat/completions", "/v1", "/v1beta/openai")):
        return "openai_compat"
    return "gemini_native"


@dataclass(frozen=True)
class SttEndpointSpec:
    """Unified endpoint specification shared across Primary and Fallback STT slots."""

    slot: Literal["primary", "fallback"]
    engine_kind: Literal["gemini", "deepseek_asr", "openai_whisper"]
    api_base: str
    api_key: str
    model: str = ""
    api_format: str = "gemini_native"
    auth_mode: str = "bearer"
    proxy: str = ""
    verify_ssl: bool = True
    source_desc: str = "standalone"
    text_normalize_endpoint: SttEndpointSpec | None = None

    @property
    def is_pure_stt(self) -> bool:
        """Return True for pure-transcript ASR engines (no rich acoustic template)."""
        return self.engine_kind in ("deepseek_asr", "openai_whisper")

    @property
    def api_mode(self) -> str:
        """Backward-compatible alias for ``api_format``."""
        return self.api_format


@dataclass(frozen=True)
class TranscriptionResult:
    """Final normalized STT result produced by ``HearVoiceEngine``."""

    raw_text: str
    transcript: str
    rich_result: RichSTTResult
    engine_used: str
    model_used: str
    fallback_used: bool
    is_rich: bool
    status: str = "ok"


def _pick_first_key(provider_config: dict[str, Any], provider: Any | None) -> str:
    if provider is not None:
        getter = getattr(provider, "get_current_key", None)
        if callable(getter):
            try:
                key = str(getter() or "").strip()
                if key:
                    return key
            except Exception:
                pass
        chosen = getattr(provider, "chosen_api_key", None)
        if isinstance(chosen, str) and chosen.strip():
            return chosen.strip()
    raw_key = provider_config.get("api_key") or provider_config.get("key")
    if isinstance(raw_key, list):
        for entry in raw_key:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
    elif isinstance(raw_key, str) and raw_key.strip():
        return raw_key.strip()
    return ""


def _extract_provider_base_url(provider_config: dict[str, Any], provider: Any | None) -> str:
    if provider is not None:
        direct = getattr(provider, "api_base", None)
        if direct:
            return str(direct).strip()
    cfg_base = provider_config.get("api_base")
    if cfg_base:
        return str(cfg_base).strip()
    if provider is not None:
        client = getattr(provider, "client", None)
        base_url = getattr(client, "base_url", None)
        if base_url:
            return str(base_url).strip()
    return ""


async def _lookup_provider(context: Any, provider_id: str) -> tuple[Any | None, dict[str, Any]]:
    if not context or not provider_id:
        return None, {}

    provider: Any | None = None
    getter = getattr(context, "get_provider_by_id", None)
    if callable(getter):
        res = getter(provider_id)
        if asyncio.iscoroutine(res):
            provider = await res
        elif res is not None:
            provider = res

    pm = getattr(context, "provider_manager", None)
    if provider is None and pm is not None:
        inst_map = getattr(pm, "inst_map", None)
        if isinstance(inst_map, dict) and provider_id in inst_map:
            provider = inst_map[provider_id]
        else:
            pm_getter = getattr(pm, "get_provider_by_id", None)
            if callable(pm_getter):
                res = pm_getter(provider_id)
                provider = await res if asyncio.iscoroutine(res) else res

    merged_conf: dict[str, Any] = {}
    if pm is not None:
        cfg_getter = getattr(pm, "get_provider_config_by_id", None)
        if callable(cfg_getter):
            try:
                conf = cfg_getter(provider_id, merged=True)
                if isinstance(conf, dict):
                    merged_conf = conf
            except TypeError:
                conf = cfg_getter(provider_id)
                if isinstance(conf, dict):
                    merged_conf = conf
            except Exception:
                pass

    pconf = getattr(provider, "provider_config", None) if provider is not None else None
    combined: dict[str, Any] = {}
    if isinstance(merged_conf, dict):
        combined.update(merged_conf)
    if isinstance(pconf, dict):
        combined.update(pconf)
    return provider, combined


def _build_spec_from_resolved_fields(
    raw: SlotRawConfig,
    cfg: HearVoiceConfig,
    *,
    raw_api_base: str,
    api_key: str,
    provider_model: str,
    provider_type: str,
    source_id: str,
    proxy: str,
    source_desc: str,
    from_provider: bool,
) -> SttEndpointSpec | None:
    if not api_key:
        return None

    engine_kind = resolve_engine_kind(
        raw.engine,
        provider_type=provider_type,
        api_base=raw_api_base,
        provider_id=raw.provider_id if from_provider else "",
        source_id=source_id,
    )

    if engine_kind == "deepseek_asr":
        text_norm_ep: SttEndpointSpec | None = None
        if from_provider and provider_model and raw_api_base:
            text_norm_ep = SttEndpointSpec(
                slot=raw.slot,
                engine_kind="gemini",
                api_base=raw_api_base,
                api_key=api_key,
                model=normalize_model_name(provider_model, enabled=cfg.enable_model_normalize),
                api_format="openai_compat",
                auth_mode="bearer",
                proxy=proxy,
                verify_ssl=cfg.verify_ssl,
                source_desc=f"{source_desc}:text_norm",
            )
        return SttEndpointSpec(
            slot=raw.slot,
            engine_kind="deepseek_asr",
            api_base=normalize_deepseek_asr_url(raw_api_base),
            api_key=api_key,
            model=normalize_model_name(raw.model, enabled=cfg.enable_model_normalize)
            if raw.model
            else "deepseek-asr",
            api_format="multipart",
            auth_mode="bearer",
            proxy=proxy,
            verify_ssl=cfg.verify_ssl,
            source_desc=source_desc,
            text_normalize_endpoint=text_norm_ep,
        )

    if engine_kind == "openai_whisper":
        whisper_model = raw.model or provider_model or DEFAULT_WHISPER_MODEL
        return SttEndpointSpec(
            slot=raw.slot,
            engine_kind="openai_whisper",
            api_base=normalize_openai_whisper_url(raw_api_base),
            api_key=api_key,
            model=normalize_model_name(whisper_model, enabled=cfg.enable_model_normalize),
            api_format="multipart",
            auth_mode="bearer",
            proxy=proxy,
            verify_ssl=cfg.verify_ssl,
            source_desc=source_desc,
        )

    effective_base = raw_api_base
    if not effective_base and from_provider and (
        "googlegenai" in provider_type.lower()
        or "gemini" in provider_type.lower()
        or raw.engine == "gemini"
    ):
        effective_base = "https://generativelanguage.googleapis.com"
    if not effective_base:
        return None

    chosen_model = raw.model or provider_model or DEFAULT_MODEL
    final_model = normalize_model_name(chosen_model, enabled=cfg.enable_model_normalize)
    final_format = _infer_gemini_api_format(raw.api_format, provider_type, effective_base)
    final_auth = (
        "bearer"
        if final_format == "openai_compat" or raw.auth_mode == "bearer"
        else "x_goog_api_key"
    )
    return SttEndpointSpec(
        slot=raw.slot,
        engine_kind="gemini",
        api_base=effective_base,
        api_key=api_key,
        model=final_model,
        api_format=final_format,
        auth_mode=final_auth,
        proxy=proxy,
        verify_ssl=cfg.verify_ssl,
        source_desc=source_desc,
    )


async def resolve_slot_endpoint(
    context: Any,
    cfg: HearVoiceConfig,
    slot: Literal["primary", "fallback"] = "primary",
) -> SttEndpointSpec | None:
    """Resolve Primary or Fallback slot into a concrete ``SttEndpointSpec``."""
    if slot == "fallback" and not cfg.enable_fallback:
        return None

    raw = cfg.get_slot_raw(slot)
    if raw.provider_id:
        provider, pconf = await _lookup_provider(context, raw.provider_id)
        if provider is not None or pconf:
            raw_api_base = raw.api_base or _extract_provider_base_url(pconf, provider)
            api_key = raw.api_key or _pick_first_key(pconf, provider)
            provider_model = ""
            if provider is not None and callable(getattr(provider, "get_model", None)):
                try:
                    provider_model = str(provider.get_model() or "").strip()
                except Exception:
                    provider_model = ""
            if not provider_model:
                provider_model = str(pconf.get("model", "") or "").strip()
            ptype = str(
                pconf.get("type", "") or (type(provider).__name__ if provider is not None else "")
            )
            source_id = str(pconf.get("provider_source_id", "") or pconf.get("provider", "") or "")
            proxy = raw.proxy or str(pconf.get("proxy", "") or "").strip()
            spec = _build_spec_from_resolved_fields(
                raw,
                cfg,
                raw_api_base=raw_api_base,
                api_key=api_key,
                provider_model=provider_model,
                provider_type=ptype,
                source_id=source_id,
                proxy=proxy,
                source_desc=f"provider:{raw.provider_id}",
                from_provider=True,
            )
            if spec is not None:
                return spec

    return _build_spec_from_resolved_fields(
        raw,
        cfg,
        raw_api_base=raw.api_base,
        api_key=raw.api_key,
        provider_model="",
        provider_type="",
        source_id="",
        proxy=raw.proxy,
        source_desc=f"{slot}:standalone",
        from_provider=False,
    )


async def resolve_endpoint(context: Any, cfg: HearVoiceConfig) -> SttEndpointSpec | None:
    """Resolve the primary STT endpoint specification."""
    return await resolve_slot_endpoint(context, cfg, "primary")


async def resolve_endpoints(
    context: Any, cfg: HearVoiceConfig
) -> tuple[SttEndpointSpec | None, SttEndpointSpec | None]:
    """Resolve both primary and fallback STT endpoint specifications."""
    primary = await resolve_slot_endpoint(context, cfg, "primary")
    fallback = await resolve_slot_endpoint(context, cfg, "fallback")
    return primary, fallback


class SttHttpTransport:
    """Shared async HTTP session, retry loop, and credential-redacting error handler."""

    def __init__(self, cfg: HearVoiceConfig) -> None:
        self.cfg = cfg
        self._session: aiohttp.ClientSession | None = None
        self.last_error: str = ""

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout, trust_env=False)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def record_empty_output(self, endpoint: SttEndpointSpec, reason: str) -> None:
        self.last_error = f"HTTP 200 empty output ({reason})"
        logger.warning(
            f"[HearVoice] 上游返回空转写结果 "
            f"(engine={endpoint.engine_kind}, model={endpoint.model}, reason={reason})"
        )

    async def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        endpoint: SttEndpointSpec,
    ) -> dict[str, Any] | None:
        return await self._request_json(
            url, headers=headers, endpoint=endpoint, json_payload=payload
        )

    async def post_multipart(
        self,
        url: str,
        *,
        headers: dict[str, str],
        form_factory: Callable[[], aiohttp.FormData],
        endpoint: SttEndpointSpec,
    ) -> dict[str, Any] | None:
        return await self._request_json(
            url, headers=headers, endpoint=endpoint, form_factory=form_factory
        )

    async def _request_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        endpoint: SttEndpointSpec,
        json_payload: dict[str, Any] | None = None,
        form_factory: Callable[[], aiohttp.FormData] | None = None,
    ) -> dict[str, Any] | None:
        self.last_error = ""
        session = await self._get_session()
        proxy = endpoint.proxy or None
        ssl_opt: Any = None if endpoint.verify_ssl else False
        retries = max(0, self.cfg.retry_times)
        safe_url = redact_url(url)
        secrets = (endpoint.api_key,) if endpoint.api_key else ()

        for attempt in range(retries + 1):
            try:
                post_kwargs: dict[str, Any] = {
                    "headers": headers,
                    "proxy": proxy,
                    "ssl": ssl_opt,
                }
                if json_payload is not None:
                    post_kwargs["json"] = json_payload
                elif form_factory is not None:
                    post_kwargs["data"] = form_factory()

                async with session.post(url, **post_kwargs) as resp:
                    raw = await resp.text()
                    if resp.status == 200:
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            summary = sanitize_error_message(raw, secrets=secrets)
                            self.last_error = f"HTTP 200 invalid JSON: {summary}"
                            logger.warning(
                                f"[HearVoice] 上游返回非 JSON 响应 "
                                f"(engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model}): {summary}"
                            )
                            return None
                        if isinstance(data, dict) and data.get("error"):
                            summary = sanitize_error_message(raw, secrets=secrets)
                            self.last_error = f"HTTP 200 error payload: {summary}"
                            logger.warning(
                                f"[HearVoice] 上游在 HTTP 200 中返回错误载荷 "
                                f"(engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model}): {summary}"
                            )
                            return None
                        if not isinstance(data, dict):
                            self.last_error = f"HTTP 200 unexpected JSON {type(data).__name__}"
                            logger.warning(
                                f"[HearVoice] 上游返回非对象 JSON ({type(data).__name__}, "
                                f"engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model})"
                            )
                            return None
                        return data

                    summary = sanitize_error_message(raw, secrets=secrets)
                    self.last_error = f"HTTP {resp.status}: {summary}"
                    if _is_retryable_status(resp.status):
                        if attempt < retries:
                            logger.warning(
                                f"[HearVoice] 上游返回可重试 HTTP {resp.status} "
                                f"(engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model}, "
                                f"attempt={attempt + 1}/{retries + 1}): {summary}"
                            )
                            delay = min(2**attempt, 8) + random.uniform(0.0, 0.25)
                            await asyncio.sleep(delay)
                            continue
                        logger.warning(
                            f"[HearVoice] 上游返回 HTTP {resp.status} 且重试耗尽 "
                            f"(engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model}): {summary}"
                        )
                        return None

                    logger.warning(
                        f"[HearVoice] 上游返回不可重试 HTTP {resp.status} "
                        f"(engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model}): {summary}"
                    )
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                err_msg = sanitize_error_message(str(exc) or type(exc).__name__, secrets=secrets)
                self.last_error = f"{type(exc).__name__}: {err_msg}"
                if attempt < retries:
                    logger.warning(
                        f"[HearVoice] 上游请求网络/超时异常，准备重试 "
                        f"(engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model}, "
                        f"attempt={attempt + 1}/{retries + 1}): {type(exc).__name__}: {err_msg}"
                    )
                    delay = min(2**attempt, 8) + random.uniform(0.0, 0.25)
                    await asyncio.sleep(delay)
                    continue
                logger.warning(
                    f"[HearVoice] 上游请求失败且重试耗尽 "
                    f"(engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model}): "
                    f"{type(exc).__name__}: {err_msg}"
                )
                return None
            except Exception as exc:
                err_msg = sanitize_error_message(str(exc) or type(exc).__name__, secrets=secrets)
                self.last_error = f"{type(exc).__name__}: {err_msg}"
                logger.warning(
                    f"[HearVoice] 上游请求未预期异常 "
                    f"(engine={endpoint.engine_kind}, url={safe_url}, model={endpoint.model}): "
                    f"{type(exc).__name__}: {err_msg}"
                )
                return None
        return None


class OpenAIWhisperClient:
    """Async OpenAI ``/v1/audio/transcriptions`` multipart STT client."""

    def __init__(
        self,
        cfg: HearVoiceConfig,
        *,
        transport: SttHttpTransport | None = None,
    ) -> None:
        self.cfg = cfg
        self._transport = transport or SttHttpTransport(cfg)

    @property
    def _session(self) -> Any:
        return self._transport._session

    @_session.setter
    def _session(self, value: Any) -> None:
        self._transport._session = value

    @property
    def last_error(self) -> str:
        return self._transport.last_error

    async def close(self) -> None:
        await self._transport.close()

    async def transcribe(
        self,
        audio: ResolvedAudio,
        *,
        endpoint: SttEndpointSpec,
    ) -> str:
        url = normalize_openai_whisper_url(endpoint.api_base)
        headers = {"Authorization": f"Bearer {endpoint.api_key}"}
        raw_bytes = audio.get_bytes()
        ext = audio.audio_format or "wav"
        filename = f"speech.{ext}"
        content_type = audio.mime_type or "audio/wav"
        model_name = endpoint.model or DEFAULT_WHISPER_MODEL

        def _build_form() -> aiohttp.FormData:
            form = aiohttp.FormData()
            form.add_field("file", raw_bytes, filename=filename, content_type=content_type)
            form.add_field("model", model_name)
            return form

        data = await self._transport.post_multipart(
            url, headers=headers, form_factory=_build_form, endpoint=endpoint
        )
        if not data:
            return ""
        text = str(data.get("text") or "").strip()
        if not text:
            self._transport.record_empty_output(endpoint, "empty_text")
        return text


class HearVoiceEngine:
    """Unified Primary + Fallback multi-engine STT orchestrator."""

    def __init__(
        self,
        context: Any,
        cfg: HearVoiceConfig,
        *,
        transport: SttHttpTransport | None = None,
        gemini_client: GeminiClient | None = None,
        deepseek_client: DeepSeekAsrClient | None = None,
        whisper_client: OpenAIWhisperClient | None = None,
    ) -> None:
        self.context = context
        self.cfg = cfg
        self._transport = transport or SttHttpTransport(cfg)
        self.gemini_client = gemini_client or GeminiClient(cfg, transport=self._transport)
        self.deepseek_client = deepseek_client or DeepSeekAsrClient(cfg, transport=self._transport)
        self.whisper_client = whisper_client or OpenAIWhisperClient(cfg, transport=self._transport)

    @property
    def last_error(self) -> str:
        return self._transport.last_error

    @staticmethod
    def safe_endpoint_summary(endpoint: SttEndpointSpec | None) -> str:
        return GeminiClient.safe_endpoint_summary(endpoint)

    async def resolve_endpoints(self) -> tuple[SttEndpointSpec | None, SttEndpointSpec | None]:
        return await resolve_endpoints(self.context, self.cfg)

    async def close(self) -> None:
        await self._transport.close()
        for client in (self.gemini_client, self.deepseek_client, self.whisper_client):
            if getattr(client, "_transport", None) is not self._transport and callable(
                getattr(client, "close", None)
            ):
                await client.close()

    async def _run_endpoint(
        self,
        audio: ResolvedAudio,
        *,
        endpoint: SttEndpointSpec,
        output_mode: str,
        user_text: str,
        custom_instruction: str,
        ffmpeg_path: str,
        is_fallback: bool,
        text_norm_fallback_ep: SttEndpointSpec | None = None,
    ) -> tuple[TranscriptionResult | None, str]:
        effective_mode = "simple" if endpoint.is_pure_stt else output_mode
        instruction = ""

        if endpoint.engine_kind == "deepseek_asr":
            raw_output = await self.deepseek_client.transcribe(
                audio, endpoint=endpoint, ffmpeg_path=ffmpeg_path
            )
        elif endpoint.engine_kind == "openai_whisper":
            raw_output = await self.whisper_client.transcribe(audio, endpoint=endpoint)
        else:
            instruction = build_stt_instruction(
                output_mode,
                custom_instruction=custom_instruction,
                user_text=user_text,
            )
            raw_output = await self.gemini_client.transcribe(
                audio, instruction=instruction, endpoint=endpoint
            )
            if not raw_output and ffmpeg_path:
                is_already_wav = (
                    (audio.audio_format or "").strip().lower() == "wav"
                    or (audio.mime_type or "").split(";", 1)[0].strip().lower()
                    in ("audio/wav", "audio/x-wav", "audio/wave")
                )
                err_detail = str(
                    getattr(self.gemini_client, "last_error", "")
                    or getattr(self._transport, "last_error", "")
                    or ""
                )
                if not is_already_wav and is_unsupported_audio_format_400(err_detail):
                    logger.warning(
                        "[HearVoice] 主通道拒绝当前音频格式 "
                        f"({audio.mime_type or audio.audio_format or 'unknown'})，"
                        "尝试转码为 16kHz 单声道 WAV 重试一次"
                    )
                    wav_audio = await transcode_audio_to_wav_resolved(
                        audio, ffmpeg_path=ffmpeg_path
                    )
                    if wav_audio is not None:
                        raw_output = await self.gemini_client.transcribe(
                            wav_audio, instruction=instruction, endpoint=endpoint
                        )

        raw_output = clean_transcript(
            raw_output,
            enabled=self.cfg.enable_transcript_clean,
            max_chars=max(self.cfg.max_transcript_chars * 2, 4000),
        )
        if not raw_output:
            detail = str(
                getattr(self._transport, "last_error", "")
                or getattr(self.gemini_client, "last_error", "")
                or getattr(self.deepseek_client, "last_error", "")
                or ""
            ).strip()
            return None, f"empty_response:{detail}" if detail else "empty_response"

        if is_provider_error_text(raw_output):
            logger.warning(
                f"[HearVoice] 上游返回错误文本，已拒收并按失败处理 "
                f"(engine={endpoint.engine_kind}, model={endpoint.model})"
            )
            return None, "provider_error_text"

        if not endpoint.is_pure_stt and is_instruction_hallucination(raw_output, instruction):
            logger.warning("[HearVoice] 检测到提示词复读幻觉，使用静音兜底文本")
            fallback_msg = "（未检测到有效语音内容，可能为空白或静音语音）"
            return (
                TranscriptionResult(
                    raw_text=fallback_msg,
                    transcript=fallback_msg,
                    rich_result=RichSTTResult(
                        raw_text=fallback_msg, transcript=fallback_msg, is_silent=True
                    ),
                    engine_used=endpoint.engine_kind,
                    model_used=endpoint.model,
                    fallback_used=is_fallback,
                    is_rich=False,
                ),
                "ok",
            )

        parsed = parse_rich_output(raw_output, output_mode=effective_mode)
        transcript = clean_transcript(
            parsed.transcript,
            enabled=self.cfg.enable_transcript_clean,
            max_chars=self.cfg.max_transcript_chars,
        )
        if not transcript:
            return None, "empty_transcript"

        if self.cfg.enable_text_normalize and not parsed.is_silent:
            norm_ep: SttEndpointSpec | None = None
            if endpoint.engine_kind == "gemini":
                norm_ep = endpoint
            elif endpoint.text_normalize_endpoint is not None:
                norm_ep = endpoint.text_normalize_endpoint
            elif text_norm_fallback_ep is not None and text_norm_fallback_ep.engine_kind == "gemini":
                norm_ep = text_norm_fallback_ep

            if norm_ep is not None:
                normalized = await self.gemini_client.normalize_transcript_with_model(
                    transcript, endpoint=norm_ep
                )
                normalized = clean_transcript(
                    normalized,
                    enabled=self.cfg.enable_transcript_clean,
                    max_chars=self.cfg.max_transcript_chars,
                )
                if normalized and not is_provider_error_text(normalized):
                    transcript = normalized

        is_rich = effective_mode == "rich" and not endpoint.is_pure_stt
        final_raw = parsed.raw_text if is_rich else transcript
        final_rich = RichSTTResult(
            raw_text=final_raw,
            transcript=transcript,
            emotion=parsed.emotion if is_rich else "",
            speaker=parsed.speaker if is_rich else "",
            speed=parsed.speed if is_rich else "",
            ambient=parsed.ambient if is_rich else "",
            summary=parsed.summary if is_rich else "",
            is_silent=parsed.is_silent,
        )
        return (
            TranscriptionResult(
                raw_text=final_raw,
                transcript=transcript,
                rich_result=final_rich,
                engine_used=endpoint.engine_kind,
                model_used=endpoint.model,
                fallback_used=is_fallback,
                is_rich=is_rich,
            ),
            "ok",
        )

    async def transcribe_audio(
        self,
        audio: ResolvedAudio,
        *,
        output_mode: str | None = None,
        user_text: str = "",
        custom_instruction: str = "",
        ffmpeg_path: str = "",
        primary: SttEndpointSpec | None = None,
        fallback: SttEndpointSpec | None = None,
    ) -> tuple[TranscriptionResult | None, str]:
        """Transcribe audio using Primary endpoint and fail over seamlessly to Fallback."""
        if primary is None and fallback is None:
            primary, fallback = await self.resolve_endpoints()

        mode = output_mode or self.cfg.output_mode
        if primary is None and fallback is None:
            return None, "no_endpoint"

        if primary is None:
            assert fallback is not None
            logger.warning(
                f"[HearVoice] 主识别通道未配置，直接使用回退通道 ({self.safe_endpoint_summary(fallback)})"
            )
            return await self._run_endpoint(
                audio,
                endpoint=fallback,
                output_mode=mode,
                user_text=user_text,
                custom_instruction=custom_instruction,
                ffmpeg_path=ffmpeg_path,
                is_fallback=True,
            )

        res, status = await self._run_endpoint(
            audio,
            endpoint=primary,
            output_mode=mode,
            user_text=user_text,
            custom_instruction=custom_instruction,
            ffmpeg_path=ffmpeg_path,
            is_fallback=False,
            text_norm_fallback_ep=fallback,
        )
        if res is not None:
            return res, "ok"

        if fallback is not None:
            safe_err = sanitize_error_message(
                status,
                secrets=tuple(k for k in (primary.api_key, fallback.api_key) if k),
            )
            logger.warning(
                f"[HearVoice] 主识别失败 ({self.safe_endpoint_summary(primary)}, reason={safe_err})，"
                f"自动切换回退通道 ({self.safe_endpoint_summary(fallback)})"
            )
            fb_res, fb_status = await self._run_endpoint(
                audio,
                endpoint=fallback,
                output_mode=mode,
                user_text=user_text,
                custom_instruction=custom_instruction,
                ffmpeg_path=ffmpeg_path,
                is_fallback=True,
            )
            if fb_res is not None:
                return fb_res, "ok"
            return None, fb_status

        return None, status
