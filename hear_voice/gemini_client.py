"""Gemini native and OpenAI-compatible multimodal audio STT client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from .audio_resolver import ResolvedAudio
from .config import HearVoiceConfig, redact_url, sanitize_error_message

if TYPE_CHECKING:
    from .engine import SttEndpointSpec, SttHttpTransport

__all__ = [
    "GeminiClient",
    "build_gemini_generate_url",
    "build_openai_chat_url",
    "extract_empty_output_reason",
    "extract_gemini_parts_text",
    "extract_openai_message_text",
    "is_unsupported_audio_format_400",
    "normalize_gemini_audio_mime",
    "sanitize_error_message",
]


def normalize_gemini_audio_mime(mime_type: str, audio_format: str = "") -> str:
    """Normalize audio MIME type for Gemini Native ``inline_data`` payloads.

    Some Gemini relays/gateways reject ``audio/mpeg`` with HTTP 400
    ``The attachment format is not supported`` while accepting ``audio/mp3``.
    """
    raw = (mime_type or "").split(";", 1)[0].strip().lower()
    if raw in ("audio/mpeg", "audio/x-mpeg", "audio/mp3", "audio/x-mp3"):
        return "audio/mp3"
    if raw:
        return raw
    fmt = (audio_format or "").strip().lower().lstrip(".")
    if fmt == "mp3":
        return "audio/mp3"
    if fmt == "wav":
        return "audio/wav"
    return f"audio/{fmt}" if fmt else "audio/wav"


def is_unsupported_audio_format_400(error_text: str) -> bool:
    """Return True when a Gemini HTTP 400 error reports unsupported audio/attachment format."""
    low = (error_text or "").lower()
    if "http 400" not in low:
        return False
    has_format_hint = any(
        kw in low
        for kw in (
            "attachment format",
            "audio format",
            "mime type",
            "mimetype",
            "mime_type",
            "media type",
            "inline_data",
            "unable to decode",
            "invalid audio",
            "unsupported audio",
        )
    )
    has_reject_hint = any(
        kw in low
        for kw in (
            "not supported",
            "unsupported",
            "invalid",
            "unable to decode",
        )
    )
    return has_format_hint and has_reject_hint


def build_gemini_generate_url(api_base: str, model: str) -> str:
    """Build Gemini generateContent URL cleanly without double path suffixes."""
    base = (api_base or "").rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1beta/openai", "/v1beta", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    model_id = model.strip()
    if model_id.startswith("models/"):
        model_id = model_id[len("models/") :]
    encoded_model = quote(model_id, safe=".-_:")
    return f"{base}/v1beta/models/{encoded_model}:generateContent"


def build_openai_chat_url(api_base: str) -> str:
    """Build OpenAI chat/completions URL from a base or full endpoint."""
    base = (api_base or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith(("/v1", "/v1beta/openai")):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def extract_gemini_parts_text(data: dict[str, Any]) -> str:
    """Extract non-thought text parts from a Gemini generateContent JSON payload."""
    if not isinstance(data, dict):
        return ""
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""

    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        # Critical: skip thinking/reasoning blocks from thinking models (e.g. Gemini 2.5/3.x).
        if bool(part.get("thought")):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts).strip()


def extract_openai_message_text(data: dict[str, Any]) -> str:
    """Extract assistant text from OpenAI chat/completions JSON, ignoring reasoning blocks."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "").lower()
            if item_type in ("thinking", "thought", "reasoning"):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        return "\n".join(texts).strip()
    return ""


def extract_empty_output_reason(data: dict[str, Any], api_format: str) -> str:
    """Summarize why an HTTP 200 payload yielded no non-thought text."""
    if api_format == "openai_compat":
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            reason = str(choices[0].get("finish_reason") or "").strip()
            return reason or "empty_message"
        return "no_choices"
    pf = data.get("promptFeedback")
    if isinstance(pf, dict) and pf.get("blockReason"):
        return f"block:{pf['blockReason']}"
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        reason = str(candidates[0].get("finishReason") or "").strip()
        return reason or "empty_parts"
    return "no_candidates"


class GeminiClient:
    """Async protocol adapter for Gemini native and OpenAI-compatible audio transcription."""

    def __init__(
        self,
        cfg: HearVoiceConfig,
        *,
        transport: SttHttpTransport | None = None,
    ) -> None:
        self.cfg = cfg
        if transport is None:
            from .engine import SttHttpTransport

            transport = SttHttpTransport(cfg)
        self._transport = transport

    @property
    def _session(self) -> Any:
        return self._transport._session

    @_session.setter
    def _session(self, value: Any) -> None:
        self._transport._session = value

    @property
    def last_error(self) -> str:
        return self._transport.last_error

    @last_error.setter
    def last_error(self, value: str) -> None:
        self._transport.last_error = value

    async def close(self) -> None:
        await self._transport.close()

    def _build_headers(self, endpoint: SttEndpointSpec) -> dict[str, str]:
        if endpoint.api_format == "openai_compat" or endpoint.auth_mode == "bearer":
            return {
                "Authorization": f"Bearer {endpoint.api_key}",
                "Content-Type": "application/json",
            }
        return {
            "x-goog-api-key": endpoint.api_key,
            "Content-Type": "application/json",
        }

    async def transcribe(
        self,
        audio: ResolvedAudio,
        *,
        instruction: str,
        endpoint: SttEndpointSpec,
    ) -> str:
        """Transcribe resolved audio via Gemini native or OpenAI-compat audio input."""
        headers = self._build_headers(endpoint)
        if endpoint.api_format == "openai_compat":
            url = build_openai_chat_url(endpoint.api_base)
            payload = {
                "model": endpoint.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio.base64_data,
                                    "format": audio.audio_format,
                                },
                            },
                            {"type": "text", "text": instruction},
                        ],
                    }
                ],
            }
            data = await self._transport.post_json(
                url, headers=headers, payload=payload, endpoint=endpoint
            )
            if not data:
                return ""
            text = extract_openai_message_text(data)
        else:
            url = build_gemini_generate_url(endpoint.api_base, endpoint.model)
            mime_type = normalize_gemini_audio_mime(audio.mime_type, audio.audio_format)
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": audio.base64_data,
                                }
                            },
                            {"text": instruction},
                        ],
                    }
                ]
            }
            data = await self._transport.post_json(
                url, headers=headers, payload=payload, endpoint=endpoint
            )
            if not data:
                return ""
            text = extract_gemini_parts_text(data)

        if not text:
            reason = extract_empty_output_reason(data, endpoint.api_format)
            self._transport.record_empty_output(endpoint, reason)
        return text

    async def normalize_transcript_with_model(
        self,
        transcript: str,
        *,
        endpoint: SttEndpointSpec,
    ) -> str:
        """Optional secondary text-only normalization pass preserving original meaning."""
        text = (transcript or "").strip()
        if not text:
            return ""

        prompt = (
            "请对以下语音转写文本做轻量规范化（仅整理标点、断句与明显同音错别字），"
            "严格保持说话人原本的口吻、用词、语气词和完整含义，不要增删信息，不要回答内容，直接输出整理后的纯文本：\n\n"
            f"{text}"
        )
        headers = self._build_headers(endpoint)

        if endpoint.api_format == "openai_compat":
            url = build_openai_chat_url(endpoint.api_base)
            payload = {
                "model": endpoint.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            data = await self._transport.post_json(
                url, headers=headers, payload=payload, endpoint=endpoint
            )
            normalized = extract_openai_message_text(data) if data else ""
        else:
            url = build_gemini_generate_url(endpoint.api_base, endpoint.model)
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            data = await self._transport.post_json(
                url, headers=headers, payload=payload, endpoint=endpoint
            )
            normalized = extract_gemini_parts_text(data) if data else ""

        return normalized.strip() or text

    @staticmethod
    def safe_endpoint_summary(endpoint: SttEndpointSpec | None) -> str:
        if endpoint is None:
            return "unconfigured"
        return (
            f"slot={endpoint.slot} engine={endpoint.engine_kind} "
            f"source={endpoint.source_desc} format={endpoint.api_format} "
            f"auth={endpoint.auth_mode} model={endpoint.model} base={redact_url(endpoint.api_base)}"
        )
