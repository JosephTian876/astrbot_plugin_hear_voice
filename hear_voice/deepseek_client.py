"""Official DeepSeek ASR (/v1/tools/asr) multipart audio transcription client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp

from .audio_resolver import ResolvedAudio, prepare_audio_for_deepseek
from .config import DEFAULT_DEEPSEEK_ASR_URL, HearVoiceConfig

if TYPE_CHECKING:
    from .engine import SttEndpointSpec, SttHttpTransport

__all__ = ["DeepSeekAsrClient", "normalize_deepseek_asr_url"]


def normalize_deepseek_asr_url(api_base: str | None) -> str:
    """Normalize DeepSeek base URL or custom proxy path to the ``/v1/tools/asr`` endpoint."""
    raw = (api_base or "").strip().rstrip("/")
    if not raw:
        return DEFAULT_DEEPSEEK_ASR_URL

    parsed = urlparse(raw)
    if "/tools/asr" in parsed.path.lower():
        return raw

    lower = raw.lower()
    if lower.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")].rstrip("/")
        lower = raw.lower()

    if lower.endswith("/v1"):
        return f"{raw}/tools/asr"
    return f"{raw}/v1/tools/asr"


class DeepSeekAsrClient:
    """Async client for DeepSeek official ``POST /v1/tools/asr`` multipart audio recognition."""

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

    async def transcribe(
        self,
        audio: ResolvedAudio,
        *,
        endpoint: SttEndpointSpec,
        ffmpeg_path: str = "",
    ) -> str:
        """Adapt audio to WAV/Ogg Opus <= 8 MiB & 120s and POST to DeepSeek ASR."""
        prepared, prep_err = await prepare_audio_for_deepseek(
            audio,
            ffmpeg_path=ffmpeg_path,
            timeout_sec=self.cfg.timeout_sec,
        )
        if prepared is None:
            self._transport.last_error = f"audio_prepare: {prep_err or 'failed'}"
            return ""

        url = normalize_deepseek_asr_url(endpoint.api_base)
        headers = {"Authorization": f"Bearer {endpoint.api_key}"}

        def _build_form() -> aiohttp.FormData:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                prepared.data,
                filename=prepared.filename,
                content_type=prepared.content_type,
            )
            return form

        data = await self._transport.post_multipart(
            url,
            headers=headers,
            form_factory=_build_form,
            endpoint=endpoint,
        )
        if not data:
            return ""

        text = str(data.get("text") or "").strip()
        if not text:
            self._transport.record_empty_output(endpoint, "empty_text")
        return text
