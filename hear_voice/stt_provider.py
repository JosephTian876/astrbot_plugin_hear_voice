"""Optional AstrBot ProviderType.SPEECH_TO_TEXT adapter delegating to HearVoiceEngine."""

from __future__ import annotations

from typing import Any

from .audio_resolver import find_ffmpeg, resolve_audio_ref
from .config import HearVoiceConfig
from .engine import HearVoiceEngine

ADAPTER_TYPE_NAME = "hear_voice_stt"
ADAPTER_DISPLAY_NAME = "HearVoice Multi-Engine STT"

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger("astrbot_plugin_hear_voice")

try:
    from astrbot.core.provider.entities import ProviderType
    from astrbot.core.provider.provider import STTProvider
    from astrbot.core.provider.register import (
        provider_cls_map,
        provider_registry,
        register_provider_adapter,
    )
except ImportError:  # pragma: no cover
    ProviderType = None  # type: ignore[assignment]
    STTProvider = object  # type: ignore[assignment,misc]
    provider_cls_map = {}  # type: ignore[assignment]
    provider_registry = []  # type: ignore[assignment]
    register_provider_adapter = None  # type: ignore[assignment]


class HearVoiceSTTProvider(STTProvider):  # type: ignore[misc]
    """AstrBot STTProvider adapter delegating through the unified Primary+Fallback engine."""

    def __init__(
        self,
        provider_config: dict[str, Any],
        provider_settings: dict[str, Any] | None = None,
        *,
        plugin_config: HearVoiceConfig | None = None,
        plugin_context: Any = None,
        engine: HearVoiceEngine | None = None,
    ) -> None:
        if STTProvider is not object:
            super().__init__(provider_config, provider_settings or {})
        else:
            self.provider_config = provider_config
            self.provider_settings = provider_settings or {}
            self.model_name = ""

        self._plugin_context = plugin_context
        self._cfg = plugin_config or HearVoiceConfig.from_mapping(provider_config)
        self._engine = engine or HearVoiceEngine(plugin_context, self._cfg)
        self._owns_engine = engine is None
        self._ffmpeg_path = find_ffmpeg(self._cfg.ffmpeg_path)

        model = str(provider_config.get("model") or self._cfg.model or "gemini-2.5-flash").strip()
        if hasattr(self, "set_model"):
            self.set_model(model)
        else:
            self.model_name = model

    async def get_text(self, audio_url: str) -> str:
        """Transcribe an audio reference into plain text using Primary+Fallback engine."""
        resolved = await resolve_audio_ref(
            audio_url, self._cfg, ffmpeg_path=self._ffmpeg_path
        )
        if resolved is None:
            raise ValueError("HearVoiceSTTProvider could not resolve audio input")

        result, status = await self._engine.transcribe_audio(
            resolved,
            output_mode="simple",
            custom_instruction=(
                "请精准转写这段音频中的原话，直接输出纯文本，不要添加标题、编号、解释或 Markdown。"
            ),
            ffmpeg_path=self._ffmpeg_path,
        )
        if result is None:
            detail = str(getattr(self._engine, "last_error", "") or status or "").strip()
            msg = "HearVoiceSTTProvider transcription failed"
            if detail:
                msg = f"{msg}: {detail}"
            raise RuntimeError(msg)

        return result.transcript

    async def terminate(self) -> None:
        if self._owns_engine:
            await self._engine.close()


def ensure_hear_voice_stt_adapter_registered() -> bool:
    """Register the ``hear_voice_stt`` adapter idempotently across plugin hot reloads."""
    if ProviderType is None or register_provider_adapter is None:
        return False

    if ADAPTER_TYPE_NAME in provider_cls_map:
        meta = provider_cls_map[ADAPTER_TYPE_NAME]
        meta.cls_type = HearVoiceSTTProvider
        return True

    register_provider_adapter(
        ADAPTER_TYPE_NAME,
        "HearVoice 多引擎语音转写适配器（Gemini + DeepSeek ASR 主/备）",
        provider_type=ProviderType.SPEECH_TO_TEXT,
        default_config_tmpl={
            "id": ADAPTER_TYPE_NAME,
            "type": ADAPTER_TYPE_NAME,
            "enable": False,
            "api_base": "",
            "api_key": "",
            "model": "gemini-2.5-flash",
        },
        provider_display_name=ADAPTER_DISPLAY_NAME,
    )(HearVoiceSTTProvider)
    return True


def mount_runtime_stt_provider(
    context: Any,
    cfg: HearVoiceConfig,
    *,
    engine: HearVoiceEngine | None = None,
) -> HearVoiceSTTProvider | None:
    """Optionally attach a runtime HearVoiceSTTProvider instance to ProviderManager."""
    if not cfg.register_stt_provider:
        return None
    ensure_hear_voice_stt_adapter_registered()

    pm = getattr(context, "provider_manager", None)
    if pm is None:
        return None

    provider_id = cfg.stt_provider_id or ADAPTER_TYPE_NAME
    inst_map = getattr(pm, "inst_map", None)
    stt_list = getattr(pm, "stt_provider_insts", None)

    if isinstance(inst_map, dict) and provider_id in inst_map:
        existing = inst_map[provider_id]
        if not isinstance(existing, HearVoiceSTTProvider) and not getattr(
            existing, "_is_hear_voice_stt_runtime_provider", False
        ):
            logger.warning(
                f"[HearVoice] provider_id={provider_id} 已被其他提供商占用，跳过挂载运行时 STT Provider"
            )
            return None

    pconf = {
        "id": provider_id,
        "type": ADAPTER_TYPE_NAME,
        "model": cfg.model or "gemini-2.5-flash",
        "enable": True,
    }
    inst = HearVoiceSTTProvider(
        pconf,
        getattr(pm, "provider_settings", {}) or {},
        plugin_config=cfg,
        plugin_context=context,
        engine=engine,
    )
    inst._is_hear_voice_stt_runtime_provider = True  # type: ignore[attr-defined]

    if isinstance(inst_map, dict):
        old = inst_map.get(provider_id)
        if old is not None and isinstance(stt_list, list) and old in stt_list:
            stt_list.remove(old)
        inst_map[provider_id] = inst
    if isinstance(stt_list, list) and inst not in stt_list:
        stt_list.append(inst)

    return inst


async def unmount_runtime_stt_provider(
    context: Any,
    provider_id: str,
    instance: HearVoiceSTTProvider | None = None,
) -> None:
    """Cleanly remove a plugin-mounted HearVoiceSTTProvider instance on terminate/reload."""
    pm = getattr(context, "provider_manager", None)
    if pm is None:
        return

    inst_map = getattr(pm, "inst_map", None)
    stt_list = getattr(pm, "stt_provider_insts", None)
    target = instance

    if isinstance(inst_map, dict):
        mapped = inst_map.get(provider_id)
        if mapped is not None and (target is None or mapped is target):
            if getattr(mapped, "_is_hear_voice_stt_runtime_provider", False):
                del inst_map[provider_id]
                target = mapped

    if isinstance(stt_list, list) and target is not None and target in stt_list:
        stt_list.remove(target)

    if getattr(pm, "curr_stt_provider_inst", None) is target and target is not None:
        pm.curr_stt_provider_inst = stt_list[0] if isinstance(stt_list, list) and stt_list else None

    if target is not None:
        await target.terminate()
