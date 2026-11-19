"""AstrBot plugin entry point for 我听得见语音 (astrbot_plugin_hear_voice)."""

from __future__ import annotations

from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

if __package__:
    from .hear_voice.config import HearVoiceConfig
    from .hear_voice.engine import HearVoiceEngine
    from .hear_voice.hook_pipeline import HearVoicePipeline
    from .hear_voice.stt_provider import (
        mount_runtime_stt_provider,
        unmount_runtime_stt_provider,
    )
else:  # pragma: no cover - direct root import
    from hear_voice.config import HearVoiceConfig
    from hear_voice.engine import HearVoiceEngine
    from hear_voice.hook_pipeline import HearVoicePipeline
    from hear_voice.stt_provider import (
        mount_runtime_stt_provider,
        unmount_runtime_stt_provider,
    )

HOOK_PRIORITY = 300000


class HearVoicePlugin(Star):
    """Non-hijacking two-stage multi-engine voice understanding plugin (我听得见语音)."""

    def __init__(self, context: Context, config: AstrBotConfig | dict[str, Any] | None = None):
        super().__init__(context)
        raw_config = dict(config or {}) if hasattr(config, "items") else {}
        resolved, notices = HearVoiceConfig.from_mapping_detailed(raw_config)
        self.cfg = resolved
        self.config_notices = notices
        for notice in notices:
            logger.warning(f"[HearVoice] 配置提示: {notice}")

        self.engine = HearVoiceEngine(context, self.cfg)
        self.pipeline = HearVoicePipeline(context, self.cfg, engine=self.engine)
        self._mounted_stt_provider = mount_runtime_stt_provider(
            context, self.cfg, engine=self.engine
        )

        logger.info(
            "[HearVoice] 插件已加载: "
            f"enable={self.cfg.enable_voice}, mode={self.cfg.output_mode}, "
            f"primary_engine={self.cfg.provider_engine}, "
            f"provider_id={self.cfg.provider_id or '<standalone>'}, "
            f"fallback={self.cfg.enable_fallback}/{self.cfg.fallback_engine}, "
            f"fallback_provider={self.cfg.fallback_provider_id or '<standalone>'}, "
            f"private={self.cfg.enable_private}, group={self.cfg.enable_group_voice}, "
            f"fail={self.cfg.on_stt_fail}, concurrency={self.cfg.stt_max_concurrency}, "
            f"ffmpeg={'✓' if self.pipeline.ffmpeg_path else '✗'}, "
            f"stt_adapter={'✓' if self._mounted_stt_provider else '✗'}"
        )

    @filter.event_message_type(filter.EventMessageType.ALL, priority=HOOK_PRIORITY)
    async def on_voice_message(self, event: AstrMessageEvent):
        """Stage 1: Pre-transcribe direct/quoted voice before companion & memory hooks."""
        async for result in self.pipeline.process_message_event(event):
            yield result

    @filter.on_llm_request(priority=HOOK_PRIORITY)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Stage 2: Inject temporary rich acoustic context and clean redundant audio attachments."""
        await self.pipeline.process_llm_request(event, req)

    async def terminate(self) -> None:
        """Release HTTP transport, background cleanup loop, and optional STT provider mount."""
        try:
            await unmount_runtime_stt_provider(
                self.context,
                self.cfg.stt_provider_id,
                self._mounted_stt_provider,
            )
            self._mounted_stt_provider = None
        except Exception as exc:
            logger.warning(f"[HearVoice] 卸载 STT Provider 失败: {type(exc).__name__}")
        await self.pipeline.shutdown()
