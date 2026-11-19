from __future__ import annotations

from typing import Any


class STTProvider:
    def __init__(self, provider_config: dict[str, Any], provider_settings: dict[str, Any]) -> None:
        self.model_name = ""
        self.provider_config = provider_config
        self.provider_settings = provider_settings

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name

    def get_model(self) -> str:
        return self.model_name

    async def get_text(self, audio_url: str) -> str:
        raise NotImplementedError
