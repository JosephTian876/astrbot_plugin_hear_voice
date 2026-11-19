from __future__ import annotations

from typing import Any


class Context:
    def __init__(self) -> None:
        self.provider_manager: Any = None


class Star:
    def __init__(self, context: Any = None) -> None:
        self.context = context
