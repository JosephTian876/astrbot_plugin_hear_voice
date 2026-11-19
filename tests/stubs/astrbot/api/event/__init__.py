from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from astrbot.api.message_components import Plain


@dataclass
class MessageChain:
    chain: list[Any] = field(default_factory=list)

    def message(self, text: str) -> "MessageChain":
        self.chain.append(Plain(text))
        return self


class _EventMessageType(enum.Enum):
    ALL = "ALL"
    GROUP_MESSAGE = "GROUP_MESSAGE"
    PRIVATE_MESSAGE = "PRIVATE_MESSAGE"


def _record_decorator(kind: str):
    def factory(*args: Any, **kwargs: Any):
        def wrapper(fn):
            meta = getattr(fn, "_astrbot_hooks", [])
            meta.append((kind, args, kwargs))
            fn._astrbot_hooks = meta
            return fn

        return wrapper

    return factory


class _Filter:
    EventMessageType = _EventMessageType
    event_message_type = staticmethod(_record_decorator("event_message_type"))
    on_llm_request = staticmethod(_record_decorator("on_llm_request"))


filter = _Filter()


class AstrMessageEvent:  # pragma: no cover - typing/protocol base
    pass
