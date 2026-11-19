from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class _ComponentType:
    def __init__(self, value: str) -> None:
        self.value = value


@dataclass
class Plain:
    text: str
    type: Any = field(default_factory=lambda: _ComponentType("Plain"))


@dataclass
class Record:
    file: str | None = ""
    url: str | None = ""
    path: str | None = None
    text: str | None = None
    data: dict[str, Any] | None = None
    type: Any = field(default_factory=lambda: _ComponentType("Record"))


@dataclass
class Reply:
    id: str = ""
    chain: list[Any] = field(default_factory=list)
    message_str: str = ""
    sender_nickname: str = ""
    type: Any = field(default_factory=lambda: _ComponentType("Reply"))


@dataclass
class Image:
    file: str = ""
    url: str = ""
    path: str | None = None
    type: Any = field(default_factory=lambda: _ComponentType("Image"))


@dataclass
class At:
    qq: str = ""
    type: Any = field(default_factory=lambda: _ComponentType("At"))
