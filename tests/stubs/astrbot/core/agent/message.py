from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextPart:
    text: str
    type: str = "text"
    _no_save: bool = field(default=False, repr=False)

    def mark_as_temp(self) -> "TextPart":
        self._no_save = True
        return self

    def model_dump_for_context(self) -> dict:
        data = {"type": self.type, "text": self.text}
        if self._no_save:
            data["_no_save"] = True
        return data
