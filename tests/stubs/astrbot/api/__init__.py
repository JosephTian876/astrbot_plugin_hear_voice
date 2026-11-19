from __future__ import annotations

import logging


class _Logger:
    def __init__(self) -> None:
        self._log = logging.getLogger("astrbot_stub")

    def debug(self, msg, *args, **kwargs):
        self._log.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log.error(msg, *args, **kwargs)


AstrBotConfig = dict
logger = _Logger()
