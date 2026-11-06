"""Tiny name → callable registry. Used to register metrics by name."""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class _Registry:
    def __init__(self) -> None:
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._items:
                logger.warning("metric %r already registered; overwriting", name)
            self._items[name] = fn
            return fn
        return _deco

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(f"metric not registered: {name!r} (have: {sorted(self._items)})")
        return self._items[name]

    def list(self) -> list[str]:
        return sorted(self._items)


_GLOBAL = _Registry()
register_metric = _GLOBAL.register


def get_metric(name: str) -> Callable[..., Any]:
    return _GLOBAL.get(name)


def list_metrics() -> list[str]:
    return _GLOBAL.list()
