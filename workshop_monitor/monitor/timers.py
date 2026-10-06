"""Антидребезг: состояние включается после on_after секунд подряд и выключается после off_after."""
from __future__ import annotations

from typing import Hashable, Iterable, Iterator


class DebouncedState:
    def __init__(self, on_after: float, off_after: float):
        self.on_after = on_after
        self.off_after = off_after
        self.on = False
        self.start: float | None = None
        self._cand: float | None = None

    def update(self, active: bool, t: float) -> tuple[float, float] | None:
        """Возвращает закрытый интервал (start, end), когда состояние выключилось."""
        if not self.on:
            if active:
                if self._cand is None:
                    self._cand = t
                if t - self._cand >= self.on_after:
                    self.on, self.start, self._cand = True, self._cand, None
            else:
                self._cand = None
            return None
        if active:
            self._cand = None
            return None
        if self._cand is None:
            self._cand = t
        if t - self._cand >= self.off_after:
            closed = (self.start, self._cand)
            self.on, self.start, self._cand = False, None, None
            return closed
        return None

    def flush(self, t: float) -> tuple[float, float] | None:
        if self.on:
            closed = (self.start, t)
            self.on, self.start, self._cand = False, None, None
            return closed
        return None


class TimerBank:
    """Набор DebouncedState по ключам (track_id, workplace_id, пара треков ...)."""

    def __init__(self, on_after: float, off_after: float):
        self.on_after, self.off_after = on_after, off_after
        self._timers: dict[Hashable, DebouncedState] = {}

    def update(self, t: float, active_keys: Iterable[Hashable]) -> Iterator[tuple[Hashable, float, float]]:
        active = set(active_keys)
        for k in active:
            self._timers.setdefault(k, DebouncedState(self.on_after, self.off_after))
        for k in list(self._timers):
            st = self._timers[k]
            closed = st.update(k in active, t)
            if closed:
                yield (k, *closed)
            if not st.on and st._cand is None and k not in active:
                del self._timers[k]

    def is_on(self, key: Hashable) -> bool:
        st = self._timers.get(key)
        return bool(st and st.on)

    def flush(self, t: float) -> Iterator[tuple[Hashable, float, float]]:
        for k, st in list(self._timers.items()):
            closed = st.flush(t)
            if closed:
                yield (k, *closed)
        self._timers.clear()
