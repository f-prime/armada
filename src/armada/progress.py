"""Terminal progress for acknowledged model transfers."""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

from .status import format_size


class Progress:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.width = self.total = self.done = 0
        self.started = self.updated = 0.0

    def finish_line(self) -> None:
        if self.width:
            print(file=self.stream, flush=True)
            self.width = 0

    def phase(self, message: str) -> None:
        self.finish_line()
        print(message, file=self.stream, flush=True)

    def start(self, label: str, total: int) -> None:
        self.phase(label)
        self.total, self.done = total, 0
        self.started = time.monotonic()
        self.updated = self.started - 1
        self.advance(0)

    def advance(self, size: int) -> None:
        self.done += size
        now = time.monotonic()
        if now - self.updated < 0.2 and self.done < self.total:
            return
        self.updated = now
        pct = self.done * 100 // self.total if self.total else 100
        line = f"  {pct}% · {format_size(self.done)} / {format_size(self.total)}"
        elapsed = now - self.started
        if self.done and elapsed > 0:
            rate = self.done / elapsed
            seconds = round((self.total - self.done) / rate)
            minutes, seconds = divmod(seconds, 60)
            line += f" · {format_size(round(rate))}/s · ETA {minutes}m {seconds:02d}s"
        print("\r" + line.ljust(self.width), end="", file=self.stream, flush=True)
        self.width = len(line)


@contextmanager
def terminal_progress() -> Iterator[Progress | None]:
    progress = Progress(sys.stderr) if sys.stderr.isatty() else None
    try:
        yield progress
    finally:
        if progress:
            progress.finish_line()
