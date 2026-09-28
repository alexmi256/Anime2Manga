"""Planned feature - subtitle translation.

Not implemented yet, but the seams are in place so it can be added without
reshaping the pipeline:

* :class:`Translator` is a :class:`typing.Protocol` describing the conversion of
  a list of :class:`~anime2manga.models.SubtitleLine` from one language to
  another.
* The CLI reserves ``--translate-to LANG`` (and ``--translate-from``).  Passing
  it currently raises :class:`~anime2manga.errors.TranslationNotImplementedError`
  after the source track has been selected and extracted, so the error message
  can name the concrete languages involved.

The intended insertion point is right after step 2 (subtitle extraction) and
before step 3, so translation benefits from the same cue timings and the rest of
the pipeline sees translated text.
"""

from __future__ import annotations

from typing import Protocol

from .errors import TranslationNotImplementedError
from .models import SubtitleLine


class Translator(Protocol):
    """Converts subtitle cues from one language to another."""

    def translate(
        self, lines: list[SubtitleLine], *, source: str, target: str
    ) -> list[SubtitleLine]:  # pragma: no cover - protocol definition
        ...


class PassthroughTranslator:
    """Identity translator (returns cues unchanged)."""

    def translate(
        self, lines: list[SubtitleLine], *, source: str, target: str
    ) -> list[SubtitleLine]:
        _ = (source, target)
        return lines


def require_translation(*, source: str, target: str) -> None:
    """Raise until a real translation backend is wired up."""
    raise TranslationNotImplementedError(
        f"Subtitle translation {source!r} -> {target!r} is a planned feature and "
        "is not implemented yet. Re-run without --translate-to."
    )
