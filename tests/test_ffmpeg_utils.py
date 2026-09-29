"""Tests for the ffmpeg wrappers that do not need the binaries."""

from __future__ import annotations

import pytest

from anime2manga.ffmpeg_utils import uniform_sample_times


def test_uniform_sample_times_keeps_interior_grid():
    times = uniform_sample_times(0.0, 1.0, 4.0, 4)
    assert times == [0.0, 0.25, 0.5, 0.75]


def test_uniform_sample_times_drops_end_boundary_frame():
    # ffmpeg can emit one extra frame exactly on ``end``; it belongs to the next
    # scene and must not be stitched into a panorama or reused for its frame.
    times = uniform_sample_times(100.0, 104.5, 4.0, 19)
    assert len(times) == 18
    assert times[-1] == pytest.approx(104.25)
    assert all(t < 104.5 for t in times)


def test_uniform_sample_times_handles_empty_and_bad_fps():
    assert uniform_sample_times(0.0, 1.0, 4.0, 0) == []
    assert uniform_sample_times(0.0, 1.0, 0.0, 4) == []
