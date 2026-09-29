"""Tests for step 7 - left/right audio focus detection."""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from anime2manga.audio import (
    AudioFocus,
    AudioFocusConfig,
    classify_stereo,
    decode_scene_stereo,
    detect_audio_focus,
)
from anime2manga.models import AudioInfo

#: Analysis sample rate used throughout the synthetic tests.
SR = 16000


def _tone(
    freq: float, *, n: int = 8000, amp: float = 0.5, sample_rate: int = SR
) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1).astype(np.float32)


def test_classify_left_dominant():
    samples = _stereo(_tone(1000.0), np.zeros(8000, dtype=np.float32))
    focus = classify_stereo(samples, sample_rate=SR)
    assert focus.direction == "left"
    assert focus.balance_db is not None and focus.balance_db > 0


def test_classify_right_dominant():
    samples = _stereo(np.zeros(8000, dtype=np.float32), _tone(1000.0))
    focus = classify_stereo(samples, sample_rate=SR)
    assert focus.direction == "right"
    assert focus.balance_db is not None and focus.balance_db < 0


def test_classify_identical_channels_is_center():
    wave = _tone(1000.0)
    focus = classify_stereo(_stereo(wave, wave), sample_rate=SR)
    assert focus.direction == "center"
    assert focus.balance_db == pytest.approx(0.0, abs=1e-6)


def test_classify_small_imbalance_is_center():
    left = _tone(1000.0)
    right = (left * 1.1).astype(np.float32)  # ~0.83 dB, below the 1.5 dB default
    focus = classify_stereo(_stereo(left, right), sample_rate=SR)
    assert focus.direction == "center"


def test_classify_threshold_boundary():
    # A 1 Hz-wide bin grid makes the tone fall on an exact FFT bin, so the
    # balance lands on the threshold rather than a smeared value.
    wave = _tone(1000.0, n=SR)
    just_above = (wave * 10 ** (-1.5005 / 20)).astype(np.float32)
    focus = classify_stereo(_stereo(wave, just_above), sample_rate=SR)
    assert focus.balance_db == pytest.approx(1.5, abs=0.02)
    assert focus.direction == "left"  # ">=" names a side at the default 1.5 dB

    just_below = (wave * 10 ** (-1.4995 / 20)).astype(np.float32)
    assert classify_stereo(_stereo(wave, just_below), sample_rate=SR).direction == "center"


def test_classify_silence_is_center():
    focus = classify_stereo(np.zeros((8000, 2), dtype=np.float32), sample_rate=SR)
    assert focus.direction == "center"
    assert focus.balance_db is None


def test_classify_quiet_tone_below_floor_is_center():
    focus = classify_stereo(
        _stereo(_tone(1000.0, amp=1e-5), np.zeros(8000, dtype=np.float32)),
        sample_rate=SR,
    )
    assert focus.direction == "center"
    assert "silence" in focus.reason


def test_centered_bass_with_tiny_left_hiss_stays_center():
    # Regression for the full-band silence gate: a loud centered sub-band bass
    # must not let near-noise speech-band energy name a side.
    wave_l = _tone(1000.0, n=SR, amp=1e-4)
    bass = _tone(100.0, n=SR, amp=0.5)
    focus = classify_stereo(_stereo(bass + wave_l, bass), sample_rate=SR)
    assert focus.direction == "center"


def test_classify_ignores_energy_outside_the_speech_band():
    # The right channel only carries a 6 kHz tone, above the 3400 Hz band, so
    # the speech-band balance must still favour the left channel.
    samples = _stereo(_tone(1000.0), _tone(6000.0))
    focus = classify_stereo(samples, sample_rate=SR)
    assert focus.direction == "left"
    # With a band that includes 6 kHz, the two channels are balanced again.
    wide = AudioFocusConfig(band_low_hz=300.0, band_high_hz=7000.0)
    assert classify_stereo(samples, sample_rate=SR, config=wide).direction == "center"


def test_classify_rejects_non_stereo():
    empty = np.zeros((0, 2), dtype=np.float32)
    assert classify_stereo(np.zeros((100,), dtype=np.float32), sample_rate=SR).direction == "center"
    assert classify_stereo(empty, sample_rate=SR).direction == "center"


def test_classify_respects_custom_threshold():
    left = _tone(1000.0)
    right = (left * 1.1).astype(np.float32)  # ~0.83 dB louder on the right
    assert classify_stereo(_stereo(left, right), sample_rate=SR).direction == "center"
    strict = AudioFocusConfig(balance_threshold_db=0.5)
    focus = classify_stereo(_stereo(left, right), sample_rate=SR, config=strict)
    assert focus.direction == "right"


def test_detect_audio_focus_without_audio_stream(make_media, make_scene):
    media = make_media()
    media = dataclasses.replace(media, audio=None)
    focus = detect_audio_focus(media, make_scene())
    assert focus == AudioFocus("center", None, "no audio stream")


def test_detect_audio_focus_mono_skips_decoding(make_media, make_scene, monkeypatch):
    media = make_media()
    media = dataclasses.replace(
        media,
        audio=AudioInfo(
            index=1,
            codec="aac",
            channels=1,
            channel_layout="mono",
            sample_rate=48000,
        ),
    )

    def fail(*args, **kwargs):
        raise AssertionError("mono sources must not be decoded")

    monkeypatch.setattr("anime2manga.audio.decode_scene_stereo", fail)
    focus = detect_audio_focus(media, make_scene())
    assert focus.direction == "center"
    assert "mono" in focus.reason


def test_audio_info_labels_and_mono_flag():
    mono = AudioInfo(1, "aac", 1, "mono", 48000, language="jpn")
    assert mono.is_mono
    assert "jpn" in mono.label and "mono" in mono.label
    stereo = AudioInfo(1, "eac3", 2, "stereo", 48000, title="Surround")
    assert not stereo.is_mono
    assert "Surround" in stereo.label


def test_audio_info_unknown_channel_count_is_not_mono():
    # channels == 0 means ffprobe could not report it; the decode path should
    # still get a chance rather than silently skipping detection.
    assert not AudioInfo(1, "aac", 0, "", 48000).is_mono


def _write_one_sided_vorbis(path: Path, *, left_only: bool) -> None:
    """Write a 1 s stereo OGG/Vorbis file with signal in one channel only."""
    pan = "c0=c0|c1=0*c0" if left_only else "c0=0*c0|c1=c0"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1:sample_rate=44100",
            "-af",
            f"pan=stereo|{pan}",
            "-c:a",
            "libvorbis",
            str(path),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("left_only", "expected"), [(True, "left"), (False, "right")]
)
def test_detect_audio_focus_on_generated_vorbis(
    tmp_path, make_media, make_scene, left_only, expected
):
    """Reproducible end-to-end decode check using a generated Vorbis fixture."""
    path = tmp_path / ("left.ogg" if left_only else "right.ogg")
    _write_one_sided_vorbis(path, left_only=left_only)
    media = make_media(duration=2.0, fps=24.0)
    media = dataclasses.replace(
        media,
        path=path,
        audio=AudioInfo(
            index=0,
            codec="vorbis",
            channels=2,
            channel_layout="stereo",
            sample_rate=44100,
        ),
    )
    focus = detect_audio_focus(media, make_scene(start=0.0, end=0.9))
    assert focus.direction == expected


def _external_sample_dir() -> Path:
    """Where optional real-world samples live, if the developer has them."""
    override = os.environ.get("ANIME2MANGA_AUDIO_SAMPLES")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "data"


_SAMPLE_DIR = _external_sample_dir()
_HAS_SAMPLES = (_SAMPLE_DIR / "audiocheck.net_L.ogg").exists() and (
    _SAMPLE_DIR / "audiocheck.net_R.ogg"
).exists()


@pytest.mark.skipif(
    not _HAS_SAMPLES,
    reason="set ANIME2MANGA_AUDIO_SAMPLES or drop the ogg samples in tests/data/",
)
@pytest.mark.parametrize(
    ("filename", "expected"),
    [("audiocheck.net_L.ogg", "left"), ("audiocheck.net_R.ogg", "right")],
)
def test_detect_audio_focus_on_real_ogg_samples(
    make_media, make_scene, filename, expected
):
    """Optional check against real supplied Vorbis L/R samples."""
    media = make_media(duration=2.0, fps=24.0)
    media = dataclasses.replace(
        media,
        path=_SAMPLE_DIR / filename,
        audio=AudioInfo(
            index=0,
            codec="vorbis",
            channels=2,
            channel_layout="stereo",
            sample_rate=44100,
        ),
    )
    focus = detect_audio_focus(media, make_scene(start=0.0, end=1.0))
    assert focus.direction == expected


def test_decode_scene_stereo_returns_stereo_shape(tmp_path):
    # A tiny generated stereo clip exercises the ffmpeg decode path without
    # depending on any external fixture.
    path = tmp_path / "clip.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5",
            "-ac",
            "2",
            str(path),
        ],
        check=True,
    )
    samples = decode_scene_stereo(path, 0.0, 0.4, stream_index=0, sample_rate=16000)
    assert samples.ndim == 2 and samples.shape[1] == 2
    assert samples.shape[0] > 0
