"""Step 7 - left/right audio focus detection.

Decides which side of a panel subtitles should favour, based on which stereo
channel carries more energy during the scene.  This is what lets step 10 place a
speech bubble on the side the voice is coming from.

Approach
--------
* Use the audio stream selected in step 1 (E-AC-3, AAC, Vorbis, FLAC, ... any
  codec ffmpeg can decode).  If the source is mono (< 2 channels) there is no
  left/right pair to compare, so the scene is reported as ``"center"`` without
  decoding anything.
* Decode just the scene's span to raw ``f32le`` and let ffmpeg downmix to two
  channels (``-ac 2``), folding 5.1/7.1 into L/R with ffmpeg's standard
  coefficients.  Only the energy balance matters, so analysis runs at a low
  sample rate.
* Compare the two channels inside the **speech band** (300-3400 Hz by default)
  rather than full-band: dialogue lives there, while centered music and
  ambience sit mostly outside it and otherwise mask a panned voice.
  ``balance_db = 10*log10(power_left / power_right)`` in that band; positive
  favours the left channel, negative the right.
* The same speech-band power gates silence: a scene whose band level is below
  ``silence_floor_db`` is ``"center"``.  No full-band measurement feeds the
  direction decision.  Otherwise a ``|balance_db|`` of at least
  ``balance_threshold_db`` names a side; anything smaller is ``"center"``.

The result is stored on ``Scene.audio_focus`` (``"left"`` / ``"center"`` /
``"right"``) with the measured ``Scene.audio_balance_db`` kept for diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg_utils import run_bytes
from .models import MediaInfo, Scene

#: Replaces a zero energy term so the dB ratio stays finite (and very large).
_EPS = 1e-12


@dataclass(frozen=True)
class AudioFocusConfig:
    """Tuning for :func:`detect_audio_focus`."""

    #: dB imbalance required before naming a side rather than "center".
    balance_threshold_db: float = 1.5
    #: Scenes whose speech-band power is below this (dBFS) are treated as centered.
    silence_floor_db: float = -60.0
    #: Sample rate used for the analysis decode; only energy matters.
    analysis_sample_rate: int = 16000
    #: Speech band (Hz) used for the balance comparison.  Restricting the
    #: measurement to where dialogue lives keeps centered music/ambience from
    #: masking a panned voice.
    band_low_hz: float = 300.0
    band_high_hz: float = 3400.0


@dataclass(frozen=True)
class AudioFocus:
    """Where a scene's audio sits, plus the measurement behind the call."""

    direction: str = "center"
    balance_db: float | None = None
    reason: str = ""


def _band_channel_power(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    *,
    block_size: int = 1 << 16,
) -> tuple[float, float] | None:
    """Return mean speech-band power per channel ``(left, right)``.

    The signal is filtered to ``[low_hz, high_hz]`` by zeroing the out-of-band
    FFT bins and transforming back, then its mean square is measured in the
    time domain.  That makes the result a real power value in dBFS, comparable
    with the silence floor.  The scene is processed in non-overlapping blocks so
    memory stays bounded for long scenes.  ``None`` when no block has a bin in
    the band (e.g. a scene too short to resolve it).
    """
    n = samples.shape[0]
    nyquist = sample_rate / 2.0
    high = min(high_hz, nyquist) if high_hz > 0 else nyquist
    total = np.zeros(2, dtype=np.float64)
    counted = 0
    for start in range(0, n, block_size):
        chunk = samples[start : start + block_size]
        length = chunk.shape[0]
        if length < 32:
            continue
        spectrum = np.fft.rfft(chunk, axis=0)
        freqs = np.fft.rfftfreq(length, d=1.0 / sample_rate)
        mask = (freqs >= low_hz) & (freqs <= high)
        if not np.any(mask):
            continue
        spectrum[~mask, :] = 0.0
        filtered = np.fft.irfft(spectrum, n=length, axis=0)
        total += np.sum(filtered**2, axis=0)
        counted += length
    if counted == 0:
        return None
    return float(total[0] / counted), float(total[1] / counted)


def classify_stereo(
    samples: np.ndarray,
    *,
    sample_rate: int,
    config: AudioFocusConfig | None = None,
) -> AudioFocus:
    """Classify a ``(n, 2)`` float array of stereo samples.

    Both the silence gate and the left/right balance use speech-band power only
    (see :class:`AudioFocusConfig`); no full-band measurement feeds the
    direction decision.  Positive ``balance_db`` favours the left channel,
    negative the right.
    """
    cfg = config or AudioFocusConfig()
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        return AudioFocus("center", None, "no stereo samples")

    band = _band_channel_power(array, sample_rate, cfg.band_low_hz, cfg.band_high_hz)
    if band is None:
        return AudioFocus("center", None, "no speech-band energy")
    band_left, band_right = band
    overall = (band_left + band_right) / 2.0
    if overall <= 0:
        return AudioFocus("center", None, "silent")
    overall_db = 10.0 * math.log10(overall)
    if overall_db < cfg.silence_floor_db:
        return AudioFocus("center", None, f"below silence floor ({overall_db:.1f} dB)")

    balance_db = 10.0 * math.log10((band_left + _EPS) / (band_right + _EPS))
    if balance_db >= cfg.balance_threshold_db:
        return AudioFocus("left", balance_db, "left channel louder")
    if balance_db <= -cfg.balance_threshold_db:
        return AudioFocus("right", balance_db, "right channel louder")
    return AudioFocus("center", balance_db, "channels balanced")


def decode_scene_stereo(
    path: Path,
    start: float,
    duration: float,
    *,
    stream_index: int,
    sample_rate: int,
    timeout: float | None = None,
) -> np.ndarray:
    """Decode ``[start, start + duration)`` to a ``(n, 2)`` float32 array.

    ``-map 0:<stream_index>`` selects the exact stream ffprobe chose and
    ``-ac 2`` downmixes 5.1/7.1 to stereo.  Seeking with ``-ss`` before ``-i``
    keeps it fast; only the small analysis span is ever decoded.
    """
    duration = max(duration, 0.0)
    if duration <= 0:
        return np.empty((0, 2), dtype=np.float32)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{max(start, 0.0):.3f}",
        "-i",
        str(path),
        "-map",
        f"0:{stream_index}",
        "-t",
        f"{duration:.3f}",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    data = run_bytes(cmd, timeout=timeout)
    pcm = np.frombuffer(data, dtype="<f4")
    if pcm.size % 2:  # guard against a truncated final frame
        pcm = pcm[:-1]
    return pcm.reshape(-1, 2)


def detect_audio_focus(
    media: MediaInfo,
    scene: Scene,
    *,
    config: AudioFocusConfig | None = None,
) -> AudioFocus:
    """Measure where ``scene``'s audio sits: ``left`` / ``center`` / ``right``."""
    cfg = config or AudioFocusConfig()
    info = media.audio
    if info is None:
        return AudioFocus("center", None, "no audio stream")
    if info.is_mono:
        return AudioFocus("center", None, "mono source")

    samples = decode_scene_stereo(
        media.path,
        scene.start,
        scene.duration,
        stream_index=info.index,
        sample_rate=cfg.analysis_sample_rate,
    )
    return classify_stereo(samples, sample_rate=cfg.analysis_sample_rate, config=cfg)
