from __future__ import annotations

import numpy as np

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
    _SOUNDDEVICE_ERROR = None
except Exception as exc:
    sd = None
    _HAS_SOUNDDEVICE = False
    _SOUNDDEVICE_ERROR = exc


class MicUnavailableError(RuntimeError):
    """Raised when microphone capture is unavailable."""


def record_ambient(duration_sec: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Record mono audio from the system's default microphone."""

    if not _HAS_SOUNDDEVICE:
        raise MicUnavailableError(
            f"sounddevice could not be loaded: {_SOUNDDEVICE_ERROR}"
        )

    num_frames = int(round(duration_sec * sr))

    try:
        recording = sd.rec(
            num_frames,
            samplerate=sr,
            channels=1,
            dtype="float32",
        )
        sd.wait()

    except Exception as exc:
        raise MicUnavailableError(
            f"Microphone recording failed: {exc}"
        ) from exc

    audio = recording.reshape(-1).astype(np.float32)

    return np.clip(audio, -1.0, 1.0)


def is_available() -> bool:
    """Return True when at least one microphone input is available."""

    if not _HAS_SOUNDDEVICE:
        return False

    try:
        devices = sd.query_devices()

        return any(
            device.get("max_input_channels", 0) > 0
            for device in devices
        )

    except Exception:
        return False