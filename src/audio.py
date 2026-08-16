"""
Audio loading, validation, and preprocessing module.

Handles all audio I/O operations including:
- File validation (format, size, integrity)
- Audio loading with format conversion
- Sample rate conversion and channel normalization
- Audio metadata extraction
- WAV output saving
"""

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf

from src.utils import (
    DEFAULT_SAMPLE_RATE,
    MAX_DURATION_SECONDS,
    MAX_FILE_SIZE_MB,
    OUTPUT_SAMPLE_RATE,
    OUTPUT_SUBTYPE,
    SUPPORTED_EXTENSIONS,
    format_duration,
    format_file_size,
    get_file_extension,
    sanitize_filename,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AudioInfo:
    """Metadata about an audio file."""

    filename: str
    duration: float  # seconds
    sample_rate: int
    channels: int
    file_size_bytes: int
    num_samples: int

    @property
    def duration_formatted(self) -> str:
        """Human-readable duration string."""
        return format_duration(self.duration)

    @property
    def file_size_formatted(self) -> str:
        """Human-readable file size string."""
        return format_file_size(self.file_size_bytes)

    @property
    def sample_rate_khz(self) -> str:
        """Sample rate formatted in kHz."""
        return f"{self.sample_rate / 1000:.1f} kHz"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class AudioValidationError(Exception):
    """Raised when audio validation fails."""
    pass


def validate_uploaded_file(
    uploaded_file: BinaryIO,
    filename: str,
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> None:
    """
    Validate an uploaded audio file before processing.

    Checks:
    1. File is not empty
    2. File extension is supported
    3. File size is within limits

    Args:
        uploaded_file: The uploaded file object (with read/seek).
        filename: Original filename.
        max_size_mb: Maximum allowed file size in MB.

    Raises:
        AudioValidationError: If validation fails.
    """
    # Check for empty file
    uploaded_file.seek(0, 2)  # Seek to end
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)  # Reset to beginning

    if file_size == 0:
        raise AudioValidationError("The uploaded file is empty.")

    # Check file extension
    ext = get_file_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioValidationError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {supported}"
        )

    # Check file size
    max_bytes = max_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise AudioValidationError(
            f"File is too large ({format_file_size(file_size)}). "
            f"Maximum allowed size is {max_size_mb:.0f} MB."
        )


def validate_audio_duration(
    duration: float,
    max_duration: float = MAX_DURATION_SECONDS,
) -> None:
    """
    Validate that audio duration is within acceptable limits.

    Args:
        duration: Audio duration in seconds.
        max_duration: Maximum allowed duration in seconds.

    Raises:
        AudioValidationError: If duration exceeds the limit.
    """
    if duration <= 0:
        raise AudioValidationError("Audio file appears to have zero duration.")

    if duration > max_duration:
        raise AudioValidationError(
            f"Audio is too long ({format_duration(duration)}). "
            f"Maximum allowed duration is {format_duration(max_duration)}."
        )


# ---------------------------------------------------------------------------
# File I/O Helpers
# ---------------------------------------------------------------------------

def save_uploaded_file(uploaded_file: BinaryIO, filename: str, temp_dir: Path) -> Path:
    """
    Save an uploaded file to a temporary directory.

    Args:
        uploaded_file: The uploaded file object.
        filename: Original filename (will be sanitized).
        temp_dir: Directory to save the file in.

    Returns:
        Path to the saved file.
    """
    safe_name = sanitize_filename(filename)
    file_path = temp_dir / safe_name
    uploaded_file.seek(0)
    file_path.write_bytes(uploaded_file.read())
    return file_path


# ---------------------------------------------------------------------------
# Audio Loading
# ---------------------------------------------------------------------------

def load_audio(
    file_path: str | Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file and optionally resample it.

    Uses librosa for robust format handling (leveraging FFmpeg backend
    for MP3, M4A, etc.). Returns audio as a float32 numpy array
    normalized to [-1.0, 1.0].

    Args:
        file_path: Path to the audio file.
        target_sr: Target sample rate. If None, uses the native rate.
        mono: If True, convert to mono. If False, preserve channels.

    Returns:
        Tuple of (audio_array, sample_rate).
        - If mono=True: shape is (num_samples,)
        - If mono=False and stereo: shape is (2, num_samples)
        - If mono=False and mono file: shape is (num_samples,)

    Raises:
        AudioValidationError: If the file cannot be loaded.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise AudioValidationError(f"Audio file not found: {file_path}")

    try:
        audio, sr = librosa.load(
            str(file_path),
            sr=target_sr,
            mono=mono,
        )
        return audio, sr
    except Exception as e:
        raise AudioValidationError(
            f"Failed to load audio file: {e}. "
            "The file may be corrupted or in an unsupported format. "
            "Ensure FFmpeg is installed for MP3/M4A support."
        ) from e


def get_audio_info(file_path: str | Path, filename: str | None = None) -> AudioInfo:
    """
    Extract metadata from an audio file without fully decoding it.

    Args:
        file_path: Path to the audio file.
        filename: Display name for the file. Defaults to the basename.

    Returns:
        AudioInfo with metadata about the file.

    Raises:
        AudioValidationError: If metadata cannot be extracted.
    """
    file_path = Path(file_path)
    if filename is None:
        filename = file_path.name

    try:
        # Use soundfile for WAV/FLAC (fast metadata)
        try:
            info = sf.info(str(file_path))
            return AudioInfo(
                filename=filename,
                duration=info.duration,
                sample_rate=info.samplerate,
                channels=info.channels,
                file_size_bytes=file_path.stat().st_size,
                num_samples=info.frames,
            )
        except Exception:
            pass

        # Fallback: use librosa (handles MP3, M4A via FFmpeg)
        duration = librosa.get_duration(path=str(file_path))
        # Load a tiny portion to get sample rate and channels
        y, sr = librosa.load(str(file_path), sr=None, mono=False, duration=0.1)
        channels = 1 if y.ndim == 1 else y.shape[0]
        num_samples = int(duration * sr)

        return AudioInfo(
            filename=filename,
            duration=duration,
            sample_rate=sr,
            channels=channels,
            file_size_bytes=file_path.stat().st_size,
            num_samples=num_samples,
        )
    except Exception as e:
        raise AudioValidationError(
            f"Could not read audio metadata: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Channel Conversion
# ---------------------------------------------------------------------------

def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono by averaging channels.

    Args:
        audio: Audio array. Shape (samples,) or (channels, samples).

    Returns:
        Mono audio array with shape (samples,).
    """
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0)
    raise AudioValidationError(
        f"Unexpected audio shape: {audio.shape}. Expected 1D or 2D array."
    )


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to stereo by duplicating mono channel.

    Args:
        audio: Audio array. Shape (samples,) or (2, samples).

    Returns:
        Stereo audio array with shape (2, samples).
    """
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=0)
    if audio.ndim == 2 and audio.shape[0] == 2:
        return audio
    if audio.ndim == 2 and audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    raise AudioValidationError(
        f"Unexpected audio shape: {audio.shape}. Expected 1D or (2, samples)."
    )


# ---------------------------------------------------------------------------
# Audio Saving
# ---------------------------------------------------------------------------

def save_audio(
    audio: np.ndarray,
    file_path: str | Path,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    subtype: str = OUTPUT_SUBTYPE,
) -> Path:
    """
    Save audio data as a WAV file.

    Converts float32 audio to the specified PCM format.
    The output is always a valid WAV file playable by Streamlit.

    Args:
        audio: Audio array. Shape (samples,) for mono or (channels, samples)
               for multi-channel. Values should be in [-1.0, 1.0].
        file_path: Output file path.
        sample_rate: Output sample rate in Hz.
        subtype: SoundFile subtype (e.g., 'PCM_16', 'PCM_24').

    Returns:
        Path to the saved file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Clip to prevent distortion
    audio = np.clip(audio, -1.0, 1.0)

    # Transpose for soundfile: expects (samples, channels)
    if audio.ndim == 2:
        audio_out = audio.T
    else:
        audio_out = audio

    sf.write(
        str(file_path),
        audio_out,
        samplerate=sample_rate,
        subtype=subtype,
    )
    return file_path


def audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    subtype: str = OUTPUT_SUBTYPE,
) -> bytes:
    """
    Convert audio array to WAV bytes for Streamlit download.

    Args:
        audio: Audio array.
        sample_rate: Sample rate in Hz.
        subtype: SoundFile subtype.

    Returns:
        WAV file content as bytes.
    """
    audio = np.clip(audio, -1.0, 1.0)
    if audio.ndim == 2:
        audio_out = audio.T
    else:
        audio_out = audio

    buffer = io.BytesIO()
    sf.write(buffer, audio_out, samplerate=sample_rate, subtype=subtype, format="WAV")
    buffer.seek(0)
    return buffer.read()
