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
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Formats that libsndfile (soundfile) can handle natively
_SOUNDFILE_FORMATS: set[str] = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


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
# FFmpeg / FFprobe Helpers
# ---------------------------------------------------------------------------

def _ensure_path_env() -> None:
    """Ensure standard binary paths are in PATH environment variable."""
    standard_paths = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    current_path = os.environ.get("PATH", "")
    paths_to_add = [p for p in standard_paths if p not in current_path and os.path.exists(p)]
    if paths_to_add:
        os.environ["PATH"] = ":".join(paths_to_add) + ":" + current_path


_ensure_path_env()


def get_ffmpeg_path() -> str:
    """
    Locate the ffmpeg binary on the system.

    Checks PATH as well as standard macOS Homebrew and Linux locations.

    Returns:
        Absolute path to ffmpeg or 'ffmpeg' if in PATH.

    Raises:
        AudioValidationError: If ffmpeg cannot be found anywhere.
    """
    import shutil

    # Try standard PATH lookup
    found = shutil.which("ffmpeg")
    if found:
        return found

    # Fallback to known absolute paths
    candidates = [
        "/opt/homebrew/bin/ffmpeg",  # Apple Silicon macOS Homebrew
        "/usr/local/bin/ffmpeg",     # Intel macOS Homebrew / Linux
        "/usr/bin/ffmpeg",           # Standard Linux
        "/bin/ffmpeg",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise AudioValidationError(
        "FFmpeg is not installed or not found on this system. "
        "Please install FFmpeg to process M4A, MP3, and other compressed audio formats:\n"
        "- macOS: brew install ffmpeg\n"
        "- Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg\n"
        "- Fedora: sudo dnf install ffmpeg"
    )


def get_ffprobe_path() -> str:
    """
    Locate the ffprobe binary on the system.

    Returns:
        Absolute path to ffprobe or 'ffprobe'.
    """
    import shutil

    found = shutil.which("ffprobe")
    if found:
        return found

    candidates = [
        "/opt/homebrew/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "/usr/bin/ffprobe",
        "/bin/ffprobe",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return "ffprobe"


def _ffprobe_info(file_path: str | Path) -> dict | None:
    """
    Extract audio metadata using ffprobe (part of FFmpeg).

    This handles ALL formats that FFmpeg supports, including M4A, AAC,
    MP3, OGG, WMA, etc. — formats that libsndfile cannot read.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dict with keys 'duration', 'sample_rate', 'channels', or None if ffprobe fails.
    """
    try:
        ffprobe_bin = get_ffprobe_path()
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find the audio stream
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if audio_stream is None:
            return None

        duration = float(data.get("format", {}).get("duration", 0))
        if duration == 0:
            duration = float(audio_stream.get("duration", 0))

        sample_rate = int(audio_stream.get("sample_rate", 44100))
        channels = int(audio_stream.get("channels", 2))

        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.debug(f"ffprobe failed for {file_path}: {e}")
        return None


def _convert_to_wav(file_path: str | Path, output_dir: Path | None = None) -> Path:
    """
    Convert any audio file to WAV format using direct ffmpeg CLI.

    This is necessary for formats like M4A/AAC/MP3 that libsndfile cannot read,
    and directly uses ffmpeg to avoid Python 3.13+ audioop standard library removal issues.
    The conversion produces a 16-bit PCM WAV that all Python audio libraries
    can handle natively.

    Args:
        file_path: Path to the source audio file.
        output_dir: Directory for the output WAV. Uses same dir as input if None.

    Returns:
        Path to the converted WAV file.

    Raises:
        AudioValidationError: If conversion fails.
    """
    file_path = Path(file_path)
    if output_dir is None:
        output_dir = file_path.parent

    wav_path = output_dir / (file_path.stem + ".wav")

    # Skip if already WAV
    if file_path.suffix.lower() == ".wav" and file_path == wav_path:
        return file_path

    try:
        ffmpeg_bin = get_ffmpeg_path()
        logger.info(f"Converting {file_path.suffix} to WAV via FFmpeg ({ffmpeg_bin}): {file_path.name}")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(file_path),
            "-vn",
            "-acodec", "pcm_s16le",
            str(wav_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {res.stderr}")

        logger.info(f"Conversion complete: {wav_path.name}")
        return wav_path

    except AudioValidationError:
        raise
    except Exception as e:
        raise AudioValidationError(
            f"Failed to convert {file_path.suffix} to WAV: {e}. "
            "Ensure FFmpeg is installed: brew install ffmpeg (macOS) "
            "or sudo apt install ffmpeg (Linux)."
        ) from e


def _needs_conversion(file_path: str | Path) -> bool:
    """Check if a file needs conversion to WAV before processing."""
    return Path(file_path).suffix.lower() not in _SOUNDFILE_FORMATS


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

    For formats not supported by libsndfile (M4A, AAC, MP3, WMA),
    the file is first converted to WAV using pydub/FFmpeg.

    Returns audio as a float32 numpy array normalized to [-1.0, 1.0].

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

    load_path = file_path

    # Convert non-WAV/FLAC formats to WAV first
    if _needs_conversion(file_path):
        try:
            load_path = _convert_to_wav(file_path)
        except AudioValidationError:
            raise
        except Exception as e:
            raise AudioValidationError(
                f"Failed to convert {file_path.suffix} audio: {e}. "
                "Ensure FFmpeg is installed for MP3/M4A support."
            ) from e

    try:
        audio, sr = librosa.load(
            str(load_path),
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

    Strategy:
    1. For WAV/FLAC/OGG: use soundfile (fast, native)
    2. For M4A/MP3/AAC/WMA: use ffprobe (handles all FFmpeg formats)
    3. Final fallback: convert to WAV via pydub, then read with soundfile

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

    # --- Strategy 1: soundfile for natively supported formats ---
    if not _needs_conversion(file_path):
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

    # --- Strategy 2: ffprobe for M4A/MP3/AAC and other FFmpeg formats ---
    probe = _ffprobe_info(file_path)
    if probe is not None and probe["duration"] > 0:
        return AudioInfo(
            filename=filename,
            duration=probe["duration"],
            sample_rate=probe["sample_rate"],
            channels=probe["channels"],
            file_size_bytes=file_path.stat().st_size,
            num_samples=int(probe["duration"] * probe["sample_rate"]),
        )

    # --- Strategy 3: convert to WAV and read with soundfile ---
    try:
        wav_path = _convert_to_wav(file_path)
        info = sf.info(str(wav_path))
        return AudioInfo(
            filename=filename,
            duration=info.duration,
            sample_rate=info.samplerate,
            channels=info.channels,
            file_size_bytes=file_path.stat().st_size,  # Report original file size
            num_samples=info.frames,
        )
    except Exception as e:
        raise AudioValidationError(
            f"Could not read audio metadata from '{filename}': {e}. "
            f"The file format ({file_path.suffix}) may not be supported, "
            "or FFmpeg may not be installed."
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
    Convert audio to stereo (2 channels).

    Handles:
    - 1D mono -> duplicates to (2, samples)
    - 2D (1, samples) -> duplicates to (2, samples)
    - 2D (2, samples) -> unchanged
    - 2D (>2, samples) -> downmixes multi-channel (e.g., 5.1 surround) to stereo (2, samples)

    Args:
        audio: Audio array.

    Returns:
        Stereo audio array with shape (2, samples).
    """
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=0)
    if audio.ndim == 2:
        if audio.shape[0] == 2:
            return audio
        if audio.shape[0] == 1:
            return np.concatenate([audio, audio], axis=0)
        if audio.shape[0] > 2:
            # Downmix multi-channel: take first 2 channels (standard stereo L/R)
            logger.info(f"Downmixing {audio.shape[0]}-channel audio to stereo")
            return audio[:2, :].copy()
    raise AudioValidationError(
        f"Unexpected audio shape: {audio.shape}. Expected 1D or 2D array."
    )


def prevent_clipping(audio: np.ndarray, headroom_db: float = 0.1) -> np.ndarray:
    """
    Ensure audio signal does not clip past [-1.0, 1.0].

    If the maximum absolute peak exceeds 1.0, scales the entire signal
    down proportionally to preserve dynamics without distortion.

    Args:
        audio: Float32 audio array.
        headroom_db: Safety headroom margin in decibels.

    Returns:
        Normalized audio array with peak <= 1.0.
    """
    max_peak = np.max(np.abs(audio)) if audio.size > 0 else 0.0
    if max_peak > 1.0:
        target_max = 10.0 ** (-headroom_db / 20.0)
        return (audio / max_peak) * target_max
    return audio


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

    # Safely scale down if peaking above 1.0 to prevent harsh distortion
    audio = prevent_clipping(audio)
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
    audio = prevent_clipping(audio)
    audio = np.clip(audio, -1.0, 1.0)
    if audio.ndim == 2:
        audio_out = audio.T
    else:
        audio_out = audio

    buffer = io.BytesIO()
    sf.write(buffer, audio_out, samplerate=sample_rate, subtype=subtype, format="WAV")
    buffer.seek(0)
    return buffer.read()
