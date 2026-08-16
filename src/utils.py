"""
Utility functions for the Vocal & Music Separator.

Provides file handling, sanitization, formatting, and configuration constants.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

# Maximum upload size in megabytes
MAX_FILE_SIZE_MB: int = 50

# Maximum audio duration in seconds (10 minutes)
MAX_DURATION_SECONDS: float = 600.0

# Duration threshold for large-file warning (seconds)
LARGE_FILE_WARNING_SECONDS: float = 300.0

# Default target sample rate for processing
DEFAULT_SAMPLE_RATE: int = 44100

# Supported audio file extensions (lowercase, with dot)
SUPPORTED_EXTENSIONS: set[str] = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}

# MIME types mapped to audio formats
SUPPORTED_MIME_TYPES: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/x-ms-wma": ".wma",
}

# Output audio format
OUTPUT_FORMAT: str = "wav"
OUTPUT_SUBTYPE: str = "PCM_16"  # 16-bit PCM WAV
OUTPUT_SAMPLE_RATE: int = 44100


# ---------------------------------------------------------------------------
# Filename Sanitization
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """
    Remove unsafe characters from a filename.

    Keeps alphanumeric characters, hyphens, underscores, and dots.
    Replaces spaces with underscores. Strips leading/trailing whitespace.

    Args:
        name: The original filename string.

    Returns:
        A sanitized filename safe for filesystem use.
    """
    # Remove directory components
    name = os.path.basename(name)
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Keep only safe characters
    name = re.sub(r"[^\w.\-]", "", name)
    # Remove leading dots (hidden files)
    name = name.lstrip(".")
    # Fallback for empty result
    if not name:
        name = "uploaded_audio"
    return name


# ---------------------------------------------------------------------------
# Temporary Directory Management
# ---------------------------------------------------------------------------

def create_temp_dir(prefix: str = "vocal_sep_") -> Path:
    """
    Create a temporary directory for processing intermediate files.

    Args:
        prefix: Prefix for the temp directory name.

    Returns:
        Path to the created temporary directory.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    return temp_dir


def cleanup_temp_dir(path: Path) -> None:
    """
    Safely remove a temporary directory and all its contents.

    Args:
        path: Path to the directory to remove.
    """
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string (MM:SS or HH:MM:SS).

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.

    Examples:
        >>> format_duration(125.7)
        '02:05'
        >>> format_duration(3661.0)
        '1:01:01'
    """
    if seconds < 0:
        return "0:00"

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted size string (e.g., '3.5 MB').
    """
    if size_bytes < 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def get_file_extension(filename: str) -> str:
    """
    Extract the lowercase file extension from a filename.

    Args:
        filename: The filename to extract extension from.

    Returns:
        Lowercase extension including the dot (e.g., '.mp3').
    """
    return Path(filename).suffix.lower()


def is_supported_format(filename: str) -> bool:
    """
    Check if a filename has a supported audio extension.

    Args:
        filename: The filename to check.

    Returns:
        True if the extension is supported.
    """
    return get_file_extension(filename) in SUPPORTED_EXTENSIONS
