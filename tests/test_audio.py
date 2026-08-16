"""
Tests for the audio module.

Uses synthetic audio signals (sine waves, white noise) to verify
audio loading, validation, channel conversion, and saving.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio import (
    AudioInfo,
    AudioValidationError,
    audio_to_bytes,
    ensure_mono,
    ensure_stereo,
    get_audio_info,
    load_audio,
    save_audio,
    validate_uploaded_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rate() -> int:
    return 44100


@pytest.fixture
def mono_signal(sample_rate: int) -> np.ndarray:
    """Generate a 1-second mono sine wave at 440 Hz."""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def stereo_signal(mono_signal: np.ndarray) -> np.ndarray:
    """Generate a stereo signal: left = 440 Hz, right = 880 Hz."""
    t = np.linspace(0, 1.0, len(mono_signal), endpoint=False)
    right = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    return np.stack([mono_signal, right], axis=0)


@pytest.fixture
def temp_wav_path(mono_signal: np.ndarray, sample_rate: int) -> Path:
    """Save mono signal to a temporary WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, mono_signal, sample_rate, subtype="PCM_16")
        return Path(f.name)


@pytest.fixture
def temp_stereo_wav_path(stereo_signal: np.ndarray, sample_rate: int) -> Path:
    """Save stereo signal to a temporary WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, stereo_signal.T, sample_rate, subtype="PCM_16")
        return Path(f.name)


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_wav_file(self, mono_signal: np.ndarray, sample_rate: int):
        """Valid WAV file should pass validation."""
        buf = io.BytesIO()
        sf.write(buf, mono_signal, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        # Should not raise
        validate_uploaded_file(buf, "test.wav")

    def test_empty_file_rejected(self):
        """Empty file should be rejected."""
        buf = io.BytesIO(b"")
        with pytest.raises(AudioValidationError, match="empty"):
            validate_uploaded_file(buf, "empty.wav")

    def test_unsupported_format_rejected(self, mono_signal: np.ndarray, sample_rate: int):
        """Unsupported file extension should be rejected."""
        buf = io.BytesIO()
        sf.write(buf, mono_signal, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        with pytest.raises(AudioValidationError, match="Unsupported"):
            validate_uploaded_file(buf, "test.xyz")

    def test_large_file_rejected(self, mono_signal: np.ndarray, sample_rate: int):
        """File exceeding size limit should be rejected."""
        buf = io.BytesIO()
        sf.write(buf, mono_signal, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        # Set max to 0.0001 MB to trigger size check
        with pytest.raises(AudioValidationError, match="too large"):
            validate_uploaded_file(buf, "test.wav", max_size_mb=0.0001)


# ---------------------------------------------------------------------------
# Loading Tests
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_mono_wav(self, temp_wav_path: Path, sample_rate: int):
        """Loading a mono WAV should return correct shape and sample rate."""
        audio, sr = load_audio(temp_wav_path, target_sr=sample_rate, mono=True)
        assert sr == sample_rate
        assert audio.ndim == 1
        assert len(audio) == sample_rate  # 1 second

    def test_load_preserves_duration(self, temp_wav_path: Path, sample_rate: int):
        """Loaded audio duration should match original."""
        audio, sr = load_audio(temp_wav_path, target_sr=sample_rate, mono=True)
        duration = len(audio) / sr
        assert abs(duration - 1.0) < 0.01  # ~1 second

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file should raise AudioValidationError."""
        with pytest.raises(AudioValidationError, match="not found"):
            load_audio("/nonexistent/file.wav")

    def test_audio_info(self, temp_wav_path: Path, sample_rate: int):
        """Audio info extraction should return correct metadata."""
        info = get_audio_info(temp_wav_path, "test.wav")
        assert isinstance(info, AudioInfo)
        assert info.filename == "test.wav"
        assert info.sample_rate == sample_rate
        assert info.channels == 1
        assert abs(info.duration - 1.0) < 0.01

    def test_audio_info_formatted_properties(self, temp_wav_path: Path):
        """AudioInfo formatted properties should return strings."""
        info = get_audio_info(temp_wav_path)
        assert isinstance(info.duration_formatted, str)
        assert isinstance(info.file_size_formatted, str)
        assert isinstance(info.sample_rate_khz, str)


# ---------------------------------------------------------------------------
# Channel Conversion Tests
# ---------------------------------------------------------------------------

class TestChannelConversion:
    def test_mono_stays_mono(self, mono_signal: np.ndarray):
        """Mono signal through ensure_mono should remain unchanged."""
        result = ensure_mono(mono_signal)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, mono_signal)

    def test_stereo_to_mono(self, stereo_signal: np.ndarray):
        """Stereo signal through ensure_mono should average channels."""
        result = ensure_mono(stereo_signal)
        assert result.ndim == 1
        expected = np.mean(stereo_signal, axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mono_to_stereo(self, mono_signal: np.ndarray):
        """Mono signal through ensure_stereo should duplicate to 2 channels."""
        result = ensure_stereo(mono_signal)
        assert result.ndim == 2
        assert result.shape[0] == 2
        np.testing.assert_array_equal(result[0], mono_signal)
        np.testing.assert_array_equal(result[1], mono_signal)

    def test_stereo_stays_stereo(self, stereo_signal: np.ndarray):
        """Stereo signal through ensure_stereo should remain unchanged."""
        result = ensure_stereo(stereo_signal)
        assert result.ndim == 2
        assert result.shape[0] == 2


# ---------------------------------------------------------------------------
# Saving Tests
# ---------------------------------------------------------------------------

class TestSaving:
    def test_save_mono_wav(self, mono_signal: np.ndarray, sample_rate: int, tmp_path: Path):
        """Saving mono audio should produce a valid WAV file."""
        out_path = tmp_path / "output.wav"
        result = save_audio(mono_signal, out_path, sample_rate)
        assert result.exists()
        assert result.stat().st_size > 0

        # Verify readable
        data, sr = sf.read(str(result))
        assert sr == sample_rate

    def test_save_stereo_wav(self, stereo_signal: np.ndarray, sample_rate: int, tmp_path: Path):
        """Saving stereo audio should produce a valid 2-channel WAV."""
        out_path = tmp_path / "output_stereo.wav"
        save_audio(stereo_signal, out_path, sample_rate)

        data, sr = sf.read(str(out_path))
        assert sr == sample_rate
        assert data.ndim == 2
        assert data.shape[1] == 2

    def test_audio_to_bytes(self, mono_signal: np.ndarray, sample_rate: int):
        """audio_to_bytes should return valid WAV bytes."""
        wav_bytes = audio_to_bytes(mono_signal, sample_rate)
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 44  # WAV header is 44 bytes
        # Verify it starts with RIFF header
        assert wav_bytes[:4] == b"RIFF"

    def test_clipping_on_save(self, sample_rate: int, tmp_path: Path):
        """Audio values outside [-1, 1] should be clipped on save."""
        loud_signal = np.array([2.0, -3.0, 1.5, -0.5, 0.0], dtype=np.float32)
        out_path = tmp_path / "clipped.wav"
        save_audio(loud_signal, out_path, sample_rate)

        data, _ = sf.read(str(out_path))
        assert data.max() <= 1.0
        assert data.min() >= -1.0
