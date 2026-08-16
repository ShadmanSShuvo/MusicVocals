"""
Tests for the DSP module.

Verifies FFT, STFT, spectrogram, and spectral feature computations
using synthetic signals with known frequency content.

Testing strategy:
- Generate signals with known properties (pure tones, multi-tone)
- Verify that DSP functions recover the expected properties
- Use tolerance-based assertions for floating-point comparisons
"""

import numpy as np
import pytest

from src.dsp import (
    compute_fft,
    compute_log_spectrogram,
    compute_rms,
    compute_spectral_bandwidth,
    compute_spectral_centroid,
    compute_spectral_rolloff,
    compute_spectrogram,
    compute_stft,
    compute_zero_crossing_rate,
    extract_audio_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rate() -> int:
    return 44100


@pytest.fixture
def sine_440(sample_rate: int) -> np.ndarray:
    """1-second 440 Hz sine wave."""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float64)


@pytest.fixture
def two_tone(sample_rate: int) -> np.ndarray:
    """1-second signal with 440 Hz + 1000 Hz."""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t) +
            0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float64)


@pytest.fixture
def white_noise(sample_rate: int) -> np.ndarray:
    """1-second white noise."""
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, sample_rate).astype(np.float64)


@pytest.fixture
def silence(sample_rate: int) -> np.ndarray:
    """1-second silence."""
    return np.zeros(sample_rate, dtype=np.float64)


# ---------------------------------------------------------------------------
# FFT Tests
# ---------------------------------------------------------------------------

class TestFFT:
    def test_fft_identifies_single_frequency(self, sine_440: np.ndarray, sample_rate: int):
        """FFT of a 440 Hz sine should peak at 440 Hz."""
        freqs, mags = compute_fft(sine_440, sample_rate)

        # Find peak frequency
        peak_idx = np.argmax(mags[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_idx]

        # Should be within 1 Hz of 440
        assert abs(peak_freq - 440) < 1.0, f"Peak at {peak_freq} Hz, expected ~440 Hz"

    def test_fft_identifies_two_frequencies(self, two_tone: np.ndarray, sample_rate: int):
        """FFT of a two-tone signal should have peaks at both frequencies."""
        freqs, mags = compute_fft(two_tone, sample_rate)

        # Find top 2 peaks (excluding DC)
        mags_no_dc = mags[1:].copy()
        freqs_no_dc = freqs[1:]

        # First peak
        idx1 = np.argmax(mags_no_dc)
        peak1 = freqs_no_dc[idx1]
        mags_no_dc[max(0, idx1 - 5) : idx1 + 5] = 0  # Zero out neighborhood

        # Second peak
        idx2 = np.argmax(mags_no_dc)
        peak2 = freqs_no_dc[idx2]

        peaks = sorted([peak1, peak2])
        assert abs(peaks[0] - 440) < 2.0, f"Expected ~440 Hz, got {peaks[0]}"
        assert abs(peaks[1] - 1000) < 2.0, f"Expected ~1000 Hz, got {peaks[1]}"

    def test_fft_output_shapes(self, sine_440: np.ndarray, sample_rate: int):
        """FFT should return matching frequency and magnitude arrays."""
        freqs, mags = compute_fft(sine_440, sample_rate)
        assert len(freqs) == len(mags)
        assert len(freqs) == len(sine_440) // 2 + 1

    def test_fft_handles_stereo(self, sample_rate: int):
        """FFT should handle stereo input by averaging to mono."""
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        stereo = np.stack([np.sin(2 * np.pi * 440 * t),
                          np.sin(2 * np.pi * 440 * t)], axis=0)
        freqs, mags = compute_fft(stereo, sample_rate)
        assert len(freqs) == sample_rate // 2 + 1


# ---------------------------------------------------------------------------
# STFT Tests
# ---------------------------------------------------------------------------

class TestSTFT:
    def test_stft_output_shape(self, sine_440: np.ndarray, sample_rate: int):
        """STFT should return matrix with correct dimensions."""
        n_fft = 2048
        hop_length = 512
        stft = compute_stft(sine_440, sample_rate, n_fft=n_fft, hop_length=hop_length)

        # Frequency bins
        assert stft.shape[0] == n_fft // 2 + 1

        # Number of frames (approximately)
        expected_frames = 1 + (len(sine_440) + n_fft - n_fft) // hop_length
        # Allow some tolerance due to padding
        assert abs(stft.shape[1] - expected_frames) <= 2

    def test_stft_is_complex(self, sine_440: np.ndarray, sample_rate: int):
        """STFT output should be complex-valued."""
        stft = compute_stft(sine_440, sample_rate)
        assert np.iscomplexobj(stft)

    def test_stft_concentrated_energy(self, sine_440: np.ndarray, sample_rate: int):
        """STFT of a pure tone should concentrate energy near that frequency."""
        n_fft = 2048
        stft = compute_stft(sine_440, sample_rate, n_fft=n_fft)
        mag = np.abs(stft)

        # Average magnitude across frames
        avg_mag = np.mean(mag, axis=1)

        # Find peak frequency bin
        peak_bin = np.argmax(avg_mag)
        freq_per_bin = sample_rate / n_fft
        peak_freq = peak_bin * freq_per_bin

        assert abs(peak_freq - 440) < freq_per_bin * 2


# ---------------------------------------------------------------------------
# Spectrogram Tests
# ---------------------------------------------------------------------------

class TestSpectrogram:
    def test_spectrogram_non_negative(self, sine_440: np.ndarray, sample_rate: int):
        """Power spectrogram should be non-negative."""
        stft = compute_stft(sine_440, sample_rate)
        spec = compute_spectrogram(stft)
        assert np.all(spec >= 0)

    def test_spectrogram_shape_matches_stft(self, sine_440: np.ndarray, sample_rate: int):
        """Spectrogram should have same shape as STFT magnitude."""
        stft = compute_stft(sine_440, sample_rate)
        spec = compute_spectrogram(stft)
        assert spec.shape == stft.shape

    def test_log_spectrogram_bounded(self, sine_440: np.ndarray, sample_rate: int):
        """Log spectrogram should have bounded dynamic range."""
        stft = compute_stft(sine_440, sample_rate)
        spec = compute_spectrogram(stft)
        log_spec = compute_log_spectrogram(spec, top_db=80.0)

        dynamic_range = log_spec.max() - log_spec.min()
        assert dynamic_range <= 80.0 + 1e-6


# ---------------------------------------------------------------------------
# Spectral Feature Tests
# ---------------------------------------------------------------------------

class TestSpectralFeatures:
    def test_rms_silence_is_zero(self, silence: np.ndarray, sample_rate: int):
        """RMS of silence should be approximately zero."""
        rms = compute_rms(silence)
        assert np.allclose(rms, 0, atol=1e-10)

    def test_rms_positive_for_signal(self, sine_440: np.ndarray):
        """RMS of a non-silent signal should be positive."""
        rms = compute_rms(sine_440)
        assert np.all(rms > 0)

    def test_spectral_centroid_of_pure_tone(self, sine_440: np.ndarray, sample_rate: int):
        """Spectral centroid of a 440 Hz tone should be near 440 Hz."""
        n_fft = 2048
        stft = compute_stft(sine_440, sample_rate, n_fft=n_fft)
        stft_mag = np.abs(stft)
        centroid = compute_spectral_centroid(stft_mag, sample_rate, n_fft=n_fft)

        # Average centroid should be close to 440 Hz
        mean_centroid = np.mean(centroid)
        assert abs(mean_centroid - 440) < 50, f"Mean centroid {mean_centroid} Hz, expected ~440"

    def test_zcr_of_sine_wave(self, sine_440: np.ndarray, sample_rate: int):
        """ZCR of a 440 Hz sine wave should reflect the frequency."""
        zcr = compute_zero_crossing_rate(sine_440, frame_length=2048, hop_length=512)
        # 440 Hz sine crosses zero ~880 times/sec
        # ZCR is normalized by frame_length
        mean_zcr = np.mean(zcr)
        expected_zcr = 2 * 440 / sample_rate  # Approximate: 2 crossings per cycle
        # Allow generous tolerance
        assert abs(mean_zcr - expected_zcr) < 0.01

    def test_noise_has_higher_zcr_than_tone(
        self, sine_440: np.ndarray, white_noise: np.ndarray
    ):
        """White noise should have higher ZCR than a pure tone."""
        zcr_tone = np.mean(compute_zero_crossing_rate(sine_440))
        zcr_noise = np.mean(compute_zero_crossing_rate(white_noise))
        assert zcr_noise > zcr_tone

    def test_spectral_rolloff_positive(self, sine_440: np.ndarray, sample_rate: int):
        """Spectral rolloff should return positive frequencies."""
        n_fft = 2048
        stft = compute_stft(sine_440, sample_rate, n_fft=n_fft)
        stft_mag = np.abs(stft)
        rolloff = compute_spectral_rolloff(stft_mag, sample_rate, n_fft=n_fft)
        assert np.all(rolloff >= 0)

    def test_spectral_bandwidth_positive(self, sine_440: np.ndarray, sample_rate: int):
        """Spectral bandwidth should be non-negative."""
        n_fft = 2048
        stft = compute_stft(sine_440, sample_rate, n_fft=n_fft)
        stft_mag = np.abs(stft)
        centroid = compute_spectral_centroid(stft_mag, sample_rate, n_fft=n_fft)
        bandwidth = compute_spectral_bandwidth(stft_mag, sample_rate, centroid, n_fft=n_fft)
        assert np.all(bandwidth >= 0)


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_extract_all_features(self, sine_440: np.ndarray, sample_rate: int):
        """extract_audio_features should return all expected keys."""
        features = extract_audio_features(sine_440, sample_rate)

        expected_keys = {
            "stft", "spectrogram", "log_spectrogram",
            "rms", "spectral_centroid", "spectral_bandwidth",
            "spectral_rolloff", "zero_crossing_rate",
        }
        assert set(features.keys()) == expected_keys

    def test_feature_shapes_consistent(self, sine_440: np.ndarray, sample_rate: int):
        """All per-frame features should have consistent length."""
        features = extract_audio_features(sine_440, sample_rate)

        frame_features = ["rms", "spectral_centroid", "spectral_bandwidth",
                         "spectral_rolloff", "zero_crossing_rate"]
        lengths = [len(features[k]) for k in frame_features]
        assert len(set(lengths)) == 1, f"Inconsistent lengths: {dict(zip(frame_features, lengths))}"
