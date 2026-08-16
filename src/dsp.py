"""
Digital Signal Processing (DSP) module.

Implements core DSP operations for audio analysis:
- Fast Fourier Transform (FFT)
- Short-Time Fourier Transform (STFT)
- Spectrogram computation
- Spectral feature extraction

Mathematical Foundation
-----------------------
Audio signals are time-domain representations: amplitude as a function of time.
The Fourier Transform decomposes a signal into its constituent frequencies,
revealing the frequency-domain representation.

For non-stationary signals like music (where frequency content changes over time),
the Short-Time Fourier Transform (STFT) applies the FFT to short, overlapping
windows of the signal, producing a time-frequency representation (spectrogram).

    x[n]  →  window  →  FFT  →  X(m, k)  →  |X(m, k)|²  →  Spectrogram

Where:
    x[n]     = discrete-time audio signal
    m        = frame (time) index
    k        = frequency bin index
    X(m, k)  = complex STFT coefficient
    |X(m,k)| = magnitude spectrum

Why STFT is Essential for Audio Analysis
-----------------------------------------
A single FFT of an entire song gives the overall frequency content but loses
all temporal information — you can't tell *when* a note was played. The STFT
solves this by computing the FFT over short overlapping frames (typically
20–50 ms), producing a 2D time-frequency map. This is the foundation for:

- Spectrograms (visual representation of frequency content over time)
- Audio source separation models (which operate on STFT representations)
- Feature extraction (spectral centroid, bandwidth, etc.)

The trade-off is the time-frequency resolution: shorter windows give better
time resolution but poorer frequency resolution, and vice versa (Heisenberg
uncertainty principle applied to signal processing).
"""

import numpy as np
from scipy.signal import get_window


# ---------------------------------------------------------------------------
# Fast Fourier Transform (FFT)
# ---------------------------------------------------------------------------

def compute_fft(
    signal: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided FFT of a signal.

    The FFT decomposes the signal into sinusoidal components at discrete
    frequencies. For a real-valued signal of length N, the FFT produces
    N/2 + 1 unique frequency bins (the other half is a mirror image).

    The magnitude |X[k]| indicates the amplitude of each frequency component.

    Args:
        signal: 1D time-domain signal (mono audio).
        sample_rate: Sampling rate in Hz.

    Returns:
        Tuple of (frequencies, magnitudes):
        - frequencies: Array of frequency values in Hz, shape (N//2 + 1,)
        - magnitudes: Magnitude spectrum |X[k]|, shape (N//2 + 1,)
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)

    n = len(signal)

    # Compute the real-valued FFT (single-sided)
    # np.fft.rfft returns complex coefficients for frequencies [0, fs/2]
    fft_complex = np.fft.rfft(signal)

    # Magnitude: |X[k]| = sqrt(Re² + Im²)
    magnitudes = np.abs(fft_complex) / n  # Normalize by signal length

    # Frequency axis: maps bin index k to frequency in Hz
    # f[k] = k * fs / N
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    return frequencies, magnitudes


# ---------------------------------------------------------------------------
# Short-Time Fourier Transform (STFT)
# ---------------------------------------------------------------------------

def compute_stft(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT).

    The STFT segments the signal into overlapping frames, applies a window
    function (to reduce spectral leakage), and computes the FFT of each frame.

    Algorithm:
        For each frame m:
            1. Extract signal segment: x[m*hop : m*hop + n_fft]
            2. Apply window: x_w[n] = x[n] * w[n]
            3. Compute FFT: X(m, k) = FFT(x_w)

    The Hann window is used by default because it provides a good balance
    between frequency resolution and spectral leakage suppression.

    Args:
        signal: 1D time-domain signal (mono).
        sample_rate: Sampling rate in Hz (used for documentation; the STFT
                     itself operates on sample indices).
        n_fft: FFT size (window length). Determines frequency resolution:
               Δf = sample_rate / n_fft. Default 2048 gives ~21.5 Hz
               resolution at 44100 Hz.
        hop_length: Number of samples between successive frames. Controls
                    time resolution: Δt = hop_length / sample_rate.
                    Default 512 gives ~11.6 ms at 44100 Hz.
        window: Window function name (e.g., 'hann', 'hamming', 'blackman').

    Returns:
        Complex STFT matrix of shape (n_fft//2 + 1, num_frames).
        Each column is the FFT of one windowed frame.
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)

    # Generate the window function
    win = get_window(window, n_fft, fftbins=True)

    # Pad the signal so we don't lose the tail
    pad_length = n_fft // 2
    signal_padded = np.pad(signal, (pad_length, pad_length), mode="reflect")

    # Calculate number of frames
    num_frames = 1 + (len(signal_padded) - n_fft) // hop_length

    # Pre-allocate the STFT matrix
    # Frequency bins: n_fft // 2 + 1 (single-sided for real signals)
    n_freq_bins = n_fft // 2 + 1
    stft_matrix = np.zeros((n_freq_bins, num_frames), dtype=np.complex128)

    # Compute FFT for each frame
    for m in range(num_frames):
        start = m * hop_length
        frame = signal_padded[start : start + n_fft]

        # Apply window to reduce spectral leakage
        windowed_frame = frame * win

        # Compute single-sided FFT
        stft_matrix[:, m] = np.fft.rfft(windowed_frame)

    return stft_matrix


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(
    stft_matrix: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """
    Compute the power spectrogram from an STFT matrix.

    The spectrogram is the squared magnitude of the STFT:
        S(m, k) = |X(m, k)|^power

    A power of 2.0 gives the power spectral density.
    A power of 1.0 gives the magnitude spectrogram.

    Args:
        stft_matrix: Complex STFT matrix, shape (freq_bins, frames).
        power: Exponent for magnitude. 2.0 = power, 1.0 = magnitude.

    Returns:
        Spectrogram matrix, shape (freq_bins, frames).
    """
    return np.abs(stft_matrix) ** power


def compute_log_spectrogram(
    spectrogram: np.ndarray,
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Convert a power spectrogram to decibel (log) scale.

    dB = 10 * log10(S / ref)

    The decibel scale is more perceptually meaningful because human
    hearing responds logarithmically to intensity.

    Args:
        spectrogram: Power spectrogram (linear scale).
        ref: Reference value for dB computation.
        amin: Minimum value to clamp to (prevents log(0)).
        top_db: Maximum dynamic range in dB below the peak.

    Returns:
        Log-scaled spectrogram in dB.
    """
    magnitude = np.maximum(spectrogram, amin)
    log_spec = 10.0 * np.log10(magnitude / ref)

    # Clamp dynamic range
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


# ---------------------------------------------------------------------------
# Spectral Features
# ---------------------------------------------------------------------------

def compute_rms(
    signal: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute the Root Mean Square (RMS) energy per frame.

    RMS measures the average signal power within each frame:
        RMS(m) = sqrt( (1/N) * Σ x[n]² )

    RMS is useful for detecting voiced vs. unvoiced segments,
    silence detection, and loudness estimation.

    Args:
        signal: 1D audio signal.
        frame_length: Number of samples per frame.
        hop_length: Number of samples between frames.

    Returns:
        RMS values per frame, shape (num_frames,).
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)

    # Pad signal
    pad_length = frame_length // 2
    signal_padded = np.pad(signal, (pad_length, pad_length), mode="reflect")

    num_frames = 1 + (len(signal_padded) - frame_length) // hop_length
    rms = np.zeros(num_frames)

    for m in range(num_frames):
        start = m * hop_length
        frame = signal_padded[start : start + frame_length]
        rms[m] = np.sqrt(np.mean(frame ** 2))

    return rms


def compute_spectral_centroid(
    stft_magnitude: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Compute the spectral centroid for each frame.

    The spectral centroid is the weighted mean of the frequencies,
    where the weights are the magnitudes:
        SC(m) = Σ(f[k] * |X(m,k)|) / Σ(|X(m,k)|)

    It indicates the "brightness" of a sound — higher centroid
    means more high-frequency energy. Vocals typically have a
    centroid in the 1–4 kHz range; cymbals are much higher.

    Args:
        stft_magnitude: Magnitude of STFT, shape (freq_bins, frames).
        sample_rate: Sampling rate in Hz.
        n_fft: FFT size used to compute the STFT.

    Returns:
        Spectral centroid per frame in Hz, shape (num_frames,).
    """
    n_freq_bins = stft_magnitude.shape[0]
    frequencies = np.linspace(0, sample_rate / 2, n_freq_bins)

    # Weighted mean: Σ(f * mag) / Σ(mag)
    magnitude_sum = np.sum(stft_magnitude, axis=0)
    magnitude_sum = np.maximum(magnitude_sum, 1e-10)  # Avoid division by zero

    centroid = np.sum(frequencies[:, np.newaxis] * stft_magnitude, axis=0) / magnitude_sum
    return centroid


def compute_spectral_bandwidth(
    stft_magnitude: np.ndarray,
    sample_rate: int,
    centroid: np.ndarray,
    n_fft: int = 2048,
    p: float = 2.0,
) -> np.ndarray:
    """
    Compute the spectral bandwidth for each frame.

    Spectral bandwidth measures the spread of the spectrum around
    the centroid (like a standard deviation in the frequency domain):
        BW(m) = ( Σ(|f[k] - SC(m)|^p * |X(m,k)|) / Σ(|X(m,k)|) )^(1/p)

    Wider bandwidth indicates a richer harmonic content.

    Args:
        stft_magnitude: Magnitude of STFT, shape (freq_bins, frames).
        sample_rate: Sampling rate in Hz.
        centroid: Spectral centroid per frame.
        n_fft: FFT size used.
        p: Order of the bandwidth (2.0 for standard deviation).

    Returns:
        Spectral bandwidth per frame in Hz, shape (num_frames,).
    """
    n_freq_bins = stft_magnitude.shape[0]
    frequencies = np.linspace(0, sample_rate / 2, n_freq_bins)

    magnitude_sum = np.sum(stft_magnitude, axis=0)
    magnitude_sum = np.maximum(magnitude_sum, 1e-10)

    deviation = np.abs(frequencies[:, np.newaxis] - centroid[np.newaxis, :]) ** p
    weighted_deviation = np.sum(deviation * stft_magnitude, axis=0) / magnitude_sum
    bandwidth = weighted_deviation ** (1.0 / p)

    return bandwidth


def compute_spectral_rolloff(
    stft_magnitude: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    roll_percent: float = 0.85,
) -> np.ndarray:
    """
    Compute the spectral rolloff frequency for each frame.

    The rolloff frequency is the frequency below which a specified
    percentage of the total spectral energy is contained:
        Σ(|X(m,k)|² for k <= k_rolloff) = roll_percent * Σ(|X(m,k)|²)

    An 85% rolloff is commonly used. It helps distinguish harmonic
    sounds (lower rolloff) from noisy sounds (higher rolloff).

    Args:
        stft_magnitude: Magnitude of STFT, shape (freq_bins, frames).
        sample_rate: Sampling rate in Hz.
        n_fft: FFT size used.
        roll_percent: Percentage of energy threshold (0.0 to 1.0).

    Returns:
        Rolloff frequency per frame in Hz, shape (num_frames,).
    """
    n_freq_bins = stft_magnitude.shape[0]
    frequencies = np.linspace(0, sample_rate / 2, n_freq_bins)

    power = stft_magnitude ** 2
    total_energy = np.sum(power, axis=0, keepdims=True)
    total_energy = np.maximum(total_energy, 1e-10)

    cumulative_energy = np.cumsum(power, axis=0) / total_energy

    rolloff = np.zeros(stft_magnitude.shape[1])
    for m in range(stft_magnitude.shape[1]):
        # Find the bin where cumulative energy exceeds the threshold
        idx = np.searchsorted(cumulative_energy[:, m], roll_percent)
        idx = min(idx, n_freq_bins - 1)
        rolloff[m] = frequencies[idx]

    return rolloff


def compute_zero_crossing_rate(
    signal: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute the Zero-Crossing Rate (ZCR) per frame.

    ZCR counts the number of times the signal changes sign within
    each frame, normalized by the frame length:
        ZCR(m) = (1/N) * Σ |sign(x[n]) - sign(x[n-1])| / 2

    High ZCR indicates noisy or unvoiced segments (like 's', 'f' sounds).
    Low ZCR indicates tonal or voiced segments (like vowels).

    Args:
        signal: 1D audio signal.
        frame_length: Number of samples per frame.
        hop_length: Number of samples between frames.

    Returns:
        ZCR values per frame, shape (num_frames,).
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)

    pad_length = frame_length // 2
    signal_padded = np.pad(signal, (pad_length, pad_length), mode="reflect")

    num_frames = 1 + (len(signal_padded) - frame_length) // hop_length
    zcr = np.zeros(num_frames)

    for m in range(num_frames):
        start = m * hop_length
        frame = signal_padded[start : start + frame_length]
        # Count sign changes
        signs = np.sign(frame)
        sign_changes = np.abs(np.diff(signs))
        zcr[m] = np.sum(sign_changes > 0) / frame_length

    return zcr


# ---------------------------------------------------------------------------
# Convenience: Extract All Features
# ---------------------------------------------------------------------------

def extract_audio_features(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, np.ndarray]:
    """
    Extract a comprehensive set of spectral features from an audio signal.

    This function computes the STFT and derives multiple features,
    providing a multi-dimensional view of the audio's characteristics.

    Args:
        signal: 1D audio signal (mono).
        sample_rate: Sampling rate in Hz.
        n_fft: FFT size.
        hop_length: Hop length in samples.

    Returns:
        Dictionary with keys:
        - 'stft': Complex STFT matrix
        - 'spectrogram': Power spectrogram
        - 'log_spectrogram': Log-scaled spectrogram (dB)
        - 'rms': RMS energy per frame
        - 'spectral_centroid': Centroid per frame (Hz)
        - 'spectral_bandwidth': Bandwidth per frame (Hz)
        - 'spectral_rolloff': Rolloff frequency per frame (Hz)
        - 'zero_crossing_rate': ZCR per frame
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)

    # Core transform
    stft = compute_stft(signal, sample_rate, n_fft=n_fft, hop_length=hop_length)
    stft_mag = np.abs(stft)
    spec = compute_spectrogram(stft, power=2.0)
    log_spec = compute_log_spectrogram(spec)

    # Spectral features
    centroid = compute_spectral_centroid(stft_mag, sample_rate, n_fft=n_fft)
    bandwidth = compute_spectral_bandwidth(stft_mag, sample_rate, centroid, n_fft=n_fft)
    rolloff = compute_spectral_rolloff(stft_mag, sample_rate, n_fft=n_fft)

    # Time-domain features
    rms = compute_rms(signal, frame_length=n_fft, hop_length=hop_length)
    zcr = compute_zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop_length)

    # Align lengths (features may differ by 1 frame due to padding)
    min_frames = min(len(rms), len(centroid), len(zcr), stft.shape[1])

    return {
        "stft": stft[:, :min_frames],
        "spectrogram": spec[:, :min_frames],
        "log_spectrogram": log_spec[:, :min_frames],
        "rms": rms[:min_frames],
        "spectral_centroid": centroid[:min_frames],
        "spectral_bandwidth": bandwidth[:min_frames],
        "spectral_rolloff": rolloff[:min_frames],
        "zero_crossing_rate": zcr[:min_frames],
    }
