"""
Visualization module for audio analysis.

Provides reusable Matplotlib plotting functions for:
- Waveform display (time-domain)
- Spectrogram display (time-frequency domain)
- Frequency spectrum (FFT magnitude)
- Spectral feature plots (centroid, bandwidth, rolloff, ZCR, RMS)

All plots use a dark theme compatible with Streamlit's default appearance
and are optimized for readability at typical web display sizes.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Use non-interactive backend for Streamlit
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Theme & Style
# ---------------------------------------------------------------------------

# Color palette
COLORS = {
    "original": "#58A6FF",    # Blue
    "vocals": "#F78166",      # Coral/Orange
    "instrumental": "#7EE787", # Green
    "accent": "#D2A8FF",       # Purple
    "grid": "#30363D",         # Subtle grid
    "background": "#0D1117",   # Dark background
    "text": "#E6EDF3",         # Light text
    "muted": "#8B949E",        # Muted text
}

SPECTROGRAM_CMAP = "magma"  # Perceptually uniform colormap


def _apply_dark_theme(fig: Figure, ax: plt.Axes) -> None:
    """Apply consistent dark theme to a figure and axes."""
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, alpha=0.15, color=COLORS["grid"])


# ---------------------------------------------------------------------------
# Waveform Plot
# ---------------------------------------------------------------------------

def plot_waveform(
    signal: np.ndarray,
    sample_rate: int,
    title: str = "Waveform",
    color: str | None = None,
    figsize: tuple[float, float] = (12, 3),
) -> Figure:
    """
    Plot the time-domain waveform of an audio signal.

    Displays amplitude (vertical) over time (horizontal). For stereo
    signals, averages channels to mono for display.

    Args:
        signal: Audio array. Shape (samples,) or (channels, samples).
        sample_rate: Sample rate in Hz.
        title: Plot title.
        color: Line color. Defaults to blue.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    if color is None:
        color = COLORS["original"]

    # Convert to mono for display
    if signal.ndim > 1:
        signal_mono = np.mean(signal, axis=0)
    else:
        signal_mono = signal

    # Create time axis
    duration = len(signal_mono) / sample_rate
    time = np.linspace(0, duration, len(signal_mono))

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_theme(fig, ax)

    # Downsample for display if signal is long
    # This keeps matplotlib rendering snappy and responsive
    max_display_points = 50_000
    if len(signal_mono) > max_display_points:
        step = max(1, len(signal_mono) // max_display_points)
        time = time[::step]
        signal_mono = signal_mono[::step]

    ax.plot(time, signal_mono, color=color, linewidth=0.3, alpha=0.85)
    ax.fill_between(time, signal_mono, alpha=0.15, color=color)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.05, 1.05)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectrogram Plot
# ---------------------------------------------------------------------------

def plot_spectrogram(
    spectrogram_db: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    title: str = "Spectrogram",
    figsize: tuple[float, float] = (12, 4),
    max_freq: float | None = None,
) -> Figure:
    """
    Plot a log-frequency spectrogram.

    Displays frequency (vertical, log scale) over time (horizontal)
    with magnitude encoded as color intensity.

    The spectrogram is the primary visualization for understanding
    the time-varying frequency content of audio. Source separation
    models internally operate on similar representations.

    Args:
        spectrogram_db: Log-scaled spectrogram in dB.
                        Shape (freq_bins, time_frames).
        sample_rate: Sample rate in Hz.
        hop_length: STFT hop length in samples (for time axis).
        title: Plot title.
        figsize: Figure size.
        max_freq: Maximum frequency to display (Hz). None = Nyquist.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_theme(fig, ax)

    n_freq_bins, n_frames = spectrogram_db.shape

    # Time axis
    duration = n_frames * hop_length / sample_rate
    time_extent = [0, duration]

    # Frequency axis
    freq_max = sample_rate / 2
    freq_extent = [0, freq_max]

    if max_freq is not None:
        freq_extent[1] = min(max_freq, freq_max)

    img = ax.imshow(
        spectrogram_db,
        aspect="auto",
        origin="lower",
        cmap=SPECTROGRAM_CMAP,
        extent=[time_extent[0], time_extent[1], freq_extent[0], freq_extent[1]],
        interpolation="bilinear",
    )

    # Use log scale for frequency axis (more perceptually meaningful)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylim(20, freq_extent[1])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.tick_params(colors=COLORS["muted"], labelsize=8)
    cbar.set_label("Magnitude (dB)", color=COLORS["text"], fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Frequency Spectrum Plot
# ---------------------------------------------------------------------------

def plot_frequency_spectrum(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    title: str = "Frequency Spectrum",
    color: str | None = None,
    figsize: tuple[float, float] = (12, 3.5),
    max_freq: float = 20000,
) -> Figure:
    """
    Plot the single-sided frequency spectrum (FFT magnitude).

    Displays magnitude (vertical) as a function of frequency (horizontal).
    Uses logarithmic frequency axis for perceptual relevance.

    Args:
        frequencies: Frequency values in Hz.
        magnitudes: FFT magnitude values.
        title: Plot title.
        color: Line color.
        figsize: Figure size.
        max_freq: Maximum frequency to display in Hz.

    Returns:
        Matplotlib Figure object.
    """
    if color is None:
        color = COLORS["accent"]

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_theme(fig, ax)

    # Filter to display range
    mask = (frequencies > 0) & (frequencies <= max_freq)

    ax.semilogx(frequencies[mask], magnitudes[mask], color=color, linewidth=0.6, alpha=0.9)
    ax.fill_between(frequencies[mask], magnitudes[mask], alpha=0.1, color=color)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectral Features Plot
# ---------------------------------------------------------------------------

def plot_audio_features(
    features: dict[str, np.ndarray],
    sample_rate: int,
    hop_length: int = 512,
    title: str = "Spectral Features",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """
    Plot multiple spectral features in a multi-panel figure.

    Displays RMS energy, spectral centroid, spectral bandwidth,
    spectral rolloff, and zero-crossing rate over time.

    These features characterize different aspects of the audio:
    - RMS: Loudness/energy
    - Centroid: Brightness (tonal quality)
    - Bandwidth: Spectral spread (harmonic richness)
    - Rolloff: High-frequency energy boundary
    - ZCR: Noisiness vs. tonality

    Args:
        features: Dict with keys like 'rms', 'spectral_centroid', etc.
        sample_rate: Sample rate in Hz.
        hop_length: Hop length in samples.
        title: Super title for the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    feature_configs = [
        ("rms", "RMS Energy", COLORS["vocals"], "Energy"),
        ("spectral_centroid", "Spectral Centroid", COLORS["original"], "Hz"),
        ("spectral_bandwidth", "Spectral Bandwidth", COLORS["instrumental"], "Hz"),
        ("spectral_rolloff", "Spectral Rolloff", COLORS["accent"], "Hz"),
        ("zero_crossing_rate", "Zero-Crossing Rate", "#FFA657", "Rate"),
    ]

    # Filter to available features
    available = [(k, l, c, u) for k, l, c, u in feature_configs if k in features]
    n_plots = len(available)

    if n_plots == 0:
        fig, ax = plt.subplots(figsize=(12, 2))
        _apply_dark_theme(fig, ax)
        ax.text(0.5, 0.5, "No features to display", ha="center", va="center",
                color=COLORS["muted"], fontsize=12, transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    fig.patch.set_facecolor(COLORS["background"])

    for i, (key, label, color, unit) in enumerate(available):
        ax = axes[i]
        _apply_dark_theme(fig, ax)

        data = features[key]
        time = np.arange(len(data)) * hop_length / sample_rate

        ax.plot(time, data, color=color, linewidth=0.8, alpha=0.9)
        ax.fill_between(time, data, alpha=0.1, color=color)
        ax.set_ylabel(f"{unit}", fontsize=9, color=COLORS["muted"])
        ax.set_title(label, fontsize=10, fontweight="bold", pad=5, loc="left")

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(title, fontsize=14, fontweight="bold",
                 color=COLORS["text"], y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparison Waveforms
# ---------------------------------------------------------------------------

def plot_comparison_waveforms(
    signals: dict[str, np.ndarray],
    sample_rate: int,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """
    Plot waveforms for multiple audio signals stacked vertically.

    Useful for comparing original, vocals, and instrumental side by side.

    Args:
        signals: Dict mapping label to audio array.
                 e.g., {"Original": array, "Vocals": array, "Instrumental": array}
        sample_rate: Sample rate in Hz.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    color_map = {
        "Original": COLORS["original"],
        "Vocals": COLORS["vocals"],
        "Instrumental": COLORS["instrumental"],
    }

    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    fig.patch.set_facecolor(COLORS["background"])

    for i, (label, signal) in enumerate(signals.items()):
        ax = axes[i]
        _apply_dark_theme(fig, ax)

        if signal.ndim > 1:
            signal = np.mean(signal, axis=0)

        duration = len(signal) / sample_rate
        time = np.linspace(0, duration, len(signal))

        # Downsample for display
        max_display = 500_000
        if len(signal) > max_display:
            step = len(signal) // max_display
            time = time[::step]
            signal = signal[::step]

        color = color_map.get(label, COLORS["accent"])
        ax.plot(time, signal, color=color, linewidth=0.3, alpha=0.85)
        ax.fill_between(time, signal, alpha=0.1, color=color)
        ax.set_ylabel("Amplitude", fontsize=9, color=COLORS["muted"])
        ax.set_title(label, fontsize=11, fontweight="bold", pad=5, loc="left",
                     color=color)
        ax.set_ylim(-1.05, 1.05)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
