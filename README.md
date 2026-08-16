# 🎤 Vocal & Music Separator

**Separate vocals and instrumental tracks from any song using AI-powered source separation and digital signal processing.**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://musicvocals.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌐 Live Demo

Try the interactive web application live on Streamlit Cloud:

👉 **[Launch App: musicvocals.streamlit.app](https://musicvocals.streamlit.app/)**

---

## 🎯 Project Overview

This application allows you to upload any song and separate it into:

1. **Vocals / Lyrics** — The singing voice
2. **Instrumental / Music** — Everything else (drums, bass, guitar, keys, etc.)

It combines two complementary technologies:

- **AI Source Separation** — A pretrained deep learning model (Demucs v4) that has learned to identify and separate vocal patterns from instrumental patterns
- **Digital Signal Processing** — FFT, STFT, and spectral analysis to visualize and understand the audio content

---

## 🔧 How It Works

```
Audio File (MP3, WAV, FLAC, M4A)
         ↓
   File Validation
   (format, size, integrity)
         ↓
   Audio Decoding
   (via librosa + FFmpeg)
         ↓
   Preprocessing
   (resampling to 44.1kHz, channel normalization)
         ↓
   Demucs v4 Hybrid Transformer
   (neural source separation)
         ↓
  ┌──────────────┐
  │              │
Vocals      Drums + Bass + Other
  │              │
  │         Recombined as
  │         "Instrumental"
  └──────┬───────┘
         ↓
   DSP Analysis
   (FFT, STFT, spectral features)
         ↓
   Visualization
   (waveforms, spectrograms)
         ↓
   Audio Players + Download
```

---

## 📊 Signal Processing Concepts

### Sampling Rate

Audio is a continuous pressure wave. To store it digitally, we **sample** the wave at regular intervals. The **sampling rate** (e.g., 44,100 Hz) determines how many measurements per second. By the Nyquist theorem, a 44.1 kHz sample rate can represent frequencies up to 22,050 Hz — covering the full range of human hearing.

### Waveform

A waveform displays **amplitude over time**. It's the most direct representation of audio — you can see loud vs. quiet sections, beats, and pauses. However, it tells you nothing about which frequencies are present.

### Fast Fourier Transform (FFT)

The FFT decomposes a signal into its constituent **sinusoidal frequencies**:

```
x[n] → FFT → X[k] = Σ x[n] · e^(-j2πkn/N)
```

Given a recording, the FFT tells you the amplitude of every frequency present. A 440 Hz sine wave produces a single spike at 440 Hz; a chord produces spikes at multiple frequencies.

### Short-Time Fourier Transform (STFT)

A single FFT of an entire song gives you the overall frequency content but loses all **temporal information** — you can't tell when a note was played.

The STFT solves this by computing the FFT over short, overlapping windows:

```
Signal:  [================================]
Frame 1: [====]
Frame 2:    [====]
Frame 3:       [====]
              ...
```

Each frame (typically 2048 samples ≈ 46 ms) produces one column of the spectrogram.

**Trade-off**: Shorter windows → better time resolution, worse frequency resolution (and vice versa). This is the **time-frequency uncertainty principle**.

### Spectrogram

The spectrogram is a 2D visualization of the STFT:
- **X-axis**: Time
- **Y-axis**: Frequency
- **Color**: Magnitude (how loud that frequency is at that moment)

Spectrograms reveal patterns invisible in waveforms: vocal formants, harmonic series, percussive transients, and more.

### Why FFT Filtering Cannot Separate Vocals

You might wonder: "Can't we just filter out the vocal frequencies?" Unfortunately, no:

1. **Overlapping frequency ranges**: Human vocals span approximately 80 Hz to 8,000 Hz. Guitars, keyboards, and most other instruments overlap this range almost entirely.

2. **Harmonics**: A singer's note at 220 Hz produces harmonics at 440, 660, 880, 1100 Hz... — the exact same frequencies where instruments play. An FFT filter cannot distinguish a vocal harmonic from a guitar harmonic at the same frequency.

3. **Phase matters**: Two signals at the same frequency can reinforce or cancel depending on their phase relationship. Simple magnitude filtering destroys phase, creating audible artifacts.

4. **Non-stationarity**: Vocal characteristics change rapidly — vowels, consonants, vibrato, pitch slides. A static filter cannot track these changes.

5. **Stereo information**: While center-panned vocal cancellation works for simple mixes, modern productions use reverb, delay, and dynamic panning that spread vocals across the stereo field.

### How Neural Source Separation Works

Models like Demucs learn **statistical priors** about what vocals and instruments "sound like" from thousands of professionally mixed songs where individual stems are available.

The model operates on the spectrogram and learns to predict **masks** — values between 0 and 1 for each time-frequency bin — indicating how much of that bin belongs to each source. This is fundamentally **pattern recognition**, not filtering.

The Demucs v4 **Hybrid Transformer** architecture processes audio in both time and frequency domains simultaneously, achieving state-of-the-art separation quality.

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+** (tested on 3.14)
- **FFmpeg** (required for MP3/M4A support)

#### Install FFmpeg

macOS:
```bash
brew install ffmpeg
```

Ubuntu/Debian:
```bash
sudo apt install ffmpeg
```

Windows:
```bash
choco install ffmpeg
```

### Project Setup

```bash
# Clone the repository
git clone https://github.com/ShadmanSShuvo/MusicVocals.git
cd MusicVocals

# Create virtual environment
python -m venv .venv

# Activate it
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

### First Run

The first time you click "Separate," the Demucs model (~80 MB) is downloaded automatically. Subsequent runs use the cached model.

---

## 🧠 Model: Demucs v4 (htdemucs)

| Property | Value |
|----------|-------|
| **Model** | htdemucs |
| **Architecture** | Hybrid Transformer (U-Net + Transformer encoder) |
| **Training data** | 800+ songs with isolated stems |
| **Output stems** | Vocals, Drums, Bass, Other |
| **Native sample rate** | 44.1 kHz |
| **Model size** | ~80 MB |

Demucs v4 processes audio in **both time and frequency domains** simultaneously:
- The **time branch** captures fine temporal details (transients, attacks)
- The **frequency branch** resolves harmonic structure and pitched content
- A **Transformer encoder** models long-range dependencies between sources

For this application, the 4 stems are recombined as:
- **Vocals** = vocals stem
- **Instrumental** = drums + bass + other

---

## 💻 Hardware

### CPU (Default)

The application works on any modern CPU. Processing a 4-minute song takes approximately:
- Apple M2: ~60–90 seconds
- Modern x86: ~90–150 seconds

### GPU Acceleration

If available, the application automatically uses:
- **CUDA** (NVIDIA GPUs): ~10–20 seconds per song
- **MPS** (Apple Silicon): ~30–50 seconds per song

The selected device is displayed in the sidebar.

### Memory

- **Minimum**: 4 GB RAM
- **Recommended**: 8 GB+ RAM
- Songs longer than 5 minutes may require more memory

---

## 📁 Project Structure

```
MusicVocals/
├── app.py                    # Streamlit UI (main entry point)
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── audio.py              # Audio I/O, validation, preprocessing
│   ├── separator.py          # Source separation (Demucs wrapper)
│   ├── dsp.py                # FFT, STFT, spectral features
│   ├── visualization.py      # Matplotlib plotting functions
│   └── utils.py              # Config, helpers, file utilities
├── outputs/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_audio.py          # Audio loading/saving tests
    ├── test_dsp.py            # FFT/STFT/feature tests
    └── test_separator.py      # Separator initialization tests
```

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_dsp.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

Tests use **synthetic signals** (sine waves, white noise) — no audio files needed.

---

## ⚠️ Known Limitations

### Separation Quality

- **Vocal bleed**: Some vocal residue may appear in the instrumental track, especially for heavily processed vocals (auto-tune, heavy reverb)
- **Instrument bleed**: Instruments with vocal-like characteristics (violin, saxophone) may partially appear in the vocal track
- **Background vocals**: Backing vocals and harmonies are generally included in the vocal stem, but may be partially split
- **Reverb tails**: Reverb applied to vocals in the original mix is difficult to separate cleanly
- **Dense mixes**: Very dense arrangements with many layered instruments may have lower separation quality

### Input Quality

- **Highly compressed audio** (low-bitrate MP3): Compression artifacts can degrade separation quality
- **Live recordings**: Audience noise and room acoustics make separation harder
- **Phone recordings**: Low quality, mono recordings give worse results than studio masters

### Technical

- **Processing time**: CPU processing can be slow for long songs (>5 minutes)
- **Memory**: 8 GB RAM is recommended; very long songs may cause out-of-memory on constrained systems
- **File size limit**: Default 50 MB maximum upload size

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Meta Research — Demucs](https://github.com/facebookresearch/demucs) for the source separation model
- [Streamlit](https://streamlit.io/) for the web framework
- [librosa](https://librosa.org/) for audio analysis utilities
- [PyTorch](https://pytorch.org/) for the deep learning backend
