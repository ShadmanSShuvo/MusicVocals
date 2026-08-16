"""
Vocal & Music Separator — Streamlit Application

Separate vocals and instrumental tracks from any song using AI-powered
source separation (Demucs v4 Hybrid Transformer) combined with digital
signal processing analysis.

Usage:
    streamlit run app.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure standard binary locations (e.g. /opt/homebrew/bin for ffmpeg) are in PATH
for p in ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin", "/usr/bin", "/bin"]:
    if p not in os.environ.get("PATH", "") and os.path.exists(p):
        os.environ["PATH"] = f"{p}:{os.environ.get('PATH', '')}"

# Ensure src package is importable
sys.path.insert(0, str(Path(__file__).parent))


from src.audio import (
    AudioInfo,
    AudioValidationError,
    audio_to_bytes,
    ensure_mono,
    get_audio_info,
    load_audio,
    save_audio,
    save_uploaded_file,
    validate_audio_duration,
    validate_uploaded_file,
)
from src.dsp import (
    compute_fft,
    compute_log_spectrogram,
    compute_spectrogram,
    compute_stft,
    extract_audio_features,
)
from src.separator import DemucsSeparator, select_device
from src.utils import (
    LARGE_FILE_WARNING_SECONDS,
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    cleanup_temp_dir,
    create_temp_dir,
    format_duration,
    format_file_size,
)
from src.visualization import (
    plot_audio_features,
    plot_comparison_waveforms,
    plot_frequency_spectrum,
    plot_spectrogram,
    plot_waveform,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vocal & Music Separator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #58A6FF, #D2A8FF, #F78166);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .main-header .subtitle {
        color: #8B949E;
        font-size: 1.1rem;
        margin-top: 0;
    }

    /* Section dividers */
    .section-header {
        border-bottom: 1px solid #30363D;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Result cards */
    .result-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }

    /* Info metrics */
    .stMetric > div {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 0.8rem;
    }

    /* Hide Streamlit footer */
    footer {visibility: hidden;}

    /* Improve button styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Caching
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_separator() -> DemucsSeparator:
    """
    Load and cache the Demucs separator model.

    Uses Streamlit's @cache_resource so the model is loaded once
    and reused across all sessions and reruns.
    """
    separator = DemucsSeparator(model_name="htdemucs")
    separator.load_model()
    return separator


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict[str, Any]:
    """Render the sidebar with app info and separation quality settings."""
    with st.sidebar:
        st.markdown("## 🎵 Vocal & Music Separator")
        st.markdown("---")

        st.markdown("### 🎚️ Separation Settings")
        quality_choice = st.selectbox(
            "Quality Profile",
            [
                "✨ High Quality (Recommended)",
                "🚀 Ultra Quality (Studio - Slower)",
                "⚡ Fast (Quick Preview)",
            ],
            index=0,
            help=(
                "High Quality uses test-time shift augmentation and 50% chunk overlap "
                "to eliminate dropped beats and boundary artifacts."
            ),
        )

        if "Ultra Quality" in quality_choice:
            shifts = 2
            overlap = 0.5
        elif "Fast" in quality_choice:
            shifts = 0
            overlap = 0.25
        else:
            shifts = 1
            overlap = 0.5

        default_mode = st.radio(
            "Default Instrumental Mode",
            [
                "Residual (Mix - Vocals) [Recommended]",
                "Additive (Drums + Bass + Other)",
            ],
            index=0,
            help=(
                "Residual Mode preserves 100% of instrumental fullness, reverbs, and subtle acoustic elements. "
                "Additive Mode sums the isolated Demucs stems."
            ),
        )
        mode_str = "residual" if "Residual" in default_mode else "additive"

        st.markdown("---")

        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. **Upload** a song (MP3, WAV, FLAC, M4A)
        2. **Preview** the original audio
        3. **Click** "Separate Vocals & Music"
        4. **Listen** to separated tracks & stems
        5. **Download** the results
        """)

        st.markdown("---")

        st.markdown("### 🎛️ Supported Formats")
        formats = ", ".join(f"`{ext}`" for ext in sorted(SUPPORTED_EXTENSIONS))
        st.markdown(formats)
        st.caption(f"Max file size: {MAX_FILE_SIZE_MB} MB")

        st.markdown("---")

        st.markdown("### 🧠 Model Info")
        device = select_device()
        device_emoji = {"cuda": "🟢", "mps": "🟡", "cpu": "🔵"}.get(str(device), "⚪")
        st.markdown(f"""
        - **Model**: Demucs v4 (htdemucs)
        - **Architecture**: Hybrid Transformer
        - **Stems**: Vocals, Drums, Bass, Other
        - **Sample Rate**: 44.1 kHz
        - {device_emoji} **Device**: `{device}`
        """)

        st.markdown("---")

        st.markdown("### 📊 Signal Processing")
        st.markdown("""
        The app demonstrates DSP concepts:
        - **FFT**: Frequency decomposition
        - **STFT**: Time-frequency analysis
        - **Spectrograms**: Visual frequency maps
        - **Spectral features**: Audio characteristics
        """)

        st.markdown("---")
        st.caption("Built with Streamlit + PyTorch + Demucs by Shuvo")

    return {
        "shifts": shifts,
        "overlap": overlap,
        "mode": mode_str,
    }


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application."""

    settings = render_sidebar()

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🎤 Vocal & Music Separator</h1>
        <p class="subtitle">Separate vocals and instrumental tracks using AI + digital signal processing</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Step 1: Upload ---
    st.markdown('<div class="section-header"><h3>📁 Step 1 — Upload Your Song</h3></div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        help=f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. Max {MAX_FILE_SIZE_MB} MB.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("👆 Upload a song to get started. Drag & drop or click to browse.")
        _render_dsp_explainer()
        return

    # --- Validate ---
    try:
        validate_uploaded_file(uploaded_file, uploaded_file.name)
    except AudioValidationError as e:
        st.error(f"❌ {e}")
        return

    # --- Save to temp and get info ---
    temp_dir = create_temp_dir()

    try:
        file_path = save_uploaded_file(uploaded_file, uploaded_file.name, temp_dir)
    except Exception as e:
        st.error(f"❌ Failed to save uploaded file: {e}")
        cleanup_temp_dir(temp_dir)
        return

    try:
        audio_info = get_audio_info(file_path, uploaded_file.name)
    except AudioValidationError as e:
        st.error(f"❌ {e}")
        cleanup_temp_dir(temp_dir)
        return

    # Validate duration
    try:
        validate_audio_duration(audio_info.duration)
    except AudioValidationError as e:
        st.error(f"❌ {e}")
        cleanup_temp_dir(temp_dir)
        return

    # --- Display audio info ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎵 Song", audio_info.filename[:30])
    with col2:
        st.metric("⏱️ Duration", audio_info.duration_formatted)
    with col3:
        st.metric("📻 Sample Rate", audio_info.sample_rate_khz)
    with col4:
        st.metric("📦 Size", audio_info.file_size_formatted)

    # Large file warning
    if audio_info.duration > LARGE_FILE_WARNING_SECONDS:
        st.warning(
            f"⚠️ This song is {audio_info.duration_formatted} long. "
            "Processing may take several minutes and use significant memory."
        )

    # --- Step 2: Original Audio ---
    st.markdown('<div class="section-header"><h3>🎧 Step 2 — Original Audio</h3></div>',
                unsafe_allow_html=True)

    # Audio player
    uploaded_file.seek(0)
    st.audio(uploaded_file.read(), format=f"audio/{file_path.suffix.lstrip('.')}")

    # Load audio for visualization
    try:
        audio_mono, sr = load_audio(file_path, mono=True)
    except AudioValidationError as e:
        st.error(f"❌ {e}")
        cleanup_temp_dir(temp_dir)
        return

    # Original waveform
    with st.expander("📈 Original Waveform", expanded=True):
        fig = plot_waveform(audio_mono, sr, title="Original Waveform")
        st.pyplot(fig)
        plt_close(fig)

    # --- Step 3: Separation ---
    st.markdown('<div class="section-header"><h3>🔬 Step 3 — Analyze & Separate</h3></div>',
                unsafe_allow_html=True)

    # Check if results already exist in session state
    results_key = f"results_{uploaded_file.name}_{uploaded_file.size}_{settings['shifts']}_{settings['overlap']}"

    if results_key in st.session_state:
        _render_results(st.session_state[results_key], sr, settings)
        return

    # Separation button
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        separate_clicked = st.button(
            "🎤 Separate Vocals & Music",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            f"Settings: **Shifts={settings['shifts']}**, **Overlap={settings['overlap']}**, "
            f"Default Instrumental: **{settings['mode'].title()} Mode**."
        )

    if not separate_clicked:
        return

    # --- Run Separation Pipeline ---
    progress = st.progress(0, text="Initializing...")
    status = st.status("🔄 Processing your song...", expanded=True)

    try:
        # Step 1: Load model
        with status:
            st.write("📥 Loading AI model...")
        progress.progress(10, text="Loading AI model...")

        try:
            separator = get_separator()
        except Exception as e:
            st.error(f"❌ Failed to load the separation model: {e}")
            progress.empty()
            cleanup_temp_dir(temp_dir)
            return

        progress.progress(30, text="Model loaded. Loading audio...")

        # Step 2: Load audio for separation (stereo, full quality)
        with status:
            st.write("🎵 Preparing audio for separation...")

        try:
            audio_stereo, sr_audio = load_audio(file_path, target_sr=44100, mono=False)
        except AudioValidationError as e:
            st.error(f"❌ {e}")
            progress.empty()
            cleanup_temp_dir(temp_dir)
            return

        progress.progress(40, text="Running source separation...")

        # Step 3: Separate
        with status:
            st.write(f"🧠 Running Demucs on {separator.get_device_name()} (shifts={settings['shifts']}, overlap={settings['overlap']})...")
            st.write("Extracting high-fidelity vocal and instrumental stems...")

        start_time = time.time()

        try:
            stems = separator.separate(
                audio_stereo,
                sr_audio,
                shifts=settings["shifts"],
                overlap=settings["overlap"],
                instrumental_mode=settings["mode"],
            )
        except RuntimeError as e:
            st.error(f"❌ Separation failed: {e}")
            progress.empty()
            cleanup_temp_dir(temp_dir)
            return

        elapsed = time.time() - start_time
        progress.progress(70, text="Separation complete! Generating analysis...")

        with status:
            st.write(f"✅ Separation completed in {elapsed:.1f}s")

        # Step 4: Post-process stems
        vocals = stems["vocals"]
        instr_residual = stems["instrumental_residual"]
        instr_additive = stems["instrumental_additive"]

        progress.progress(80, text="Computing DSP analysis...")

        # Step 5: Compute DSP features
        with status:
            st.write("📊 Computing spectral analysis...")

        vocals_mono = ensure_mono(vocals)
        instr_res_mono = ensure_mono(instr_residual)
        instr_add_mono = ensure_mono(instr_additive)

        # STFT and spectrograms
        n_fft = 2048
        hop_length = 512

        vocals_stft = compute_stft(vocals_mono, 44100, n_fft=n_fft, hop_length=hop_length)
        instr_res_stft = compute_stft(instr_res_mono, 44100, n_fft=n_fft, hop_length=hop_length)
        instr_add_stft = compute_stft(instr_add_mono, 44100, n_fft=n_fft, hop_length=hop_length)

        vocals_spec = compute_log_spectrogram(compute_spectrogram(vocals_stft))
        instr_res_spec = compute_log_spectrogram(compute_spectrogram(instr_res_stft))
        instr_add_spec = compute_log_spectrogram(compute_spectrogram(instr_add_stft))

        # Spectral features
        vocals_features = extract_audio_features(vocals_mono, 44100, n_fft=n_fft, hop_length=hop_length)
        instr_res_features = extract_audio_features(instr_res_mono, 44100, n_fft=n_fft, hop_length=hop_length)
        instr_add_features = extract_audio_features(instr_add_mono, 44100, n_fft=n_fft, hop_length=hop_length)

        # FFT
        vocals_freqs, vocals_mags = compute_fft(vocals_mono, 44100)
        instr_res_freqs, instr_res_mags = compute_fft(instr_res_mono, 44100)
        instr_add_freqs, instr_add_mags = compute_fft(instr_add_mono, 44100)

        # Build 4-stems dictionary for stem inspection
        stems_dict_processed = {}
        for stem_name in ["drums", "bass", "other"]:
            if stem_name in stems:
                stem_audio = stems[stem_name]
                stems_dict_processed[stem_name] = {
                    "audio": stem_audio,
                    "mono": ensure_mono(stem_audio),
                    "bytes": audio_to_bytes(stem_audio),
                }

        progress.progress(100, text="Done!")

        with status:
            st.write("🎉 All processing complete!")
        status.update(label="✅ Processing complete!", state="complete")

        # Store results
        results = {
            "vocals": vocals,
            "vocals_mono": vocals_mono,
            "vocals_bytes": audio_to_bytes(vocals),
            "vocals_spec": vocals_spec,
            "vocals_features": vocals_features,
            "vocals_freqs": vocals_freqs,
            "vocals_mags": vocals_mags,
            # Residual Instrumental (Original - Vocals)
            "instr_residual": instr_residual,
            "instr_res_mono": instr_res_mono,
            "instr_res_bytes": audio_to_bytes(instr_residual),
            "instr_res_spec": instr_res_spec,
            "instr_res_features": instr_res_features,
            "instr_res_freqs": instr_res_freqs,
            "instr_res_mags": instr_res_mags,
            # Additive Instrumental (Drums + Bass + Other)
            "instr_additive": instr_additive,
            "instr_add_mono": instr_add_mono,
            "instr_add_bytes": audio_to_bytes(instr_additive),
            "instr_add_spec": instr_add_spec,
            "instr_add_features": instr_add_features,
            "instr_add_freqs": instr_add_freqs,
            "instr_add_mags": instr_add_mags,
            # Stems
            "stems": stems_dict_processed,
            "elapsed": elapsed,
            "sr": 44100,
            "n_fft": n_fft,
            "hop_length": hop_length,
        }

        st.session_state[results_key] = results

    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        logger.exception("Unexpected error during separation")
        return
    finally:
        cleanup_temp_dir(temp_dir)

    # Render results
    _render_results(results, sr, settings)


# ---------------------------------------------------------------------------
# Results Rendering
# ---------------------------------------------------------------------------

def _render_results(results: dict, sr: int, settings: dict) -> None:
    """Render the separation results with audio players, visualizations, and downloads."""

    hop_length = results["hop_length"]

    st.markdown('<div class="section-header"><h3>🎉 Step 4 — Results</h3></div>',
                unsafe_allow_html=True)

    st.success(f"✅ Separation completed in {results['elapsed']:.1f} seconds")

    # --- Mode Selection for Instrumental ---
    st.markdown("#### 🎛️ Instrumental Mode Selection")
    mode_selection = st.radio(
        "Choose how the instrumental track is reconstructed:",
        [
            "🌟 Residual Mode (Original - Vocals) [Recommended — Full Fidelity & Reverb]",
            "🥁 Additive Mode (Sum of Demucs Stems: Drums + Bass + Other)",
        ],
        index=0 if settings.get("mode") == "residual" else 1,
        horizontal=True,
        help=(
            "Residual Mode subtracts the vocals from the original mix, guaranteeing no missed bits or dropouts. "
            "Additive Mode combines the isolated drums, bass, and other instruments."
        ),
    )

    is_residual = "Residual" in mode_selection

    if is_residual:
        instr_mono = results["instr_res_mono"]
        instr_bytes = results["instr_res_bytes"]
        instr_spec = results["instr_res_spec"]
        instr_features = results["instr_res_features"]
        instr_freqs = results["instr_res_freqs"]
        instr_mags = results["instr_res_mags"]
        mode_label = "Residual (Mix - Vocals)"
    else:
        instr_mono = results["instr_add_mono"]
        instr_bytes = results["instr_add_bytes"]
        instr_spec = results["instr_add_spec"]
        instr_features = results["instr_add_features"]
        instr_freqs = results["instr_add_freqs"]
        instr_mags = results["instr_add_mags"]
        mode_label = "Additive (Drums + Bass + Other)"

    # --- Comparison Waveforms ---
    with st.expander("📊 Waveform Comparison", expanded=False):
        fig = plot_comparison_waveforms(
            {
                "Vocals": results["vocals_mono"],
                f"Instrumental ({mode_label})": instr_mono,
            },
            results["sr"],
        )
        st.pyplot(fig)
        plt_close(fig)

    # --- Vocals & Instrumental side by side ---
    col_vocals, col_instr = st.columns(2)

    # --- Vocals Column ---
    with col_vocals:
        st.markdown("### 🎤 Vocals")

        # Audio player
        st.audio(results["vocals_bytes"], format="audio/wav")

        # Download button
        st.download_button(
            label="⬇️ Download Vocals (WAV)",
            data=results["vocals_bytes"],
            file_name="vocals.wav",
            mime="audio/wav",
            use_container_width=True,
        )

        # Waveform
        with st.expander("📈 Vocal Waveform", expanded=True):
            fig = plot_waveform(
                results["vocals_mono"], results["sr"],
                title="Vocals — Waveform", color="#F78166",
            )
            st.pyplot(fig)
            plt_close(fig)

        # Spectrogram
        with st.expander("🌈 Vocal Spectrogram", expanded=True):
            fig = plot_spectrogram(
                results["vocals_spec"], results["sr"],
                hop_length=hop_length,
                title="Vocals — Spectrogram",
            )
            st.pyplot(fig)
            plt_close(fig)

    # --- Instrumental Column ---
    with col_instr:
        st.markdown(f"### 🎸 Instrumental ({mode_label})")

        # Audio player
        st.audio(instr_bytes, format="audio/wav")

        # Download button
        st.download_button(
            label=f"⬇️ Download Instrumental — {mode_label} (WAV)",
            data=instr_bytes,
            file_name=f"instrumental_{'residual' if is_residual else 'additive'}.wav",
            mime="audio/wav",
            use_container_width=True,
        )

        # Waveform
        with st.expander("📈 Instrumental Waveform", expanded=True):
            fig = plot_waveform(
                instr_mono, results["sr"],
                title=f"Instrumental ({mode_label}) — Waveform", color="#7EE787",
            )
            st.pyplot(fig)
            plt_close(fig)

        # Spectrogram
        with st.expander("🌈 Instrumental Spectrogram", expanded=True):
            fig = plot_spectrogram(
                instr_spec, results["sr"],
                hop_length=hop_length,
                title=f"Instrumental ({mode_label}) — Spectrogram",
            )
            st.pyplot(fig)
            plt_close(fig)

    # --- 4-Stem Explorer Section ---
    if results.get("stems"):
        st.markdown('<div class="section-header"><h3>🥁 Isolated 4-Stem Inspector</h3></div>',
                    unsafe_allow_html=True)
        st.caption("Inspect and download individual constituent stems separated by Demucs v4:")

        stem_cols = st.columns(3)
        stem_configs = [
            ("drums", "🥁 Drums", "#58A6FF"),
            ("bass", "🎸 Bass", "#BC8CFF"),
            ("other", "🎹 Other Instruments", "#39D353"),
        ]

        for idx, (stem_key, stem_title, stem_color) in enumerate(stem_configs):
            if stem_key in results["stems"]:
                stem_data = results["stems"][stem_key]
                with stem_cols[idx]:
                    st.markdown(f"**{stem_title}**")
                    st.audio(stem_data["bytes"], format="audio/wav")
                    st.download_button(
                        label=f"⬇️ Download {stem_key.capitalize()} (WAV)",
                        data=stem_data["bytes"],
                        file_name=f"{stem_key}.wav",
                        mime="audio/wav",
                        use_container_width=True,
                    )
                    with st.expander(f"{stem_key.capitalize()} Waveform", expanded=False):
                        fig = plot_waveform(
                            stem_data["mono"], results["sr"],
                            title=f"{stem_key.capitalize()} — Waveform",
                            color=stem_color,
                        )
                        st.pyplot(fig)
                        plt_close(fig)

    # --- Advanced Analysis ---
    st.markdown('<div class="section-header"><h3>🔬 Advanced Analysis</h3></div>',
                unsafe_allow_html=True)

    tab_features, tab_spectrum, tab_about = st.tabs([
        "📊 Spectral Features", "🎵 Frequency Spectrum", "📖 About STFT"
    ])

    with tab_features:
        col_vf, col_if = st.columns(2)
        with col_vf:
            fig = plot_audio_features(
                results["vocals_features"], results["sr"],
                hop_length=hop_length,
                title="Vocals — Spectral Features",
            )
            st.pyplot(fig)
            plt_close(fig)

        with col_if:
            fig = plot_audio_features(
                instr_features, results["sr"],
                hop_length=hop_length,
                title=f"Instrumental ({mode_label}) — Spectral Features",
            )
            st.pyplot(fig)
            plt_close(fig)

    with tab_spectrum:
        col_vs, col_is = st.columns(2)
        with col_vs:
            fig = plot_frequency_spectrum(
                results["vocals_freqs"], results["vocals_mags"],
                title="Vocals — Frequency Spectrum",
                color="#F78166",
            )
            st.pyplot(fig)
            plt_close(fig)

        with col_is:
            fig = plot_frequency_spectrum(
                instr_freqs, instr_mags,
                title=f"Instrumental ({mode_label}) — Frequency Spectrum",
                color="#7EE787",
            )
            st.pyplot(fig)
            plt_close(fig)

    with tab_about:
        _render_stft_explainer()


# ---------------------------------------------------------------------------
# Educational Content
# ---------------------------------------------------------------------------

def _render_dsp_explainer() -> None:
    """Render an educational section about DSP and source separation."""
    with st.expander("📖 How does audio source separation work?", expanded=False):
        st.markdown("""
        ### The Challenge of Source Separation

        When you listen to a song, your brain effortlessly separates the singer's voice
        from the guitar, drums, and bass. But for a computer, this is extraordinarily
        difficult because **all instruments are mixed together in a single audio signal**.

        ### Why Simple Filtering Doesn't Work

        You might think: "Just filter out the frequencies of the voice!" Unfortunately:

        - **Overlapping frequencies**: Vocals (80 Hz – 8 kHz) overlap with guitars,
          keyboards, and most other instruments.
        - **Harmonics**: A singer's fundamental note at 220 Hz produces harmonics at
          440, 660, 880 Hz... — the same frequencies where instruments play.
        - **Phase**: The timing relationships between frequency components carry critical
          information that simple magnitude filtering destroys.

        ### How AI Solves It

        Modern source separation models like **Demucs** learn from thousands of songs
        where the individual stems (vocals, drums, bass, other) are available separately.

        The model learns:
        - What vocals "look like" in the spectrogram
        - Statistical patterns of instruments
        - How sources interact in a mix

        This is fundamentally different from filtering — it's **pattern recognition**
        applied to audio, similar to how image AI can separate objects from backgrounds.

        ### The DSP + ML Pipeline

        ```
        Song (time domain)
             ↓
        STFT → Spectrogram (time-frequency domain)
             ↓
        Neural Network → Estimated source masks
             ↓
        Apply masks + Inverse STFT
             ↓
        Separated sources (time domain)
        ```
        """)


def _render_stft_explainer() -> None:
    """Render an educational section about the STFT."""
    st.markdown("""
    ### Short-Time Fourier Transform (STFT)

    The **STFT** is the foundation of modern audio analysis and the primary representation
    used by source separation models.

    #### The Problem with a Single FFT

    A regular FFT computes the frequency content of an **entire signal at once**.
    This tells you *what* frequencies are present, but not *when* they occur.
    For music (where notes change over time), this is useless.

    #### The STFT Solution

    The STFT divides the signal into short, overlapping segments (frames) and
    computes the FFT of each one:

    ```
    Signal:  [============================]
    Frame 1: [====]
    Frame 2:    [====]
    Frame 3:       [====]
    Frame 4:          [====]
              ...
    ```

    Each frame produces one column of the **spectrogram** — a 2D image where:
    - **X-axis** = Time
    - **Y-axis** = Frequency
    - **Color** = Magnitude (how loud that frequency is at that time)

    #### Key Parameters

    | Parameter | Default | Effect |
    |-----------|---------|--------|
    | `n_fft` | 2048 | Window size. Larger = better frequency resolution |
    | `hop_length` | 512 | Step size. Smaller = better time resolution |
    | Window | Hann | Reduces spectral leakage at frame boundaries |

    #### The Uncertainty Trade-off

    You can't have perfect time **and** frequency resolution simultaneously
    (analogous to Heisenberg's uncertainty principle). Shorter windows give
    better time resolution but smear frequencies; longer windows resolve
    frequencies precisely but blur timing.

    The default values (2048 samples ≈ 46 ms at 44.1 kHz, hop = 512 ≈ 11.6 ms)
    provide a good balance for music analysis.

    #### Why This Matters for Source Separation

    Source separation models like Demucs operate on STFT representations
    internally. They learn to predict **masks** — values between 0 and 1 for
    each time-frequency bin — that indicate how much of each bin belongs to
    each source. The masked STFT is then converted back to audio using the
    **inverse STFT (iSTFT)**.
    """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plt_close(fig) -> None:
    """Close a matplotlib figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
