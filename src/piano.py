"""
Piano arrangement module.

Converts an instrumental audio track into a piano rendition by:
1. Transcribing the musical content (melody, chords, rhythm) into note events
2. Constructing a MIDI representation with piano instrumentation
3. Synthesizing the MIDI using a SoundFont-based piano sampler (FluidSynth)

This produces a *piano arrangement* — NOT a piano stem extraction. The output
sounds like the instrumental's musical content has been re-played on piano.

Architecture
------------
    PianoConverter
        ├── _transcribe()    — CQT + onset + harmonic analysis → NoteEvent list
        ├── _notes_to_midi() — NoteEvent list → PrettyMIDI object
        └── _synthesize()    — PrettyMIDI → WAV audio via FluidSynth

Transcription Algorithm
-----------------------
Traditional FFT peak detection fails on mixed instrumental music because
overlapping harmonics from multiple instruments create ambiguous frequency
peaks. Instead, this module uses a multi-stage pipeline:

1. **Harmonic-Percussive Separation** (HPSS): Isolate the tonal/harmonic
   content from percussive transients. This prevents drum hits from being
   transcribed as pitched notes.

2. **Constant-Q Transform** (CQT): Unlike the FFT (which has linear
   frequency bins), the CQT uses logarithmically-spaced bins that align
   exactly with musical note frequencies (1 bin = 1 semitone). This gives
   note-level frequency resolution across the full pitch range.

3. **Onset Detection**: Identify the start times of musical events using
   spectral flux. Each onset marks a potential new note or chord.

4. **Per-Segment Note Extraction**: Between consecutive onsets, analyze the
   CQT energy to identify which notes are active. A note is considered
   present if its CQT bin energy exceeds a threshold relative to the
   segment's peak energy.

5. **Velocity Estimation**: Map each note's CQT energy to a MIDI velocity
   value (0-127), preserving dynamic variation from the original.

6. **Note Merging**: Consecutive segments with the same pitch are merged
   into a single sustained note to avoid machine-gun retriggering.

Why FluidSynth?
---------------
Sine-wave synthesis sounds artificial and lifeless. FluidSynth renders MIDI
using sampled recordings of real piano notes (SoundFont format), producing
natural attack, sustain, and release characteristics with proper velocity
layering and stereo imaging.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """A single detected musical note.

    Attributes:
        pitch: MIDI note number (0-127). Middle C = 60.
        start_time: Note onset in seconds.
        end_time: Note offset in seconds.
        velocity: MIDI velocity (1-127). Higher = louder.
    """

    pitch: int
    start_time: float
    end_time: float
    velocity: int = 80

    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.end_time - self.start_time

    @property
    def note_name(self) -> str:
        """Human-readable note name (e.g., 'C4', 'F#5')."""
        return librosa.midi_to_note(self.pitch)


@dataclass
class PianoResult:
    """Result of a piano arrangement conversion.

    Attributes:
        audio: Synthesized piano audio, shape (samples,), mono, float32.
        sample_rate: Sample rate in Hz.
        note_count: Number of notes detected and synthesized.
        duration: Duration of the output audio in seconds.
        elapsed_time: Processing time in seconds.
        midi_data: The intermediate PrettyMIDI object (for inspection/export).
        note_events: The raw detected note events.
    """

    audio: np.ndarray
    sample_rate: int
    note_count: int
    duration: float
    elapsed_time: float
    midi_data: object = None
    note_events: list[NoteEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PianoConversionError(Exception):
    """Raised when piano conversion fails."""

    pass


# ---------------------------------------------------------------------------
# SoundFont Discovery
# ---------------------------------------------------------------------------

# Common system paths where SoundFont files are installed
_SF2_SEARCH_PATHS: list[str] = [
    "/opt/homebrew/share/fluid-synth/sf2/",
    "/opt/homebrew/share/soundfonts/",
    "/usr/share/sounds/sf2/",
    "/usr/share/soundfonts/",
    "/usr/local/share/fluid-synth/sf2/",
    "/usr/local/share/soundfonts/",
]

# Default SF2 download URL (GeneralUser GS — high-quality free General MIDI)
_DEFAULT_SF2_URL = (
    "https://github.com/FluidSynth/fluidsynth/releases/download/"
    "v2.3.4/GeneralUser_GS_v1.471.sf2"
)
_DEFAULT_SF2_FILENAME = "GeneralUser_GS.sf2"


def find_soundfont(custom_path: str | Path | None = None) -> Path | None:
    """
    Locate a SoundFont (.sf2) file on the system.

    Search order:
    1. Custom path if provided
    2. App data directory (~/.cache/musicvocals/soundfonts/)
    3. Common system paths (Homebrew, /usr/share, etc.)

    Args:
        custom_path: Explicit path to a .sf2 file.

    Returns:
        Path to the SoundFont file, or None if not found.
    """
    # 1. Custom path
    if custom_path is not None:
        p = Path(custom_path)
        if p.exists() and p.suffix.lower() in (".sf2", ".sf3"):
            return p

    # 2. App cache directory
    cache_dir = Path.home() / ".cache" / "musicvocals" / "soundfonts"
    if cache_dir.exists():
        for sf in cache_dir.glob("*.sf2"):
            return sf

    # 3. System paths
    for search_dir in _SF2_SEARCH_PATHS:
        d = Path(search_dir)
        if d.exists():
            for sf in sorted(d.glob("*.sf2")):
                # Skip files with encoding-garbled names
                try:
                    _ = str(sf)
                    if sf.stat().st_size > 100_000:  # at least 100 KB
                        return sf
                except (OSError, UnicodeError):
                    continue

    return None


def download_soundfont(
    url: str = _DEFAULT_SF2_URL,
    filename: str = _DEFAULT_SF2_FILENAME,
) -> Path:
    """
    Download a SoundFont file to the local cache.

    Args:
        url: URL to download the SF2 file from.
        filename: Local filename to save as.

    Returns:
        Path to the downloaded SoundFont.

    Raises:
        PianoConversionError: If download fails.
    """
    cache_dir = Path.home() / ".cache" / "musicvocals" / "soundfonts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename

    if target.exists() and target.stat().st_size > 100_000:
        logger.info(f"SoundFont already cached: {target}")
        return target

    logger.info(f"Downloading SoundFont from {url} ...")

    try:
        import urllib.request

        urllib.request.urlretrieve(url, str(target))
        logger.info(f"SoundFont downloaded to {target} ({target.stat().st_size:,} bytes)")
        return target
    except Exception as e:
        # Clean up partial download
        if target.exists():
            target.unlink()
        raise PianoConversionError(
            f"Failed to download SoundFont: {e}. "
            "You can manually place a .sf2 file in "
            f"{cache_dir}/ or install one via your package manager."
        ) from e


# ---------------------------------------------------------------------------
# Piano Converter
# ---------------------------------------------------------------------------

class PianoConverter:
    """
    Converts instrumental audio into a piano arrangement.

    Pipeline:
        Audio → Transcription (CQT + onset) → MIDI → Piano Synthesis (FluidSynth)

    Usage::

        converter = PianoConverter()
        result = converter.convert(audio_array, sample_rate=44100)
        piano_audio = result.audio  # numpy array, mono, 44.1 kHz
    """

    # Transcription parameters
    DEFAULT_HOP_LENGTH: int = 512
    DEFAULT_N_BINS: int = 72       # 6 octaves (C2–B7)
    DEFAULT_BINS_PER_OCTAVE: int = 12
    DEFAULT_FMIN: float = 65.41    # C2
    MIN_NOTE_DURATION: float = 0.05   # 50 ms minimum note length
    NOTE_THRESHOLD: float = 0.12   # CQT energy threshold (relative to peak)
    MIDI_BASE_NOTE: int = 36       # C2 = MIDI 36

    # Synthesis
    DEFAULT_SR: int = 44100
    MAX_DURATION_WARNING: float = 300.0  # 5 minutes

    def __init__(self, sf2_path: str | Path | None = None):
        """
        Initialize the piano converter.

        Args:
            sf2_path: Path to a SoundFont (.sf2) file. If None,
                      auto-discovers one from system paths.
        """
        self._sf2_path: Path | None = None

        if sf2_path is not None:
            p = Path(sf2_path)
            if p.exists():
                self._sf2_path = p
            else:
                logger.warning(f"Provided SF2 path does not exist: {sf2_path}")

        if self._sf2_path is None:
            self._sf2_path = find_soundfont()

        if self._sf2_path is not None:
            logger.info(f"Using SoundFont: {self._sf2_path}")
        else:
            logger.warning(
                "No SoundFont found. Will attempt download on first synthesis. "
                "Install FluidSynth with: brew install fluid-synth"
            )

    @property
    def soundfont_path(self) -> Path | None:
        """Current SoundFont path, or None if not yet resolved."""
        return self._sf2_path

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int = DEFAULT_SR,
        note_threshold: float | None = None,
        min_note_duration: float | None = None,
    ) -> PianoResult:
        """
        Convert instrumental audio into a piano arrangement.

        Args:
            audio: Audio array, shape (channels, samples) or (samples,).
                   Values in [-1.0, 1.0].
            sample_rate: Sample rate of the input audio in Hz.
            note_threshold: CQT energy threshold for note detection.
                            Lower = more notes, higher = only prominent notes.
            min_note_duration: Minimum note length in seconds.

        Returns:
            PianoResult with synthesized piano audio and metadata.

        Raises:
            PianoConversionError: If transcription or synthesis fails.
        """
        t0 = time.time()

        threshold = note_threshold or self.NOTE_THRESHOLD
        min_dur = min_note_duration or self.MIN_NOTE_DURATION

        # --- Validate input ---
        if audio is None or audio.size == 0:
            raise PianoConversionError("Input audio is empty.")

        if audio.ndim > 2:
            raise PianoConversionError(
                f"Unexpected audio shape: {audio.shape}. "
                "Expected (samples,) or (channels, samples)."
            )

        # Convert to mono
        if audio.ndim == 2:
            if audio.shape[0] <= audio.shape[1]:
                # (channels, samples)
                audio_mono = np.mean(audio, axis=0).astype(np.float32)
            else:
                # (samples, channels)
                audio_mono = np.mean(audio, axis=1).astype(np.float32)
        else:
            audio_mono = audio.astype(np.float32)

        duration = len(audio_mono) / sample_rate
        if duration < 0.5:
            raise PianoConversionError(
                f"Audio too short for transcription ({duration:.1f}s). "
                "Minimum duration is 0.5 seconds."
            )

        logger.info(
            f"Starting piano conversion: {duration:.1f}s audio "
            f"at {sample_rate} Hz"
        )

        # --- Step 1: Transcribe ---
        try:
            note_events = self._transcribe(
                audio_mono, sample_rate,
                threshold=threshold,
                min_note_duration=min_dur,
            )
        except Exception as e:
            raise PianoConversionError(
                f"Music transcription failed: {e}"
            ) from e

        if not note_events:
            raise PianoConversionError(
                "No musical notes detected in the instrumental. "
                "The audio may be too quiet, too percussive, or too short."
            )

        logger.info(f"Transcribed {len(note_events)} notes")

        # --- Step 2: Build MIDI ---
        try:
            midi_data = self._notes_to_midi(note_events)
        except Exception as e:
            raise PianoConversionError(
                f"MIDI construction failed: {e}"
            ) from e

        # --- Step 3: Synthesize ---
        try:
            piano_audio = self._synthesize(midi_data, sample_rate)
        except Exception as e:
            raise PianoConversionError(
                f"Piano synthesis failed: {e}"
            ) from e

        elapsed = time.time() - t0
        logger.info(
            f"Piano conversion complete: {len(note_events)} notes, "
            f"{elapsed:.1f}s processing time"
        )

        return PianoResult(
            audio=piano_audio,
            sample_rate=sample_rate,
            note_count=len(note_events),
            duration=len(piano_audio) / sample_rate,
            elapsed_time=elapsed,
            midi_data=midi_data,
            note_events=note_events,
        )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe(
        self,
        audio_mono: np.ndarray,
        sr: int,
        threshold: float = NOTE_THRESHOLD,
        min_note_duration: float = MIN_NOTE_DURATION,
    ) -> list[NoteEvent]:
        """
        Transcribe audio into note events using CQT + onset analysis.

        Pipeline:
        1. Harmonic-percussive source separation
        2. CQT computation (note-level frequency resolution)
        3. Onset detection
        4. Per-segment note extraction with energy thresholding
        5. Velocity estimation
        6. Note merging and filtering

        Args:
            audio_mono: Mono audio array.
            sr: Sample rate.
            threshold: CQT energy threshold (0.0–1.0).
            min_note_duration: Minimum note duration in seconds.

        Returns:
            List of NoteEvent objects sorted by start time.
        """
        hop = self.DEFAULT_HOP_LENGTH

        # 1. Harmonic-percussive separation: keep only harmonic content
        harmonic, _ = librosa.effects.hpss(audio_mono)

        # 2. Compute Constant-Q Transform
        cqt = np.abs(librosa.cqt(
            y=harmonic,
            sr=sr,
            hop_length=hop,
            fmin=self.DEFAULT_FMIN,
            n_bins=self.DEFAULT_N_BINS,
            bins_per_octave=self.DEFAULT_BINS_PER_OCTAVE,
        ))

        # Convert to dB scale for better dynamic range handling
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

        # 3. Onset detection on the original (not just harmonic) for timing
        onset_frames = librosa.onset.onset_detect(
            y=audio_mono,
            sr=sr,
            hop_length=hop,
            backtrack=True,
            units="frames",
        )

        # Ensure we have at least a start frame
        if len(onset_frames) == 0 or onset_frames[0] != 0:
            onset_frames = np.concatenate([[0], onset_frames])

        # Add end frame
        total_frames = cqt.shape[1]
        if onset_frames[-1] < total_frames - 1:
            onset_frames = np.concatenate([onset_frames, [total_frames - 1]])

        onset_times = librosa.frames_to_time(
            onset_frames, sr=sr, hop_length=hop
        )

        # 4. Per-segment note extraction
        raw_notes: list[NoteEvent] = []

        for i in range(len(onset_frames) - 1):
            seg_start_frame = onset_frames[i]
            seg_end_frame = onset_frames[i + 1]
            t_start = onset_times[i]
            t_end = onset_times[i + 1]

            if t_end - t_start < 0.02:  # skip tiny segments (< 20 ms)
                continue

            # Extract CQT segment
            seg = cqt[:, seg_start_frame:seg_end_frame]
            if seg.size == 0:
                continue

            # Mean energy per frequency bin across the segment
            bin_energy = np.mean(seg, axis=1)
            peak_energy = np.max(bin_energy)

            if peak_energy < 1e-6:
                continue  # silence

            # Find active notes: bins with energy above threshold
            active_threshold = peak_energy * threshold
            active_bins = np.where(bin_energy > active_threshold)[0]

            for bin_idx in active_bins:
                midi_note = bin_idx + self.MIDI_BASE_NOTE

                # Clamp to valid MIDI range
                if midi_note < 21 or midi_note > 108:
                    continue  # outside standard piano range (A0–C8)

                # Velocity: map energy to 30–120 range
                note_energy = bin_energy[bin_idx] / peak_energy
                velocity = int(30 + note_energy * 90)
                velocity = max(30, min(127, velocity))

                raw_notes.append(NoteEvent(
                    pitch=midi_note,
                    start_time=float(t_start),
                    end_time=float(t_end),
                    velocity=velocity,
                ))

        # 5. Merge consecutive same-pitch notes
        notes = self._merge_notes(raw_notes)

        # 6. Filter by minimum duration
        notes = [n for n in notes if n.duration >= min_note_duration]

        # Sort by start time, then pitch
        notes.sort(key=lambda n: (n.start_time, n.pitch))

        return notes

    @staticmethod
    def _merge_notes(
        notes: list[NoteEvent],
        merge_gap: float = 0.03,
    ) -> list[NoteEvent]:
        """
        Merge consecutive note events with the same pitch.

        If two notes of the same pitch have a gap ≤ merge_gap seconds,
        they are combined into a single sustained note. This prevents
        machine-gun retriggering artifacts.

        Args:
            notes: Unsorted list of NoteEvent objects.
            merge_gap: Maximum gap (seconds) between notes to merge.

        Returns:
            Merged list of NoteEvent objects.
        """
        if not notes:
            return []

        # Group by pitch
        by_pitch: dict[int, list[NoteEvent]] = {}
        for n in notes:
            by_pitch.setdefault(n.pitch, []).append(n)

        merged: list[NoteEvent] = []

        for pitch, pitch_notes in by_pitch.items():
            pitch_notes.sort(key=lambda n: n.start_time)

            current = NoteEvent(
                pitch=pitch_notes[0].pitch,
                start_time=pitch_notes[0].start_time,
                end_time=pitch_notes[0].end_time,
                velocity=pitch_notes[0].velocity,
            )

            for n in pitch_notes[1:]:
                if n.start_time - current.end_time <= merge_gap:
                    # Merge: extend current note, keep higher velocity
                    current.end_time = n.end_time
                    current.velocity = max(current.velocity, n.velocity)
                else:
                    merged.append(current)
                    current = NoteEvent(
                        pitch=n.pitch,
                        start_time=n.start_time,
                        end_time=n.end_time,
                        velocity=n.velocity,
                    )

            merged.append(current)

        return merged

    # ------------------------------------------------------------------
    # MIDI Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _notes_to_midi(note_events: list[NoteEvent]) -> "pretty_midi.PrettyMIDI":
        """
        Construct a PrettyMIDI object from note events.

        All notes are assigned to a single Acoustic Grand Piano instrument
        (MIDI program 0), preserving the original timing and velocity.

        Args:
            note_events: List of NoteEvent objects.

        Returns:
            A PrettyMIDI object ready for synthesis.
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)

        # Acoustic Grand Piano (General MIDI program 0)
        piano = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")

        for event in note_events:
            note = pretty_midi.Note(
                velocity=event.velocity,
                pitch=event.pitch,
                start=event.start_time,
                end=event.end_time,
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        return midi

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        midi_data: "pretty_midi.PrettyMIDI",
        sample_rate: int = DEFAULT_SR,
    ) -> np.ndarray:
        """
        Render a PrettyMIDI object to audio using FluidSynth.

        Tries in order:
        1. pretty_midi.fluidsynth() with a discovered/downloaded SoundFont
        2. FluidSynth CLI fallback (if pyfluidsynth binding fails)
        3. pretty_midi.synthesize() sine-wave fallback (last resort)

        Args:
            midi_data: PrettyMIDI object to render.
            sample_rate: Target sample rate.

        Returns:
            Audio numpy array, shape (samples,), float32, normalized.
        """
        # Ensure we have a SoundFont
        sf2 = self._sf2_path
        if sf2 is None:
            logger.info("No SoundFont found, attempting download...")
            try:
                sf2 = download_soundfont()
                self._sf2_path = sf2
            except PianoConversionError:
                logger.warning("SoundFont download failed. Using sine-wave fallback.")

        # --- Strategy 1: pretty_midi.fluidsynth() ---
        if sf2 is not None:
            try:
                import fluidsynth as _fs

                synth = _fs.Synth(samplerate=float(sample_rate))
                sfid = synth.sfload(str(sf2))
                synth.program_select(0, sfid, 0, 0)
                audio = midi_data.fluidsynth(
                    fs=sample_rate, synthesizer=synth,
                )
                synth.delete()
                logger.info("Synthesis via pretty_midi.fluidsynth() succeeded")
                return self._normalize(audio)
            except Exception as e:
                logger.warning(f"pretty_midi.fluidsynth() failed: {e}")

        # --- Strategy 2: FluidSynth CLI ---
        if sf2 is not None and shutil.which("fluidsynth"):
            try:
                audio = self._synthesize_cli(midi_data, sample_rate, sf2)
                logger.info("Synthesis via FluidSynth CLI succeeded")
                return self._normalize(audio)
            except Exception as e:
                logger.warning(f"FluidSynth CLI synthesis failed: {e}")

        # --- Strategy 3: Sine-wave fallback ---
        logger.warning(
            "Using sine-wave synthesis fallback. Install FluidSynth for "
            "natural-sounding piano: brew install fluid-synth"
        )
        try:
            audio = midi_data.synthesize(fs=sample_rate)
            return self._normalize(audio)
        except Exception as e:
            raise PianoConversionError(
                f"All synthesis methods failed. Last error: {e}. "
                "Please install FluidSynth: brew install fluid-synth"
            ) from e

    def _synthesize_cli(
        self,
        midi_data: "pretty_midi.PrettyMIDI",
        sample_rate: int,
        sf2_path: Path,
    ) -> np.ndarray:
        """
        Render MIDI to audio using the FluidSynth command-line tool.

        This is a fallback when the pyfluidsynth Python binding fails.
        It saves the MIDI to a temp file, renders via CLI, and loads
        the resulting WAV.

        Args:
            midi_data: PrettyMIDI object to render.
            sample_rate: Target sample rate.
            sf2_path: Path to the SoundFont file.

        Returns:
            Audio numpy array.
        """
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="piano_") as tmpdir:
            midi_path = os.path.join(tmpdir, "piano.mid")
            wav_path = os.path.join(tmpdir, "piano.wav")

            midi_data.write(midi_path)

            fluidsynth_bin = shutil.which("fluidsynth") or "fluidsynth"

            result = subprocess.run(
                [
                    fluidsynth_bin,
                    "-ni",           # non-interactive
                    "-F", wav_path,  # output file
                    "-r", str(sample_rate),
                    "-g", "1.0",     # gain
                    str(sf2_path),
                    midi_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FluidSynth CLI exited with code {result.returncode}: "
                    f"{result.stderr}"
                )

            audio, _ = sf.read(wav_path, dtype="float32")

            # Convert to mono if stereo
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            return audio

    @staticmethod
    def _normalize(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
        """
        Peak-normalize audio with safety headroom.

        Args:
            audio: Audio array.
            headroom_db: Headroom below 0 dBFS (e.g., -1.0 = normalize to -1 dBFS).

        Returns:
            Normalized audio array, float32.
        """
        audio = audio.astype(np.float32)

        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio  # silence

        target_peak = 10 ** (headroom_db / 20.0)  # e.g., -1 dB → 0.891
        audio = audio * (target_peak / peak)

        return audio
