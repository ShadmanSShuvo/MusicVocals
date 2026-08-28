"""
Tests for the piano arrangement module (src/piano.py).

All tests use synthetic audio signals (sine waves, silence, chords)
so no pretrained model downloads or external files are needed.
"""

import numpy as np
import pytest

from src.piano import (
    NoteEvent,
    PianoConversionError,
    PianoConverter,
    PianoResult,
    find_soundfont,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_wave(freq: float, duration: float, sr: int = 44100, amplitude: float = 0.5) -> np.ndarray:
    """Generate a mono sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _chord(freqs: list[float], duration: float, sr: int = 44100) -> np.ndarray:
    """Generate a mono chord (sum of sines)."""
    amp = 0.4 / len(freqs)
    return sum(_sine_wave(f, duration, sr, amp) for f in freqs)


def _arpeggio(freqs: list[float], note_dur: float = 1.0, sr: int = 44100) -> np.ndarray:
    """Generate a mono arpeggio (sequential notes)."""
    parts = [_sine_wave(f, note_dur, sr) for f in freqs]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# NoteEvent
# ---------------------------------------------------------------------------

class TestNoteEvent:
    """Tests for the NoteEvent dataclass."""

    def test_basic_properties(self):
        note = NoteEvent(pitch=60, start_time=0.0, end_time=1.0, velocity=100)
        assert note.pitch == 60
        assert note.start_time == 0.0
        assert note.end_time == 1.0
        assert note.velocity == 100

    def test_duration(self):
        note = NoteEvent(pitch=60, start_time=0.5, end_time=1.5)
        assert note.duration == pytest.approx(1.0)

    def test_note_name(self):
        # Middle C
        note = NoteEvent(pitch=60, start_time=0.0, end_time=1.0)
        assert "C" in note.note_name

    def test_default_velocity(self):
        note = NoteEvent(pitch=60, start_time=0.0, end_time=1.0)
        assert note.velocity == 80


# ---------------------------------------------------------------------------
# PianoResult
# ---------------------------------------------------------------------------

class TestPianoResult:
    """Tests for the PianoResult dataclass."""

    def test_basic_construction(self):
        audio = np.zeros(44100, dtype=np.float32)
        result = PianoResult(
            audio=audio,
            sample_rate=44100,
            note_count=10,
            duration=1.0,
            elapsed_time=0.5,
        )
        assert result.note_count == 10
        assert result.duration == 1.0
        assert result.sample_rate == 44100
        assert result.audio.shape == (44100,)

    def test_default_fields(self):
        audio = np.zeros(100)
        result = PianoResult(
            audio=audio, sample_rate=44100,
            note_count=0, duration=0.0, elapsed_time=0.0,
        )
        assert result.midi_data is None
        assert result.note_events == []


# ---------------------------------------------------------------------------
# SoundFont Discovery
# ---------------------------------------------------------------------------

class TestSoundfontDiscovery:
    """Tests for SoundFont file discovery."""

    def test_find_soundfont_returns_path_or_none(self):
        result = find_soundfont()
        # May be None on CI/containers, but should not crash
        assert result is None or result.exists()

    def test_find_soundfont_with_invalid_custom_path(self):
        result = find_soundfont("/nonexistent/path/fake.sf2")
        # Should fall back to system search, not crash
        assert result is None or result.exists()


# ---------------------------------------------------------------------------
# PianoConverter Initialization
# ---------------------------------------------------------------------------

class TestConverterInit:
    """Tests for PianoConverter initialization."""

    def test_default_init(self):
        converter = PianoConverter()
        # Should not crash; soundfont may or may not be found
        assert isinstance(converter, PianoConverter)

    def test_custom_sf2_path_invalid(self):
        converter = PianoConverter(sf2_path="/nonexistent/fake.sf2")
        # Should fall back gracefully, not crash
        assert isinstance(converter, PianoConverter)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class TestTranscription:
    """Tests for the music transcription pipeline."""

    def test_detects_single_note(self):
        """A pure C4 sine wave should yield note events near MIDI 60."""
        audio = _sine_wave(261.63, 2.0)  # C4 for 2 seconds
        converter = PianoConverter()
        notes = converter._transcribe(audio, 44100)

        assert len(notes) > 0
        pitches = [n.pitch for n in notes]
        # Should contain MIDI note 60 (C4) or neighbors (59, 61)
        assert any(59 <= p <= 61 for p in pitches), f"Expected C4 (60), got {pitches}"

    def test_detects_multiple_notes(self):
        """An arpeggio should detect multiple distinct pitches."""
        audio = _arpeggio([261.63, 329.63, 392.00], note_dur=1.0)  # C4, E4, G4
        converter = PianoConverter()
        notes = converter._transcribe(audio, 44100)

        pitches = set(n.pitch for n in notes)
        assert len(pitches) >= 2, f"Expected multiple pitches, got {pitches}"

    def test_chord_detection(self):
        """A simultaneous chord should detect multiple concurrent pitches."""
        audio = _chord([261.63, 329.63, 392.00], duration=2.0)  # C maj chord
        converter = PianoConverter()
        notes = converter._transcribe(audio, 44100)

        pitches = set(n.pitch for n in notes)
        assert len(pitches) >= 2, f"Expected multiple pitches in chord, got {pitches}"

    def test_silence_returns_empty(self):
        """Silence should yield no notes."""
        audio = np.zeros(44100 * 2, dtype=np.float32)
        converter = PianoConverter()
        notes = converter._transcribe(audio, 44100)
        assert len(notes) == 0

    def test_note_timing_reasonable(self):
        """Notes should have start/end times within the audio duration."""
        audio = _sine_wave(440.0, 3.0)
        converter = PianoConverter()
        notes = converter._transcribe(audio, 44100)

        for n in notes:
            assert n.start_time >= 0.0
            assert n.end_time <= 3.5  # small tolerance for onset detection
            assert n.end_time > n.start_time

    def test_velocity_range(self):
        """All velocities should be in valid MIDI range."""
        audio = _sine_wave(440.0, 2.0)
        converter = PianoConverter()
        notes = converter._transcribe(audio, 44100)

        for n in notes:
            assert 1 <= n.velocity <= 127


# ---------------------------------------------------------------------------
# Note Merging
# ---------------------------------------------------------------------------

class TestNoteMerging:
    """Tests for the note merging algorithm."""

    def test_merges_consecutive_same_pitch(self):
        notes = [
            NoteEvent(pitch=60, start_time=0.0, end_time=0.5, velocity=80),
            NoteEvent(pitch=60, start_time=0.51, end_time=1.0, velocity=90),
        ]
        merged = PianoConverter._merge_notes(notes, merge_gap=0.05)
        assert len(merged) == 1
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 1.0
        assert merged[0].velocity == 90  # keeps max velocity

    def test_does_not_merge_different_pitches(self):
        notes = [
            NoteEvent(pitch=60, start_time=0.0, end_time=0.5),
            NoteEvent(pitch=64, start_time=0.0, end_time=0.5),
        ]
        merged = PianoConverter._merge_notes(notes)
        assert len(merged) == 2

    def test_does_not_merge_with_large_gap(self):
        notes = [
            NoteEvent(pitch=60, start_time=0.0, end_time=0.5),
            NoteEvent(pitch=60, start_time=1.0, end_time=1.5),
        ]
        merged = PianoConverter._merge_notes(notes, merge_gap=0.03)
        assert len(merged) == 2

    def test_empty_input(self):
        assert PianoConverter._merge_notes([]) == []


# ---------------------------------------------------------------------------
# MIDI Construction
# ---------------------------------------------------------------------------

class TestMIDIConstruction:
    """Tests for MIDI file construction from note events."""

    def test_creates_piano_instrument(self):
        import pretty_midi

        notes = [NoteEvent(pitch=60, start_time=0.0, end_time=1.0, velocity=100)]
        midi = PianoConverter._notes_to_midi(notes)

        assert isinstance(midi, pretty_midi.PrettyMIDI)
        assert len(midi.instruments) == 1
        assert midi.instruments[0].program == 0  # Acoustic Grand Piano
        assert not midi.instruments[0].is_drum

    def test_note_count_matches(self):
        notes = [
            NoteEvent(pitch=60, start_time=0.0, end_time=0.5),
            NoteEvent(pitch=64, start_time=0.5, end_time=1.0),
            NoteEvent(pitch=67, start_time=1.0, end_time=1.5),
        ]
        midi = PianoConverter._notes_to_midi(notes)
        assert len(midi.instruments[0].notes) == 3

    def test_preserves_timing(self):
        notes = [NoteEvent(pitch=60, start_time=0.5, end_time=1.5, velocity=100)]
        midi = PianoConverter._notes_to_midi(notes)
        midi_note = midi.instruments[0].notes[0]
        assert midi_note.start == pytest.approx(0.5)
        assert midi_note.end == pytest.approx(1.5)

    def test_preserves_velocity(self):
        notes = [NoteEvent(pitch=60, start_time=0.0, end_time=1.0, velocity=42)]
        midi = PianoConverter._notes_to_midi(notes)
        assert midi.instruments[0].notes[0].velocity == 42


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

class TestSynthesis:
    """Tests for piano audio synthesis."""

    def test_output_is_float32(self):
        converter = PianoConverter()
        notes = [NoteEvent(pitch=60, start_time=0.0, end_time=1.0, velocity=100)]
        midi = converter._notes_to_midi(notes)
        audio = converter._synthesize(midi, 44100)
        assert audio.dtype == np.float32

    def test_output_is_mono(self):
        converter = PianoConverter()
        notes = [NoteEvent(pitch=60, start_time=0.0, end_time=1.0, velocity=100)]
        midi = converter._notes_to_midi(notes)
        audio = converter._synthesize(midi, 44100)
        assert audio.ndim == 1

    def test_no_clipping(self):
        converter = PianoConverter()
        notes = [NoteEvent(pitch=60, start_time=0.0, end_time=1.0, velocity=127)]
        midi = converter._notes_to_midi(notes)
        audio = converter._synthesize(midi, 44100)
        assert np.max(np.abs(audio)) <= 1.0

    def test_normalization(self):
        audio = np.array([0.0, 0.5, -0.5, 0.3], dtype=np.float32)
        normalized = PianoConverter._normalize(audio, headroom_db=-1.0)
        peak = np.max(np.abs(normalized))
        expected_peak = 10 ** (-1.0 / 20.0)
        assert peak == pytest.approx(expected_peak, rel=0.01)

    def test_silence_normalization(self):
        """Normalizing silence should return silence without error."""
        audio = np.zeros(100, dtype=np.float32)
        normalized = PianoConverter._normalize(audio)
        assert np.all(normalized == 0.0)


# ---------------------------------------------------------------------------
# Full Pipeline (convert)
# ---------------------------------------------------------------------------

class TestConvert:
    """End-to-end tests for the full piano conversion pipeline."""

    def test_convert_mono_input(self):
        audio = _sine_wave(440.0, 2.0)
        converter = PianoConverter()
        result = converter.convert(audio, 44100)

        assert isinstance(result, PianoResult)
        assert result.note_count > 0
        assert result.audio.ndim == 1
        assert result.sample_rate == 44100
        assert result.elapsed_time > 0

    def test_convert_stereo_input(self):
        mono = _sine_wave(440.0, 2.0)
        stereo = np.stack([mono, mono])  # (2, samples)
        converter = PianoConverter()
        result = converter.convert(stereo, 44100)

        assert result.note_count > 0
        assert result.audio.ndim == 1

    def test_empty_input_raises(self):
        converter = PianoConverter()
        with pytest.raises(PianoConversionError, match="empty"):
            converter.convert(np.array([]), 44100)

    def test_too_short_raises(self):
        converter = PianoConverter()
        short = np.zeros(1000, dtype=np.float32)  # ~23ms at 44100
        with pytest.raises(PianoConversionError, match="too short"):
            converter.convert(short, 44100)

    def test_silence_raises(self):
        converter = PianoConverter()
        silence = np.zeros(44100 * 2, dtype=np.float32)
        with pytest.raises(PianoConversionError, match="No musical notes"):
            converter.convert(silence, 44100)

    def test_result_has_note_events(self):
        audio = _arpeggio([261.63, 440.0], note_dur=1.0)
        converter = PianoConverter()
        result = converter.convert(audio, 44100)
        assert len(result.note_events) == result.note_count
        assert all(isinstance(n, NoteEvent) for n in result.note_events)

    def test_result_has_midi_data(self):
        import pretty_midi

        audio = _sine_wave(440.0, 2.0)
        converter = PianoConverter()
        result = converter.convert(audio, 44100)
        assert isinstance(result.midi_data, pretty_midi.PrettyMIDI)

    def test_bad_shape_raises(self):
        converter = PianoConverter()
        bad = np.zeros((2, 3, 44100), dtype=np.float32)
        with pytest.raises(PianoConversionError, match="shape"):
            converter.convert(bad, 44100)
