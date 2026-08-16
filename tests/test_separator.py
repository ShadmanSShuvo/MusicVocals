"""
Tests for the separator module.

Tests device detection and separator initialization.
Full separation tests require the Demucs model (slow download),
so they are marked as integration tests.
"""

import numpy as np
import pytest
import torch

from src.separator import DemucsSeparator, SeparatorBase, select_device


# ---------------------------------------------------------------------------
# Device Selection Tests
# ---------------------------------------------------------------------------

class TestDeviceSelection:
    def test_select_device_returns_torch_device(self):
        """select_device should return a valid torch.device."""
        device = select_device()
        assert isinstance(device, torch.device)

    def test_select_device_is_known_type(self):
        """Device type should be one of cpu, cuda, or mps."""
        device = select_device()
        assert device.type in {"cpu", "cuda", "mps"}


# ---------------------------------------------------------------------------
# Separator Initialization Tests
# ---------------------------------------------------------------------------

class TestSeparatorInit:
    def test_demucs_separator_is_separator_base(self):
        """DemucsSeparator should implement SeparatorBase."""
        sep = DemucsSeparator(model_name="htdemucs")
        assert isinstance(sep, SeparatorBase)

    def test_not_loaded_initially(self):
        """Model should not be loaded until load_model() is called."""
        sep = DemucsSeparator()
        assert not sep.is_loaded

    def test_get_device_name(self):
        """get_device_name should return a string."""
        sep = DemucsSeparator()
        name = sep.get_device_name()
        assert isinstance(name, str)
        assert name in {"cpu", "cuda", "mps"}

    def test_separate_without_loading_raises(self):
        """Calling separate before load_model should raise RuntimeError."""
        sep = DemucsSeparator()
        dummy_audio = np.zeros(44100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            sep.separate(dummy_audio, 44100)


# ---------------------------------------------------------------------------
# Integration Test (requires model download, skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="Integration test skipped on CPU-only (slow model download)"
)
class TestSeparationIntegration:
    """These tests actually load the model and run separation.
    They are slow and require network access on first run."""

    def test_full_separation_pipeline(self):
        """End-to-end separation with a synthetic signal."""
        sep = DemucsSeparator(model_name="htdemucs")
        sep.load_model()
        assert sep.is_loaded

        # Create a 3-second stereo signal
        sr = 44100
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        left = 0.3 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 880 * t)
        stereo = np.stack([left, right]).astype(np.float32)

        results = sep.separate(stereo, sr, shifts=0, overlap=0.25, instrumental_mode="residual")

        assert "vocals" in results
        assert "instrumental" in results
        assert "instrumental_residual" in results
        assert "instrumental_additive" in results
        assert "drums" in results
        assert "bass" in results
        assert "other" in results
        assert results["vocals"].shape[0] == 2  # Stereo output
        assert results["instrumental"].shape[0] == 2
        assert results["instrumental_residual"].shape[0] == 2
        assert results["instrumental_additive"].shape[0] == 2
