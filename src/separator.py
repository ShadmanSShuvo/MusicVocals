"""
Source separation module.

Provides an abstract interface for audio source separation and a concrete
implementation using Meta's Demucs v4 (Hybrid Transformer) model.

Architecture
------------
The separator is designed with a clean abstraction layer:

    SeparatorBase (ABC)
        │
        └── DemucsSeparator
                - Uses htdemucs pretrained model
                - Separates into 4 stems: vocals, drums, bass, other
                - Recombines non-vocal stems into 'instrumental'

This makes it possible to swap in alternative models (e.g., Band-Split
Roformer, Open-Unmix) without changing the rest of the application.

Why Neural Source Separation?
-----------------------------
Traditional signal processing (e.g., FFT filtering, stereo subtraction)
cannot reliably separate vocals from instruments because:

1. **Overlapping frequency ranges**: Vocals (80 Hz–8 kHz) overlap heavily
   with guitars, keyboards, and other instruments.
2. **Harmonics**: Both vocals and instruments produce harmonics at the same
   frequencies.
3. **Phase relationships**: Simple phase cancellation only works for
   center-panned vocals in stereo recordings.
4. **Reverb and effects**: Mixing effects spread sources across the
   stereo field and frequency spectrum.

Neural models like Demucs learn statistical priors about what vocals and
instruments "sound like" from thousands of training examples, enabling
them to separate overlapping sources that are mathematically inseparable
by linear filtering alone.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class SeparatorBase(ABC):
    """
    Abstract base class for audio source separators.

    Subclasses must implement `load_model()` and `separate()`.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model into memory."""
        ...

    @abstractmethod
    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Separate an audio signal into component sources.

        Args:
            audio: Audio array, shape (channels, samples) or (samples,).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            Dictionary mapping source name to audio array.
            At minimum: {'vocals': ..., 'instrumental': ...}
        """
        ...

    @abstractmethod
    def get_device_name(self) -> str:
        """Return the name of the processing device (e.g., 'cpu', 'cuda', 'mps')."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        ...


# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """
    Select the best available processing device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device for the selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU for processing")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal) for processing")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for processing")
    return device


# ---------------------------------------------------------------------------
# Demucs Implementation
# ---------------------------------------------------------------------------

class DemucsSeparator(SeparatorBase):
    """
    Audio source separator using Meta's Demucs v4 (Hybrid Transformer).

    Demucs processes audio in both time and frequency domains using a
    Hybrid Transformer architecture. The model separates a stereo mix
    into 4 stems: vocals, drums, bass, and other instruments.

    For this application, we recombine drums + bass + other into a
    single 'instrumental' track.

    Model: htdemucs (Hybrid Transformer Demucs)
    - Input: stereo audio at 44.1 kHz
    - Output: 4 stems at 44.1 kHz
    - Architecture: U-Net with Transformer encoder in both time and
      frequency branches
    """

    # Demucs native sample rate
    NATIVE_SR: int = 44100

    def __init__(self, model_name: str = "htdemucs", device: torch.device | None = None):
        """
        Initialize the Demucs separator.

        Args:
            model_name: Name of the Demucs model to use.
                        Options: 'htdemucs', 'htdemucs_ft', 'mdx_extra'
            device: Processing device. Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or select_device()
        self._model: Any = None
        self._model_sr: int = self.NATIVE_SR

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return self._model is not None

    def get_device_name(self) -> str:
        """Return the processing device name."""
        return str(self.device)

    def load_model(self) -> None:
        """
        Load the pretrained Demucs model.

        Downloads the model weights on first run, then caches them
        in the torch hub cache directory (~/.cache/torch/hub/).
        """
        if self._model is not None:
            logger.info("Model already loaded, skipping")
            return

        try:
            from demucs.pretrained import get_model

            logger.info(f"Loading Demucs model: {self.model_name}")
            self._model = get_model(self.model_name)
            self._model.to(self.device)
            self._model.eval()

            # Get the model's native sample rate
            if hasattr(self._model, "samplerate"):
                self._model_sr = self._model.samplerate

            logger.info(
                f"Model loaded successfully on {self.device} "
                f"(sample rate: {self._model_sr} Hz)"
            )

        except ImportError as e:
            raise RuntimeError(
                "Demucs is not installed. Install it with: pip install demucs"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Demucs model '{self.model_name}': {e}. "
                "This may be a network issue (first run downloads model weights) "
                "or a compatibility issue."
            ) from e

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Separate audio into vocals and instrumental tracks.

        Pipeline:
        1. Convert numpy array to PyTorch tensor
        2. Ensure stereo format (Demucs requires stereo)
        3. Resample to model's native rate if needed
        4. Run inference with torch.no_grad()
        5. Extract stems and recombine into vocals + instrumental
        6. Convert back to numpy arrays

        Args:
            audio: Audio array, shape (samples,) or (channels, samples).
                   Values in [-1.0, 1.0].
            sample_rate: Sample rate of the input audio in Hz.

        Returns:
            Dictionary with:
            - 'vocals': Vocal track, shape (channels, samples)
            - 'instrumental': Instrumental track, shape (channels, samples)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # --- Step 1: Prepare tensor ---
        tensor = self._numpy_to_tensor(audio)

        # --- Step 2: Ensure stereo ---
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).expand(2, -1)
        elif tensor.dim() == 2 and tensor.shape[0] == 1:
            tensor = tensor.expand(2, -1)

        # Add batch dimension: (batch, channels, samples)
        tensor = tensor.unsqueeze(0)

        # --- Step 3: Move to device ---
        tensor = tensor.to(self.device)

        # --- Step 4: Run inference ---
        try:
            with torch.no_grad():
                # Demucs apply_model handles chunking for long audio
                from demucs.apply import apply_model

                sources = apply_model(
                    self._model,
                    tensor,
                    device=self.device,
                    progress=False,
                )
                # sources shape: (batch, num_sources, channels, samples)

        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                "GPU ran out of memory during separation. "
                "Try a shorter audio file or switch to CPU processing."
            )
        except Exception as e:
            raise RuntimeError(f"Source separation failed: {e}") from e

        # --- Step 5: Extract and recombine stems ---
        # Demucs source order: drums, bass, other, vocals
        source_names = self._get_source_names()

        # Move to CPU and convert to numpy
        sources_np = sources.squeeze(0).cpu().numpy()

        vocals_idx = source_names.index("vocals") if "vocals" in source_names else -1
        if vocals_idx < 0:
            raise RuntimeError("Model did not produce a 'vocals' stem.")

        vocals = sources_np[vocals_idx]  # shape: (channels, samples)

        # Instrumental = sum of all non-vocal stems
        instrumental = np.zeros_like(vocals)
        for i, name in enumerate(source_names):
            if name != "vocals":
                instrumental += sources_np[i]

        return {
            "vocals": vocals,
            "instrumental": instrumental,
        }

    def _numpy_to_tensor(self, audio: np.ndarray) -> torch.Tensor:
        """Convert numpy audio to a float32 PyTorch tensor."""
        return torch.from_numpy(audio.astype(np.float32))

    def _get_source_names(self) -> list[str]:
        """Get the ordered list of source names from the model."""
        if hasattr(self._model, "sources"):
            return list(self._model.sources)
        # Default Demucs v4 order
        return ["drums", "bass", "other", "vocals"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_separator(
    model_name: str = "htdemucs",
    device: torch.device | None = None,
) -> SeparatorBase:
    """
    Factory function to create a source separator.

    Args:
        model_name: Name of the separation model.
        device: Processing device. Auto-detected if None.

    Returns:
        A SeparatorBase implementation.
    """
    return DemucsSeparator(model_name=model_name, device=device)
