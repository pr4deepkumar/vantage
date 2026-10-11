"""Shared helpers for running open models locally with PyTorch and Hugging Face transformers."""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, nullcontext
from importlib.util import find_spec
from typing import Any

INSTALL_HINT = 'Local models need the optional dependencies: pip install -e ".[local]"'


class LocalModelError(RuntimeError):
    """A local model can't run: its dependencies are missing or it failed to load."""


def local_models_installed() -> bool:
    """Whether the ``local`` extra is installed, checked without importing torch (which is slow)."""
    return all(find_spec(name) is not None for name in ("torch", "transformers"))


def import_torch() -> Any:
    """Import torch (and check transformers is present), or explain how to install them."""
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as exc:
        raise LocalModelError(INSTALL_HINT) from exc
    return torch


def pick_device(torch: Any, preference: str = "auto") -> str:
    """``auto`` picks CUDA if available, then the Apple Silicon GPU (MPS), else the CPU."""
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_ACCELERATOR_LOCK = threading.Lock()


def device_lock(device: str) -> AbstractContextManager[Any]:
    """Serialize work on a GPU. The chat model and the NLI model share it, and PyTorch's MPS
    backend isn't safe to drive from several threads at once; CPU work runs in parallel."""
    return nullcontext() if device == "cpu" else _ACCELERATOR_LOCK


def load_failure(model_name: str, exc: Exception) -> LocalModelError:
    return LocalModelError(
        f"Could not load the local model {model_name} ({exc}). The first run downloads it from "
        "the Hugging Face Hub, which needs internet access once."
    )
