"""HaloScope: unsupervised hallucination detection from LLM activations."""

from .core import LatentSubspace, SubspaceConfig
from .pipeline import HaloScope, SearchConfig

__all__ = ["HaloScope", "LatentSubspace", "SearchConfig", "SubspaceConfig"]
__version__ = "0.1.0"

