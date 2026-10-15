"""Evidence-Aware Three-Way Semantic Contrastive Decoding."""
from .config import ETConfig
from .core import decode_step

__all__ = ["ETConfig", "decode_step"]
__version__ = "0.1.0"

