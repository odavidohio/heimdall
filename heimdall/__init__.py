"""
HEIMDALL: Real-Time Hallucination Detection via Coherence Inversion

A framework for detecting hallucinations in Large Language Models through
topological analysis of attention patterns.
"""

__version__ = "1.0.0"
__author__ = "David Ohio"
__license__ = "Apache-2.0"

from .detector import HeimdallDetector
from .model_config import MODEL_CONFIGS, get_model_config

__all__ = [
    "HeimdallDetector",
    "MODEL_CONFIGS",
    "get_model_config",
    "__version__"
]
