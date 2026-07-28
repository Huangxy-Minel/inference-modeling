"""Model-specific cost estimators (one file per model family).

Each estimator subclasses ModelCostEstimator and returns (FLOPs, bytes) for
the attention and FFN stages of one decode/prefill step.
"""

from .base import MODEL_REGISTRY, ModelCostEstimator, register_model

# Importing the concrete modules registers them into MODEL_REGISTRY.
from . import deepseek_v3  # noqa: F401
from . import deepseek_v4  # noqa: F401
from . import glm_v5       # noqa: F401
from . import glm_v52      # noqa: F401


def get_model(name: str) -> ModelCostEstimator:
    """Look up a model estimator by name (e.g. 'deepseek-v3.2')."""
    if name not in MODEL_REGISTRY:
        avail = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {avail}")
    return MODEL_REGISTRY[name]


__all__ = ["MODEL_REGISTRY", "ModelCostEstimator", "register_model", "get_model"]
