"""Evaluation metrics, registered against the package-local registry."""
from __future__ import annotations

from .._registry import get_metric, list_metrics, register_metric

# Importing each module side-effect-registers its metrics via @register_metric(...)
from . import deterministic, perceptual, tool_call  # noqa: F401

__all__ = ["get_metric", "list_metrics", "register_metric"]
