"""
src/classification/baseline_classifier.py
=========================================
Baseline classifier implementations for intent classification.
"""

from __future__ import annotations

from typing import Any, Sequence
from sklearn.dummy import DummyClassifier


class MajorityClassBaseline(DummyClassifier):
    """Simple baseline classifier that constantly predicts the most frequent training class."""

    def __init__(self) -> None:
        super().__init__(strategy="most_frequent")
