"""
src/classification/intent_loader.py
=====================================
Load and validate the intent taxonomy from intents.yaml.

This module is the single source of truth for intent configuration.
All other components (classifier, evaluation pipelines, labelling tools) should
import intents through this module rather than reading the YAML directly.

USAGE
-----
    from src.classification.intent_loader import load_intents, get_intent_names

    intents = load_intents()             # list of intent dicts
    names = get_intent_names()           # list of intent name strings
    fallback = get_fallback_intent()     # str, e.g. "other"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Default path to the taxonomy file
# ---------------------------------------------------------------------------
_DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent / "intents.yaml"

# Required fields that every intent dict must contain
_REQUIRED_INTENT_FIELDS = {
    "name",
    "display_name",
    "definition",
    "auto_handle",
    "escalation_risk",
    "examples",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class IntentConfigError(ValueError):
    """Raised when the intent configuration file is malformed."""


def _validate(config: dict) -> None:
    """
    Validate a loaded intent config dict.

    Checks:
      - "intents" key exists and is a non-empty list
      - No duplicate intent names
      - Every intent has the required fields
      - "metadata.fallback_intent" exists and refers to a known intent
    """
    if "intents" not in config:
        raise IntentConfigError("Missing 'intents' key in taxonomy file.")

    intents = config["intents"]
    if not isinstance(intents, list) or len(intents) == 0:
        raise IntentConfigError("'intents' must be a non-empty list.")

    names_seen: set[str] = set()
    for intent in intents:
        name = intent.get("name")
        if not name:
            raise IntentConfigError(
                f"Intent entry is missing 'name': {intent}"
            )
        if name in names_seen:
            raise IntentConfigError(
                f"Duplicate intent name: '{name}'"
            )
        names_seen.add(name)

        missing = _REQUIRED_INTENT_FIELDS - set(intent.keys())
        if missing:
            raise IntentConfigError(
                f"Intent '{name}' is missing required fields: {sorted(missing)}"
            )

        definition = intent.get("definition", "")
        if not isinstance(definition, str) or not definition.strip():
            raise IntentConfigError(
                f"Intent '{name}' has an empty or invalid 'definition'."
            )

    # Check fallback intent
    metadata = config.get("metadata", {})
    fallback = metadata.get("fallback_intent")
    if not fallback:
        raise IntentConfigError(
            "Missing 'metadata.fallback_intent' in taxonomy file."
        )
    if fallback not in names_seen:
        raise IntentConfigError(
            f"fallback_intent '{fallback}' is not in the list of defined intents."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_intents(path: Path | None = None) -> list[dict[str, Any]]:
    """
    Load and validate the intent taxonomy.

    Parameters
    ----------
    path : Path, optional
        Path to the YAML taxonomy file.  Defaults to
        src/classification/intents.yaml.

    Returns
    -------
    list[dict]
        List of intent configuration dicts, one per intent.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    IntentConfigError
        If the file is structurally invalid.
    """
    taxonomy_path = path or _DEFAULT_TAXONOMY_PATH
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"Intent taxonomy not found at {taxonomy_path}"
        )

    with open(taxonomy_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate(config)
    return config["intents"]


def get_intent_names(path: Path | None = None) -> list[str]:
    """
    Return an ordered list of intent name strings.

    Parameters
    ----------
    path : Path, optional
        Path to the taxonomy YAML.

    Returns
    -------
    list[str]
        Intent names in the order they appear in the file.
    """
    return [intent["name"] for intent in load_intents(path)]


def get_fallback_intent(path: Path | None = None) -> str:
    """
    Return the name of the fallback / catch-all intent.

    Parameters
    ----------
    path : Path, optional
        Path to the taxonomy YAML.

    Returns
    -------
    str
        The fallback intent name (e.g. "other").
    """
    taxonomy_path = path or _DEFAULT_TAXONOMY_PATH
    with open(taxonomy_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _validate(config)
    return config["metadata"]["fallback_intent"]


def get_intent_by_name(name: str, path: Path | None = None) -> dict[str, Any]:
    """
    Return the full config dict for a single intent by name.

    Parameters
    ----------
    name : str
        Exact intent name (e.g. "battery_performance_issue").
    path : Path, optional
        Path to the taxonomy YAML.

    Returns
    -------
    dict
        Full intent configuration dict.

    Raises
    ------
    KeyError
        If the intent name is not found.
    """
    intents = {i["name"]: i for i in load_intents(path)}
    if name not in intents:
        raise KeyError(
            f"Intent '{name}' not found. Available: {sorted(intents)}"
        )
    return intents[name]
