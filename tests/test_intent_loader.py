"""
tests/test_intent_loader.py
============================
Tests for src/classification/intent_loader.py

Tests verify:
  - Successful load of the real intents.yaml
  - Intent names are unique
  - Every intent has required fields and a non-empty definition
  - The fallback intent exists in the list
  - get_intent_names() returns a list of strings
  - get_fallback_intent() returns the correct name
  - get_intent_by_name() retrieves the correct dict
  - get_intent_by_name() raises KeyError for unknown names
  - Validation catches duplicate names (synthetic config)
  - Validation catches missing fallback (synthetic config)
  - Validation catches missing required fields (synthetic config)
"""

import pytest
import yaml
from pathlib import Path

from src.classification.intent_loader import (
    IntentConfigError,
    _validate,
    get_fallback_intent,
    get_intent_by_name,
    get_intent_names,
    load_intents,
)

INTENTS_YAML = Path(__file__).resolve().parent.parent / "src" / "classification" / "intents.yaml"


# ---------------------------------------------------------------------------
# Tests against the real intents.yaml
# ---------------------------------------------------------------------------

class TestRealIntentsYaml:
    """Tests that load the actual committed intents.yaml file."""

    def test_file_exists(self):
        assert INTENTS_YAML.exists(), f"intents.yaml not found at {INTENTS_YAML}"

    def test_load_returns_list(self):
        intents = load_intents(INTENTS_YAML)
        assert isinstance(intents, list)
        assert len(intents) > 0

    def test_intent_names_are_unique(self):
        intents = load_intents(INTENTS_YAML)
        names = [i["name"] for i in intents]
        assert len(names) == len(set(names)), "Duplicate intent names found"

    def test_every_intent_has_required_fields(self):
        required = {"name", "display_name", "definition",
                    "auto_handle", "escalation_risk", "examples"}
        intents = load_intents(INTENTS_YAML)
        for intent in intents:
            missing = required - set(intent.keys())
            assert not missing, (
                f"Intent '{intent.get('name')}' is missing: {sorted(missing)}"
            )

    def test_every_definition_is_non_empty(self):
        intents = load_intents(INTENTS_YAML)
        for intent in intents:
            definition = str(intent.get("definition", "")).strip()
            assert definition, f"Intent '{intent['name']}' has an empty definition"

    def test_fallback_intent_exists(self):
        names = get_intent_names(INTENTS_YAML)
        fallback = get_fallback_intent(INTENTS_YAML)
        assert fallback in names, (
            f"fallback_intent '{fallback}' is not in the intent list: {names}"
        )

    def test_get_intent_names_returns_strings(self):
        names = get_intent_names(INTENTS_YAML)
        for name in names:
            assert isinstance(name, str) and name.strip()

    def test_get_intent_by_name_returns_dict(self):
        names = get_intent_names(INTENTS_YAML)
        intent = get_intent_by_name(names[0], INTENTS_YAML)
        assert isinstance(intent, dict)
        assert intent["name"] == names[0]

    def test_get_intent_by_name_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_intent_by_name("this_intent_does_not_exist_xyz", INTENTS_YAML)

    def test_at_least_one_auto_handle_intent(self):
        intents = load_intents(INTENTS_YAML)
        auto = [i for i in intents if i.get("auto_handle") is True]
        assert len(auto) >= 1, "At least one intent should be auto_handle=true"

    def test_fallback_intent_is_not_auto_handle(self):
        """The catch-all 'other' intent should not be auto-handled."""
        intents = load_intents(INTENTS_YAML)
        fallback_name = get_fallback_intent(INTENTS_YAML)
        fallback = next(i for i in intents if i["name"] == fallback_name)
        # Catch-all should not be automatically handled
        assert fallback.get("auto_handle") is False, (
            "The fallback intent should not be auto_handle=true"
        )


# ---------------------------------------------------------------------------
# Validation edge-case tests (using synthetic configs)
# ---------------------------------------------------------------------------

def _make_valid_config(**overrides) -> dict:
    """Build a minimal valid config dict, with optional overrides."""
    base = {
        "intents": [
            {
                "name": "intent_a",
                "display_name": "Intent A",
                "definition": "A valid definition.",
                "inclusion_criteria": ["something"],
                "exclusion_criteria": [],
                "auto_handle": True,
                "escalation_risk": False,
                "examples": ["example tweet"],
            },
            {
                "name": "other",
                "display_name": "Other",
                "definition": "Catch-all.",
                "auto_handle": False,
                "escalation_risk": False,
                "examples": ["unclear"],
            },
        ],
        "metadata": {"fallback_intent": "other"},
    }
    base.update(overrides)
    return base


class TestValidationSynthetic:

    def test_valid_config_passes(self):
        _validate(_make_valid_config())  # should not raise

    def test_missing_intents_key_raises(self):
        with pytest.raises(IntentConfigError, match="Missing 'intents'"):
            _validate({"metadata": {"fallback_intent": "other"}})

    def test_empty_intents_list_raises(self):
        with pytest.raises(IntentConfigError):
            _validate(_make_valid_config(intents=[]))

    def test_duplicate_name_raises(self):
        config = _make_valid_config()
        # Add a duplicate
        config["intents"].append(config["intents"][0].copy())
        with pytest.raises(IntentConfigError, match="Duplicate intent name"):
            _validate(config)

    def test_missing_required_field_raises(self):
        config = _make_valid_config()
        del config["intents"][0]["definition"]
        with pytest.raises(IntentConfigError, match="missing required fields"):
            _validate(config)

    def test_missing_fallback_in_metadata_raises(self):
        config = _make_valid_config()
        del config["metadata"]["fallback_intent"]
        with pytest.raises(IntentConfigError, match="fallback_intent"):
            _validate(config)

    def test_fallback_not_in_list_raises(self):
        config = _make_valid_config()
        config["metadata"]["fallback_intent"] = "nonexistent_intent"
        with pytest.raises(IntentConfigError, match="not in the list"):
            _validate(config)

    def test_empty_definition_raises(self):
        config = _make_valid_config()
        config["intents"][0]["definition"] = "   "
        with pytest.raises(IntentConfigError, match="empty or invalid 'definition'"):
            _validate(config)
