import os
from pathlib import Path
import sys
import unittest

# Avoid a real network round-trip in test_get_model_config_unknown_model_raises:
# get_model_config's auto-detect fallback calls AutoConfig.from_pretrained,
# which should fail fast (and be caught) rather than hang, offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_registry import (
    MODELS,
    REQUIRED_FIELDS,
    get_model_config,
    _extract_layer_idx,
)


class ModelRegistryImportPathTests(unittest.TestCase):
    """Focused import/path test for utils/model_registry.py (moved from
    tool_calling/tasks/sycophancy/sycophancy_model_registry.py on
    building-agent). This module has no filesystem path constants of its
    own, so the check here is purely: does it import cleanly under its new
    canonical name and does its behavior still hold."""

    def test_imports_under_canonical_name(self):
        self.assertTrue(len(MODELS) > 0)
        self.assertIn("meta-llama/Meta-Llama-3-8B-Instruct", MODELS)

    def test_every_registry_entry_has_required_fields(self):
        for name, config in MODELS.items():
            missing = REQUIRED_FIELDS - set(config)
            self.assertFalse(missing, f"{name} missing fields: {missing}")

    def test_get_model_config_known_model_returns_copy(self):
        config = get_model_config("meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(config["n_layers"], 32)
        config["n_layers"] = -1
        self.assertEqual(MODELS["meta-llama/Llama-3.1-8B-Instruct"]["n_layers"], 32)

    def test_get_model_config_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            get_model_config("not-a-real-org/not-a-real-model-xyz")

    def test_extract_layer_idx(self):
        self.assertEqual(_extract_layer_idx("model.layers.3.self_attn.o_proj"), 3)
        with self.assertRaises(ValueError):
            _extract_layer_idx("model.embed_tokens")


if __name__ == "__main__":
    unittest.main()
