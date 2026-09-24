import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probing.probe.baseline_training as baseline_training
import probing.probe.baseline_probes as baseline_probes
import probing.probe.train_baseline_probes as train_baseline_probes
import probing.probe.extract_baseline_activations as extract_baseline_activations
import probing.probe.dim as dim


REPO_ROOT = Path(__file__).resolve().parents[1]


class CoreModuleImportPathTests(unittest.TestCase):
    """Focused import/path tests for the five core modules moved from
    building-agent's tool_calling/tasks/sycophancy/{sycophancy_probes.py,
    sycophancy_dim.py, pipeline_scripts/{training.py, probing/train_probes.py,
    activations/extract_activations.py}} into probing/probe/. Confirms the
    import-mapping table (pipeline_scripts.* -> probing.utils.baseline_probes_*,
    sycophancy_model_registry -> utils.model_registry, sycophancy_probes ->
    probing.probe.baseline_probes, sycophancy_dim -> probing.probe.dim) was
    actually applied, and that path-resolution constants point at the real
    repo root from the new probing/probe/ location.
    """

    def test_extract_baseline_activations_repo_root_resolves(self):
        self.assertEqual(extract_baseline_activations.REPO_ROOT, REPO_ROOT)
        self.assertTrue((extract_baseline_activations.REPO_ROOT / "probing").is_dir())
        self.assertTrue((extract_baseline_activations.REPO_ROOT / "utils").is_dir())

    def test_train_baseline_probes_repo_root_resolves(self):
        self.assertEqual(train_baseline_probes.REPO_ROOT, REPO_ROOT)

    def test_extract_baseline_activations_imports_shared_cache_and_datasets(self):
        source = inspect.getsource(extract_baseline_activations)
        self.assertIn("from probing.utils.probes_cache import save_activation_cache", source)
        self.assertIn("from probing.utils.probes_datasets import normalize_records", source)
        self.assertIn("from utils.model_registry import get_model_config, register_hooks, remove_hooks", source)
        self.assertNotIn("pipeline_scripts", source)
        self.assertNotIn("sycophancy_model_registry", source)

    def test_extract_baseline_activations_still_bootstraps_utils_model_locally(self):
        # utils.model is not part of this migration (root-level, unmoved) --
        # the function-local import inside extract() should be untouched.
        source = inspect.getsource(extract_baseline_activations.extract)
        self.assertIn("from utils.model import cleanup, load_model_and_tokenizer", source)

    def test_baseline_training_imports_common_from_canonical_location(self):
        source = inspect.getsource(baseline_training)
        self.assertIn("from probing.utils.probes_common import t_confidence_interval", source)
        self.assertNotIn("from .common import", source)

    def test_train_baseline_probes_imports_from_canonical_locations(self):
        source = inspect.getsource(train_baseline_probes)
        self.assertIn("from probing.utils.probes_cache import load_activation_cache", source)
        self.assertIn("from probing.utils.probes_common import json_dump, parse_dataset_spec", source)
        self.assertIn("from probing.utils.probes_datasets import prepare_cache", source)
        self.assertIn("from probing.probe.baseline_training import", source)
        self.assertNotIn("pipeline_scripts", source)

    def test_baseline_probes_collect_activations_imports_model_registry_from_canonical_location(self):
        source = inspect.getsource(baseline_probes.collect_activations)
        self.assertIn("from utils.model_registry import register_hooks, remove_hooks", source)
        self.assertNotIn("sycophancy_model_registry", source)

    def test_baseline_probes_collect_sentence_activations_imports_model_registry_from_canonical_location(self):
        source = inspect.getsource(baseline_probes.collect_sentence_activations)
        self.assertIn("from utils.model_registry import register_hooks, remove_hooks", source)
        self.assertIn("from utils.inference import build_chat_prompt", source)
        self.assertNotIn("sycophancy_model_registry", source)

    def test_baseline_probes_docstring_no_longer_references_stale_module_name(self):
        module_doc = baseline_probes.__doc__ or ""
        self.assertNotIn("sycophancy_probes", module_doc)
        self.assertIn("probing.probe.baseline_probes", module_doc)

    def test_dim_module_docstring_updated_for_sibling_rename(self):
        module_doc = dim.__doc__ or ""
        # sycophancy_probes.py was renamed to baseline_probes.py; the DIM
        # module's docstring cross-references it by name and should track
        # the rename.
        self.assertNotIn("sycophancy_probes.py", module_doc)
        self.assertIn("baseline_probes.py", module_doc)
        # sycophancy_steering.py is NOT part of this migration (still lives
        # at its original path on building-agent / will stay put) -- that
        # reference is legitimate and must NOT be rewritten.
        self.assertIn("sycophancy_steering.py", module_doc)

    def test_dim_module_has_zero_steering_imports(self):
        source = inspect.getsource(dim)
        self.assertNotIn("import sycophancy_steering", source)
        self.assertNotIn("ActivationSteerer", source)

    def test_baseline_probes_module_has_zero_steering_imports(self):
        source = inspect.getsource(baseline_probes)
        self.assertNotIn("import sycophancy_steering", source)
        self.assertNotIn("ActivationSteerer", source)


class LinearProbeDuplicationTests(unittest.TestCase):
    """Documents the relationship between the two independent `LinearProbe`
    classes this extraction surfaced:

      - probing.probe.baseline_training.LinearProbe   (from pipeline_scripts/training.py)
      - probing.probe.baseline_probes.LinearProbe      (from sycophancy_probes.py)

    Per the migration plan, these are NOT consolidated here even though they
    turn out to be structurally and behaviorally identical (see findings
    below) -- that's a decision for a later task, once this is documented.

    Finding: both classes are `nn.Linear(input_dim, 1)` wrapped in an
    identical forward (`self.linear(input).squeeze(-1)`), just with a
    differently-named forward parameter (`values` vs `x`) and defined in
    two different modules with no import relationship between them. They
    are drop-in interchangeable: same constructor signature, same
    state_dict() keys, cross-loadable state, and bit-identical output given
    the same weights and input.
    """

    def test_two_independent_class_objects_not_one_importing_the_other(self):
        self.assertIsNot(baseline_training.LinearProbe, baseline_probes.LinearProbe)
        self.assertNotEqual(
            baseline_training.LinearProbe.__module__, baseline_probes.LinearProbe.__module__
        )

    def test_constructor_signature_matches(self):
        sig_a = inspect.signature(baseline_training.LinearProbe.__init__)
        sig_b = inspect.signature(baseline_probes.LinearProbe.__init__)
        self.assertEqual(list(sig_a.parameters), list(sig_b.parameters))
        self.assertEqual(list(sig_a.parameters), ["self", "input_dim"])

    def test_constructed_linear_layer_shape_matches(self):
        probe_a = baseline_training.LinearProbe(16)
        probe_b = baseline_probes.LinearProbe(16)
        self.assertEqual(probe_a.linear.weight.shape, probe_b.linear.weight.shape)
        self.assertEqual(probe_a.linear.weight.shape, (1, 16))
        self.assertEqual(probe_a.linear.bias.shape, probe_b.linear.bias.shape)

    def test_state_dict_keys_match(self):
        probe_a = baseline_training.LinearProbe(8)
        probe_b = baseline_probes.LinearProbe(8)
        self.assertEqual(set(probe_a.state_dict().keys()), {"linear.weight", "linear.bias"})
        self.assertEqual(set(probe_a.state_dict().keys()), set(probe_b.state_dict().keys()))

    def test_state_dict_is_cross_loadable(self):
        """A state_dict produced by one LinearProbe implementation loads
        cleanly into the other -- proof the two class bodies are structurally
        identical (same single `linear` submodule, same shape), not merely
        coincidentally same-named."""
        source_probe = baseline_training.LinearProbe(4)
        with torch.no_grad():
            source_probe.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
            source_probe.linear.bias.copy_(torch.tensor([0.5]))

        target_probe = baseline_probes.LinearProbe(4)
        target_probe.load_state_dict(source_probe.state_dict())  # raises on mismatch

        np.testing.assert_allclose(
            target_probe.linear.weight.detach().numpy(),
            source_probe.linear.weight.detach().numpy(),
        )
        np.testing.assert_allclose(
            target_probe.linear.bias.detach().numpy(),
            source_probe.linear.bias.detach().numpy(),
        )

    def test_forward_scoring_is_bit_identical_given_same_weights_and_input(self):
        """Both forward()s reduce to the same computation
        (self.linear(input).squeeze(-1)), just with differently-named
        parameters (`values` in baseline_training, `x` in baseline_probes).
        Confirm the actual numeric output is identical, not just
        shape-compatible."""
        torch.manual_seed(0)
        weight = torch.randn(1, 6)
        bias = torch.randn(1)
        x = torch.randn(5, 6)

        probe_a = baseline_training.LinearProbe(6)
        with torch.no_grad():
            probe_a.linear.weight.copy_(weight)
            probe_a.linear.bias.copy_(bias)

        probe_b = baseline_probes.LinearProbe(6)
        with torch.no_grad():
            probe_b.linear.weight.copy_(weight)
            probe_b.linear.bias.copy_(bias)

        with torch.no_grad():
            out_a = probe_a(x)
            out_b = probe_b(x)

        self.assertEqual(out_a.shape, (5,))  # squeeze(-1) applied in both
        np.testing.assert_allclose(out_a.numpy(), out_b.numpy(), rtol=1e-6, atol=1e-6)

    def test_scoring_helpers_around_each_probe_agree_on_predictions(self):
        """baseline_training.scores() and baseline_probes._probe_scores() are
        each module's own way of getting raw logits out of its LinearProbe;
        confirm they agree when pointed at cross-loaded, identical probes."""
        torch.manual_seed(1)
        probe_a = baseline_training.LinearProbe(3)
        probe_b = baseline_probes.LinearProbe(3)
        probe_b.load_state_dict(probe_a.state_dict())

        X = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32)

        logits_a = baseline_training.scores(probe_a, X)
        logits_b = baseline_probes._probe_scores(probe_b, X)

        np.testing.assert_allclose(logits_a, logits_b, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
