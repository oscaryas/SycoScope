import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probing.probe.aita_dim_pipeline as aita_dim_pipeline
import probing.probe.category_residual_probe_pipeline as category_residual_probe_pipeline
import probing.probe.mixture_category_residual_probe_pipeline as mixture_category_residual_probe_pipeline
import probing.probe.mixture_residual_probe_pipeline as mixture_residual_probe_pipeline
import probing.probe.mixture_residual_probe_pipeline_v2 as mixture_residual_probe_pipeline_v2
import probing.probe.moral_avg_residual_probe_pipeline as moral_avg_residual_probe_pipeline
import probing.probe.oeq_probe_pipeline as oeq_probe_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
SYCOPHANCY_DIR = REPO_ROOT / "tool_calling" / "tasks" / "sycophancy"

ALL_DRIVERS = [
    aita_dim_pipeline,
    category_residual_probe_pipeline,
    mixture_category_residual_probe_pipeline,
    mixture_residual_probe_pipeline,
    mixture_residual_probe_pipeline_v2,
    moral_avg_residual_probe_pipeline,
    oeq_probe_pipeline,
]

# cross_dataset_generalization.py depends on the application-owned live-
# steering machinery, so it stays a flat (non-package) file under
# tool_calling/tasks/sycophancy/ (building-agent only) rather than moving
# into probing/ -- this is OUT of this migration's scope, so its bare-import
# style is intentional, not a leftover. moral_sycophancy_judge.py and
# social_sycophancy_judge.py, by contrast, DID move into the shared
# probing.evaluations.baseline_probes.judge package (Task 4 of the shared
# extraction), so the drivers below import them by that qualified name
# instead of the old flat-file sibling import.
SIBLING_FLAT_IMPORTS = {
    "cross_dataset_generalization",
}


class DriverImportPathTests(unittest.TestCase):
    """Focused import/path tests for the seven steering-free probe drivers
    moved from building-agent's tool_calling/tasks/sycophancy/scripts/ into
    probing/probe/ (no renaming). Confirms:
      - each driver imports cleanly under its new location
      - REPO_ROOT / SYCOPHANCY_DIR path constants resolve correctly from the
        new probing/probe/ location (SYCOPHANCY_DIR still points at
        tool_calling/tasks/sycophancy/ by name, matching building-agent/
        pre-cleanup main -- but on the SAE branch tool_calling/ was removed
        by Task 9 of the branch-split plan, so the directory itself is gone;
        only the still-present drivers' SYCOPHANCY_DIR/results defaults for
        application-owned data are being confirmed here, not the directory's
        existence)
      - the import-mapping table was applied (sycophancy_model_registry ->
        utils.model_registry, sycophancy_probes -> probing.probe.baseline_probes,
        sycophancy_dim -> probing.probe.dim)
      - moral_sycophancy_judge / social_sycophancy_judge are imported from
        their shared probing.evaluations.baseline_probes.judge package home;
        cross_dataset_generalization.py remains a tool_calling-only, flat,
        try/except-guarded import (see SIBLING_FLAT_IMPORTS), since it is
        steering-dependent and out of this migration's scope
    """

    def test_all_drivers_resolve_repo_root_and_sycophancy_dir(self):
        for driver in ALL_DRIVERS:
            with self.subTest(driver=driver.__name__):
                self.assertEqual(driver.REPO_ROOT, REPO_ROOT)
                self.assertEqual(driver.SYCOPHANCY_DIR, SYCOPHANCY_DIR)
                # NOT asserting SYCOPHANCY_DIR.is_dir(): true on building-agent/
                # pre-cleanup main (tool_calling/ present), false on SAE (Task 9
                # removed tool_calling/ from this branch entirely).

    def test_no_driver_references_stale_pipeline_scripts_path(self):
        for driver in ALL_DRIVERS:
            with self.subTest(driver=driver.__name__):
                source = inspect.getsource(driver)
                self.assertNotIn("pipeline_scripts", source)
                self.assertNotIn("HERE.parents[3]", source)

    def test_no_driver_imports_sycophancy_model_registry_by_old_name(self):
        for driver in ALL_DRIVERS:
            with self.subTest(driver=driver.__name__):
                source = inspect.getsource(driver)
                self.assertNotIn("sycophancy_model_registry", source)
                self.assertIn("from utils.model_registry import get_model_config", source)

    def test_no_driver_has_any_steering_dependency(self):
        for driver in ALL_DRIVERS:
            with self.subTest(driver=driver.__name__):
                source = inspect.getsource(driver)
                self.assertNotIn("sycophancy_steering", source)
                self.assertNotIn("ActivationSteerer", source)

    def test_aita_dim_pipeline_imports_dim_and_baseline_probes_by_new_name(self):
        source = inspect.getsource(aita_dim_pipeline)
        self.assertIn("from probing.probe.baseline_probes import collect_activations, bootstrap_ci", source)
        self.assertIn("from probing.probe.dim import (", source)
        self.assertNotIn("from sycophancy_probes import", source)
        self.assertNotIn("from sycophancy_dim import", source)
        # moral_sycophancy_judge.py moved into the shared judge package.
        self.assertIn(
            "from probing.evaluations.baseline_probes.judge.moral_sycophancy_judge import",
            source,
        )
        self.assertNotIn("from moral_sycophancy_judge import", source)
        # cross_dataset_generalization.py is out-of-scope and stays flat-imported.
        self.assertIn("from cross_dataset_generalization import", source)

    def test_oeq_probe_pipeline_imports_baseline_probes_by_new_name(self):
        source = inspect.getsource(oeq_probe_pipeline)
        self.assertIn("from probing.probe.baseline_probes import (", source)
        self.assertNotIn("from sycophancy_probes import", source)
        # social_sycophancy_judge.py moved into the shared judge package.
        self.assertIn(
            "from probing.evaluations.baseline_probes.judge.social_sycophancy_judge import",
            source,
        )
        self.assertNotIn("from social_sycophancy_judge import", source)
        # cross_dataset_generalization.py is out-of-scope and stays flat-imported.
        self.assertIn("from cross_dataset_generalization import", source)

    def test_pipelines_sharing_collect_residual_only_use_package_qualified_import(self):
        """category_residual_probe_pipeline.py, mixture_category_residual_probe_pipeline.py,
        mixture_residual_probe_pipeline_v2.py and moral_avg_residual_probe_pipeline.py all
        import collect_residual_only from mixture_residual_probe_pipeline.py -- another one
        of the 12 moved files, now a sibling under probing/probe/. Confirm the import was
        rewritten to be package-qualified rather than relying on a bare sibling import."""
        for driver in (
            category_residual_probe_pipeline,
            mixture_category_residual_probe_pipeline,
            mixture_residual_probe_pipeline_v2,
            moral_avg_residual_probe_pipeline,
        ):
            with self.subTest(driver=driver.__name__):
                source = inspect.getsource(driver)
                self.assertIn(
                    "from probing.probe.mixture_residual_probe_pipeline import collect_residual_only",
                    source,
                )
                self.assertIs(
                    driver.collect_residual_only,
                    mixture_residual_probe_pipeline.collect_residual_only,
                )

    def test_mixture_residual_probe_pipeline_collect_residual_only_uses_shared_pool(self):
        # collect_residual_only relies on baseline_probes._pool for its pooling
        # convention -- confirm that import survived the move.
        source = inspect.getsource(mixture_residual_probe_pipeline)
        self.assertIn("from probing.probe.baseline_probes import _pool, train_residual_probes, save_probe_results", source)

    def test_output_dir_defaults_still_point_at_original_results_tree(self):
        """Per this repo's established convention (see probing/data/build_llama31_subset.py
        pointing at prompt_probes/results/, not a new probing/-relative results dir),
        moved driver scripts keep their --output-dir/--input defaults pointed at the
        ORIGINAL tool_calling/tasks/sycophancy/results/ tree, since the actual result data
        was not moved as part of this extraction."""
        source = inspect.getsource(mixture_residual_probe_pipeline.main)
        self.assertIn("SYCOPHANCY_DIR / \"results\"", source)


class BuildLabelsCallSiteTests(unittest.TestCase):
    """Regression coverage for a bug that slipped past every other test in this
    file: aita_dim_pipeline.build_labels() and oeq_probe_pipeline.build_labels()
    called the relocated generate_moral_sycophancy_labels /
    generate_social_sycophancy_labels without input_path. That parameter is
    required (no default) on the shared probing.evaluations.baseline_probes.judge
    versions of these functions (Task 4 of this plan deliberately removed the old
    silent-default-into-SAE/results/ behavior), so the old tool_calling-local
    copies' optional input_path let this call pattern work -- but the shared
    versions raise TypeError: missing 1 required positional argument:
    'input_path'. That TypeError is only reachable by actually calling
    build_labels(); compileall, plain import and --help smoke tests never
    exercise this path. These tests call build_labels() directly (with the
    judge functions monkeypatched to avoid any real model/anthropic-client
    dependency) so a future signature change here is caught without a GPU or
    network access.
    """

    def test_aita_build_labels_binds_and_passes_default_input_path(self):
        captured_kwargs = {}
        real_generate = aita_dim_pipeline.generate_moral_sycophancy_labels

        def fake_generate(*args, **kwargs):
            # Fails loudly (TypeError) if the call site can't bind against the
            # REAL shared function's signature -- e.g. a missing input_path.
            inspect.signature(real_generate).bind(*args, **kwargs)
            captured_kwargs.update(kwargs)
            return {"records": [{"label": 1}, {"label": 0}, {"label": 1}]}

        aita_dim_pipeline.generate_moral_sycophancy_labels = fake_generate
        try:
            result = aita_dim_pipeline.build_labels(tokenizer=object(), n_examples=3)
        finally:
            aita_dim_pipeline.generate_moral_sycophancy_labels = real_generate

        self.assertEqual(captured_kwargs.get("input_path"), aita_dim_pipeline.DEFAULT_INPUT_PATH)
        self.assertEqual(captured_kwargs.get("n_pairs"), 3)
        self.assertEqual(result["n_pos"], 2)
        self.assertEqual(result["n_neg"], 1)
        self.assertEqual(result["n_judged"], 3)

    def test_oeq_build_labels_binds_and_passes_default_input_path(self):
        captured_kwargs = {}
        real_generate = oeq_probe_pipeline.generate_social_sycophancy_labels

        def fake_generate(*args, **kwargs):
            inspect.signature(real_generate).bind(*args, **kwargs)
            captured_kwargs.update(kwargs)
            return {"records": [{"label": 1}, {"label": 0}]}

        oeq_probe_pipeline.generate_social_sycophancy_labels = fake_generate
        try:
            result = oeq_probe_pipeline.build_labels(tokenizer=object(), n_label=2)
        finally:
            oeq_probe_pipeline.generate_social_sycophancy_labels = real_generate

        self.assertEqual(
            captured_kwargs.get("input_path"),
            oeq_probe_pipeline.DEFAULT_RESULTS_DIR / "OEQ.jsonl",
        )
        self.assertEqual(captured_kwargs.get("metric"), "validation")
        self.assertEqual(captured_kwargs.get("n_examples"), 2)
        self.assertEqual(result["n_pos"], 1)
        self.assertEqual(result["n_neg"], 1)


if __name__ == "__main__":
    unittest.main()
