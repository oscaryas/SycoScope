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

# Which sibling module remains a flat (non-package) file under
# tool_calling/tasks/sycophancy/ -- these are OUT of this migration's scope,
# so their bare-import style is intentional, not a leftover.
SIBLING_FLAT_IMPORTS = {
    "moral_sycophancy_judge",
    "social_sycophancy_judge",
    "cross_dataset_generalization",
}


class DriverImportPathTests(unittest.TestCase):
    """Focused import/path tests for the seven steering-free probe drivers
    moved from building-agent's tool_calling/tasks/sycophancy/scripts/ into
    probing/probe/ (no renaming). Confirms:
      - each driver imports cleanly under its new location
      - REPO_ROOT / SYCOPHANCY_DIR path constants resolve correctly from the
        new probing/probe/ location (SYCOPHANCY_DIR must still point at the
        ORIGINAL tool_calling/tasks/sycophancy/ directory, since results/
        data was not moved -- only the driver scripts were)
      - the import-mapping table was applied (sycophancy_model_registry ->
        utils.model_registry, sycophancy_probes -> probing.probe.baseline_probes,
        sycophancy_dim -> probing.probe.dim)
      - the still-in-place tool_calling/tasks/sycophancy/ flat-file imports
        (moral_sycophancy_judge, social_sycophancy_judge,
        cross_dataset_generalization) are left as bare imports, since those
        modules are out of this migration's scope
    """

    def test_all_drivers_resolve_repo_root_and_sycophancy_dir(self):
        for driver in ALL_DRIVERS:
            with self.subTest(driver=driver.__name__):
                self.assertEqual(driver.REPO_ROOT, REPO_ROOT)
                self.assertEqual(driver.SYCOPHANCY_DIR, SYCOPHANCY_DIR)
                self.assertTrue(driver.SYCOPHANCY_DIR.is_dir())

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
        # Out-of-scope sibling modules stay flat-imported.
        self.assertIn("from moral_sycophancy_judge import", source)
        self.assertIn("from cross_dataset_generalization import", source)

    def test_oeq_probe_pipeline_imports_baseline_probes_by_new_name(self):
        source = inspect.getsource(oeq_probe_pipeline)
        self.assertIn("from probing.probe.baseline_probes import (", source)
        self.assertNotIn("from sycophancy_probes import", source)
        self.assertIn("from social_sycophancy_judge import", source)
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


if __name__ == "__main__":
    unittest.main()
