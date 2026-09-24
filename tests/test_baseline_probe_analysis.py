"""Smoke tests for the baseline-probe shared analysis code moved from
building-agent's tool_calling/tasks/sycophancy/{pipeline_scripts/visualizations/*,
scripts/{bootstrap_nc1_pipeline,mixture_holdout_eval_pipeline,plot_all_probes,
plot_category_nc1,plot_nc1_bootstrap_ci,plot_probe_transfer_summary,
plot_sycophancy_results,rq1_gate1_geometry,truthfulqa_probe_transfer_pipeline}.py,
sycophancy_compare.py} into probing/analyze_probes/.

These tests only import modules and parse --help output (or verify required
CLI arguments are enforced) -- no model loading, no GPU, no network calls.
Any heavier import (torch, transformers) module-level in a couple of these
files is lightweight enough for --help to still exit immediately at
argparse.parse_args(), before any model/tokenizer is actually loaded.
"""
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import probing.analyze_probes.plot_judge_rates as plot_judge_rates
import probing.analyze_probes.plot_probe_accuracy as plot_probe_accuracy
import probing.analyze_probes.plot_steering_curve as plot_steering_curve
import probing.analyze_probes.bootstrap_nc1_pipeline as bootstrap_nc1_pipeline
import probing.analyze_probes.mixture_holdout_eval_pipeline as mixture_holdout_eval_pipeline
import probing.analyze_probes.plot_all_probes as plot_all_probes
import probing.analyze_probes.plot_category_nc1 as plot_category_nc1
import probing.analyze_probes.plot_nc1_bootstrap_ci as plot_nc1_bootstrap_ci
import probing.analyze_probes.plot_probe_transfer_summary as plot_probe_transfer_summary
import probing.analyze_probes.plot_sycophancy_results as plot_sycophancy_results
import probing.analyze_probes.rq1_gate1_geometry as rq1_gate1_geometry
import probing.analyze_probes.truthfulqa_probe_transfer_pipeline as truthfulqa_probe_transfer_pipeline
import probing.analyze_probes.sycophancy_compare as sycophancy_compare

ALL_MODULES = [
    plot_judge_rates, plot_probe_accuracy, plot_steering_curve, bootstrap_nc1_pipeline,
    mixture_holdout_eval_pipeline, plot_all_probes, plot_category_nc1, plot_nc1_bootstrap_ci,
    plot_probe_transfer_summary, plot_sycophancy_results, rq1_gate1_geometry,
    truthfulqa_probe_transfer_pipeline, sycophancy_compare,
]

# (dotted module path for `python -m`, extra args needed just to reach --help
# cleanly -- none of these actually need extra args since --help short-circuits
# argparse before required-arg enforcement)
CLI_MODULES = [
    "probing.analyze_probes.plot_judge_rates",
    "probing.analyze_probes.plot_probe_accuracy",
    "probing.analyze_probes.plot_steering_curve",
    "probing.analyze_probes.bootstrap_nc1_pipeline",
    "probing.analyze_probes.mixture_holdout_eval_pipeline",
    "probing.analyze_probes.plot_all_probes",
    "probing.analyze_probes.plot_category_nc1",
    "probing.analyze_probes.plot_nc1_bootstrap_ci",
    "probing.analyze_probes.plot_probe_transfer_summary",
    "probing.analyze_probes.plot_sycophancy_results",
    "probing.analyze_probes.rq1_gate1_geometry",
    "probing.analyze_probes.truthfulqa_probe_transfer_pipeline",
    "probing.analyze_probes.sycophancy_compare",
]

# Modules with no CLI at all (sycophancy_compare and plot_steering_curve DO have
# argparse; every one of the 13 does). Kept for symmetry with other task's test
# files, empty here.
NO_CLI_MODULES = []


def run_help(dotted_module: str):
    return subprocess.run(
        [sys.executable, "-m", dotted_module, "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )


def run_without_required(dotted_module: str, args=()):
    return subprocess.run(
        [sys.executable, "-m", dotted_module, *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )


class AnalysisModuleImportTests(unittest.TestCase):
    """All 13 modules import cleanly under their new probing/analyze_probes/ location."""

    def test_no_stale_pipeline_scripts_or_old_name_references(self):
        for module in ALL_MODULES:
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("pipeline_scripts", source)
                self.assertNotIn("sycophancy_model_registry", source)
                self.assertNotIn("from sycophancy_probes import", source)
                self.assertNotIn("from sycophancy_dim import", source)
                self.assertNotIn("from sycophancy_data import", source)
                self.assertNotIn("import mixture_residual_probe_pipeline\n", source)

    def test_no_module_has_any_executable_steering_dependency(self):
        """A historical string or plot label (e.g. plot_steering_curve.py's
        axis labels, or a comment referencing the unmoved sycophancy_steering.py)
        is fine; an executable import or ActivationSteerer(...) call is not."""
        for module in ALL_MODULES:
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("import sycophancy_steering", source)
                executable_hits = [
                    line for line in source.splitlines()
                    if "ActivationSteerer" in line and not line.strip().startswith("#")
                ]
                self.assertEqual(executable_hits, [], f"{module.__name__}: {executable_hits}")

    def test_plot_steering_curve_only_plots_precomputed_summaries(self):
        """plot_steering_curve.py's name suggests it might run live steering --
        confirm every "steering" occurrence is just a plot label/docstring word
        (never sycophancy_steering or ActivationSteerer), and that it has no
        cross-package imports or sys.path bootstrap at all: it only reads a
        static --input summary.json produced by some earlier steering-sweep run."""
        source = inspect.getsource(plot_steering_curve)
        self.assertNotIn("sycophancy_steering", source)
        self.assertNotIn("ActivationSteerer", source)
        self.assertNotIn("import sys", source)
        self.assertNotIn("sys.path", source)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import transformers", source)
        self.assertIn("--input", source)
        self.assertIn("required=True", source)

    def test_repo_root_resolves_for_every_module_that_defines_one(self):
        for module in (plot_judge_rates, plot_probe_accuracy, bootstrap_nc1_pipeline,
                        mixture_holdout_eval_pipeline, rq1_gate1_geometry,
                        truthfulqa_probe_transfer_pipeline):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.REPO_ROOT, REPO_ROOT)
                self.assertTrue((module.REPO_ROOT / "probing").is_dir())
                self.assertTrue((module.REPO_ROOT / "utils").is_dir())

    def test_truthfulqa_probe_transfer_keeps_sycophancy_dir_on_sys_path(self):
        """This module has a function-local `from mixture_dim_pipeline import
        extract_all_activations` -- mixture_dim_pipeline.py is application-owned
        and out of this migration's scope, so SYCOPHANCY_DIR must stay on
        sys.path (even though the sibling file isn't present in every checkout
        of tool_calling/tasks/sycophancy/scripts/ -- that's a pre-existing,
        out-of-scope gap, not something this move introduced)."""
        self.assertEqual(
            truthfulqa_probe_transfer_pipeline.SYCOPHANCY_DIR,
            REPO_ROOT / "tool_calling" / "tasks" / "sycophancy",
        )
        source = inspect.getsource(truthfulqa_probe_transfer_pipeline)
        self.assertIn("from mixture_dim_pipeline import extract_all_activations", source)

    def test_mixture_holdout_eval_resolves_probes_core_fit_probe(self):
        """probing/probe/ is retired: the held-out eval fits with the sklearn
        probes_core.fit_probe, not the old torch _fit_probe."""
        source = inspect.getsource(mixture_holdout_eval_pipeline)
        self.assertIn("from probing.analyze_probes.probes_core import fit_probe, score", source)
        import probing.analyze_probes.probes_core as probes_core
        self.assertIs(mixture_holdout_eval_pipeline.fit_probe, probes_core.fit_probe)

    def test_pipelines_sharing_collect_residual_only_use_package_qualified_import(self):
        for module in (bootstrap_nc1_pipeline, mixture_holdout_eval_pipeline,
                        rq1_gate1_geometry, truthfulqa_probe_transfer_pipeline):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                if "collect_residual_only" in source:
                    self.assertIn(
                        "from probing.analyze_probes.probes_core import collect_residual_only",
                        source,
                    )

    def test_no_module_silently_defaults_into_application_owned_results_tree(self):
        """Per the audit instruction: no argparse default (or bare module-level
        constant) may silently point at tool_calling/tasks/sycophancy/results/
        or a computed SYCOPHANCY_DIR/"results" path -- either a required CLI
        arg, or an explicit non-app path, but never a silent default."""
        for module in ALL_MODULES:
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("SYCOPHANCY_DIR / \"results\"", source)
                self.assertNotIn('SYCOPHANCY_DIR / "results"', source)
                self.assertNotIn("default=str(SYCOPHANCY_DIR", source)
                self.assertNotRegex(source, r'RESULTS_DIR\s*=\s*Path\(__file__\)')


class AnalysisHelpSmokeTests(unittest.TestCase):
    """`--help` parses and exits 0 for every one of the 13 modules, without
    loading any model, tokenizer, or contacting a network service."""

    def test_help_exits_cleanly_for_every_module(self):
        for dotted in CLI_MODULES:
            with self.subTest(module=dotted):
                result = run_help(dotted)
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("usage", result.stdout.lower())


class AnalysisRequiredArgTests(unittest.TestCase):
    """Application-owned output/input defaults (into tool_calling/tasks/sycophancy's
    results/ tree) were replaced with required CLI arguments -- verify that
    omitting them now fails argument parsing (exit code 2)."""

    def test_bootstrap_nc1_pipeline_requires_input_and_output_dir(self):
        result = run_without_required("probing.analyze_probes.bootstrap_nc1_pipeline", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--input", result.stderr)
        self.assertIn("--output-dir", result.stderr)

    def test_mixture_holdout_eval_requires_mixture_judged_output_dir(self):
        result = run_without_required("probing.analyze_probes.mixture_holdout_eval_pipeline", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--mixture", result.stderr)
        self.assertIn("--judged", result.stderr)
        self.assertIn("--output-dir", result.stderr)

    def test_plot_all_probes_requires_results_dir(self):
        result = run_without_required("probing.analyze_probes.plot_all_probes", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--results-dir", result.stderr)

    def test_plot_category_nc1_requires_probing_dir(self):
        result = run_without_required("probing.analyze_probes.plot_category_nc1", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--probing-dir", result.stderr)

    def test_plot_nc1_bootstrap_ci_requires_probing_dir(self):
        result = run_without_required("probing.analyze_probes.plot_nc1_bootstrap_ci", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--probing-dir", result.stderr)

    def test_plot_probe_transfer_summary_requires_results_dir(self):
        result = run_without_required("probing.analyze_probes.plot_probe_transfer_summary", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--results-dir", result.stderr)

    def test_plot_sycophancy_results_requires_results_dir(self):
        result = run_without_required("probing.analyze_probes.plot_sycophancy_results", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--results-dir", result.stderr)

    def test_rq1_gate1_geometry_requires_output_dir_and_data_paths(self):
        result = run_without_required("probing.analyze_probes.rq1_gate1_geometry", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output-dir", result.stderr)
        self.assertIn("--moral-judged-path", result.stderr)
        self.assertIn("--social-pooled-path", result.stderr)
        self.assertIn("--praise-checkpoint-path", result.stderr)

    def test_truthfulqa_probe_transfer_requires_generations_dir_probe_dir_output_dir(self):
        result = run_without_required("probing.analyze_probes.truthfulqa_probe_transfer_pipeline", [])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--generations-dir", result.stderr)
        self.assertIn("--probe-dir", result.stderr)
        self.assertIn("--output-dir", result.stderr)

    def test_truthfulqa_probe_transfer_dim_dir_required_only_at_runtime_for_dim_format(self):
        """--dim-dir has no argparse default, but is only enforced at runtime
        (inside main()) when --direction-format dim is selected, since --probe-dir
        is the one that's always required regardless of format."""
        parser_help = run_help("probing.analyze_probes.truthfulqa_probe_transfer_pipeline")
        self.assertIn("--dim-dir", parser_help.stdout)


if __name__ == "__main__":
    unittest.main()
