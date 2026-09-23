"""Smoke tests for the baseline-probe generation modules moved from
building-agent's tool_calling/tasks/sycophancy/{pipeline_scripts/generations/*,
scripts/{dissociating_sycophancy_generate,moral_generate_openrouter,
social_generate_openrouter,syconbench_fetch_generations,
verify_model_generation_setup}.py} into
probing/evaluations/baseline_probes/generation/.

These tests only import modules and parse --help output (or verify required
CLI arguments are enforced) -- no model loading, no GPU, no network calls,
no OPENROUTER_API_KEY/ANTHROPIC_API_KEY required. Module-level imports in
every one of these files are lightweight (argparse/json/pathlib/etc. plus
shared probing/utils package code); any heavier import (torch, transformers,
anthropic client construction, ActivationSteerer) is deferred inside main()
and is never reached by --help, since argparse exits at parse_args() first.
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import probing.evaluations.baseline_probes.generation.common as gen_common
import probing.evaluations.baseline_probes.generation.generate_are_you_sure as generate_are_you_sure
import probing.evaluations.baseline_probes.generation.generate_sae as generate_sae
import probing.evaluations.baseline_probes.generation.generate_syconbench as generate_syconbench
import probing.evaluations.baseline_probes.generation.generate_sypr as generate_sypr
import probing.evaluations.baseline_probes.generation.generate_truthfulqa as generate_truthfulqa
import probing.evaluations.baseline_probes.generation.dissociating_sycophancy_generate as dissociating_sycophancy_generate
import probing.evaluations.baseline_probes.generation.moral_generate_openrouter as moral_generate_openrouter
import probing.evaluations.baseline_probes.generation.social_generate_openrouter as social_generate_openrouter
import probing.evaluations.baseline_probes.generation.syconbench_fetch_generations as syconbench_fetch_generations
import probing.evaluations.baseline_probes.generation.verify_model_generation_setup as verify_model_generation_setup

# (module, dotted path for `python -m`, extra required args besides --help)
CLI_MODULES = [
    (generate_are_you_sure, "probing.evaluations.baseline_probes.generation.generate_are_you_sure"),
    (generate_sae, "probing.evaluations.baseline_probes.generation.generate_sae"),
    (generate_syconbench, "probing.evaluations.baseline_probes.generation.generate_syconbench"),
    (generate_sypr, "probing.evaluations.baseline_probes.generation.generate_sypr"),
    (generate_truthfulqa, "probing.evaluations.baseline_probes.generation.generate_truthfulqa"),
    (dissociating_sycophancy_generate, "probing.evaluations.baseline_probes.generation.dissociating_sycophancy_generate"),
    (moral_generate_openrouter, "probing.evaluations.baseline_probes.generation.moral_generate_openrouter"),
    (social_generate_openrouter, "probing.evaluations.baseline_probes.generation.social_generate_openrouter"),
    (syconbench_fetch_generations, "probing.evaluations.baseline_probes.generation.syconbench_fetch_generations"),
    (verify_model_generation_setup, "probing.evaluations.baseline_probes.generation.verify_model_generation_setup"),
]

ALL_MODULE_FILES = [
    gen_common.__file__, generate_are_you_sure.__file__, generate_sae.__file__,
    generate_syconbench.__file__, generate_sypr.__file__, generate_truthfulqa.__file__,
    dissociating_sycophancy_generate.__file__, moral_generate_openrouter.__file__,
    social_generate_openrouter.__file__, syconbench_fetch_generations.__file__,
    verify_model_generation_setup.__file__,
]


def run_help(dotted_module: str, extra_args=None):
    return subprocess.run(
        [sys.executable, "-m", dotted_module, "--help", *(extra_args or [])],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )


def run_without_required(dotted_module: str, args):
    return subprocess.run(
        [sys.executable, "-m", dotted_module, *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )


class GenerationModuleImportTests(unittest.TestCase):
    """All 11 generation modules import cleanly and expose the expected surface."""

    def test_common_exposes_shared_helpers(self):
        for name in ("add_generation_args", "generate_via_openrouter_with_finish_reasons",
                     "generate_via_openrouter", "write_metadata", "source_revision"):
            self.assertTrue(hasattr(gen_common, name), f"generation.common missing {name}")

    def test_no_stale_pipeline_scripts_or_model_registry_references(self):
        import inspect
        for module in (gen_common, generate_are_you_sure, generate_sae, generate_syconbench,
                       generate_sypr, generate_truthfulqa, dissociating_sycophancy_generate,
                       moral_generate_openrouter, social_generate_openrouter,
                       syconbench_fetch_generations, verify_model_generation_setup):
            source = inspect.getsource(module)
            self.assertNotIn("pipeline_scripts", source, f"{module.__name__} still references pipeline_scripts")
            self.assertNotIn("sycophancy_model_registry", source, f"{module.__name__} still references sycophancy_model_registry")
            self.assertNotIn("import sycophancy_steering", source, f"{module.__name__} has an executable sycophancy_steering import")
            # A comment documenting *why* an unmoved sibling stays out of scope
            # (e.g. "depends on sycophancy_steering.py's ActivationSteerer") is
            # fine; an executable reference (ActivationSteerer(...) call, or an
            # import binding the name) is not -- this module never steers.
            executable_hits = [
                line for line in source.splitlines()
                if "ActivationSteerer" in line and not line.strip().startswith("#")
            ]
            self.assertEqual(executable_hits, [], f"{module.__name__} has an executable ActivationSteerer reference: {executable_hits}")

    def test_repo_root_resolves_for_every_module_that_defines_one(self):
        for module in (generate_sae, generate_syconbench, generate_sypr,
                        dissociating_sycophancy_generate, moral_generate_openrouter,
                        social_generate_openrouter, verify_model_generation_setup):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.REPO_ROOT, REPO_ROOT)
                self.assertTrue((module.REPO_ROOT / "probing").is_dir())
                self.assertTrue((module.REPO_ROOT / "utils").is_dir())

    def test_are_you_sure_and_truthfulqa_sycophancy_dir_resolves(self):
        # These two keep SYCOPHANCY_DIR/SCRIPTS_DIR on sys.path for bare imports
        # of the still-application-owned, steering-dependent generation helpers.
        for module in (generate_are_you_sure, generate_truthfulqa):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.REPO_ROOT, REPO_ROOT)
                self.assertEqual(
                    module.SYCOPHANCY_DIR, REPO_ROOT / "tool_calling" / "tasks" / "sycophancy"
                )


class GenerationHelpSmokeTests(unittest.TestCase):
    """`--help` parses and exits 0 for every generation module with a CLI, without
    loading any model, tokenizer, or contacting a network service."""

    def test_help_exits_cleanly_for_every_cli_module(self):
        for _, dotted in CLI_MODULES:
            with self.subTest(module=dotted):
                result = run_help(dotted)
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("usage", result.stdout.lower())


class GenerationRequiredOutputArgTests(unittest.TestCase):
    """Application-owned output defaults (into tool_calling/tasks/sycophancy's
    results/ tree) were replaced with required CLI arguments rather than a new
    default under probing/ -- verify that omitting the output flag now fails
    argument parsing (exit code 2) instead of silently writing somewhere."""

    def test_dissociating_sycophancy_generate_output_required(self):
        result = run_without_required(
            "probing.evaluations.baseline_probes.generation.dissociating_sycophancy_generate", []
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output", result.stderr)

    def test_moral_generate_openrouter_out_required(self):
        result = run_without_required(
            "probing.evaluations.baseline_probes.generation.moral_generate_openrouter", []
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--out", result.stderr)

    def test_social_generate_openrouter_out_required(self):
        result = run_without_required(
            "probing.evaluations.baseline_probes.generation.social_generate_openrouter", []
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--out", result.stderr)

    def test_syconbench_fetch_generations_output_required(self):
        result = run_without_required(
            "probing.evaluations.baseline_probes.generation.syconbench_fetch_generations", []
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output", result.stderr)

    def test_generate_sae_output_required_via_shared_common(self):
        # --output is required=True inside add_generation_args (generation/common.py),
        # shared by generate_are_you_sure/generate_sae/generate_syconbench/
        # generate_sypr/generate_truthfulqa -- exercised once here.
        result = run_without_required(
            "probing.evaluations.baseline_probes.generation.generate_sae",
            ["--input", "/tmp/does-not-matter.jsonl", "--dataset-type", "social", "--model", "x"],
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output", result.stderr)


if __name__ == "__main__":
    unittest.main()
