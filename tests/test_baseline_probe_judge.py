"""Smoke tests for the baseline-probe judge modules moved from
building-agent's tool_calling/tasks/sycophancy/{pipeline_scripts/judges/*,
are_you_sure_correctness_judge,moral_sycophancy_judge,social_sycophancy_judge,
sycophantic_praise_judge,truthfulqa_verdict_judge}.py and scripts/
{incremental_judge_moral_flip,incremental_judge_social,
run_moral_sycophancy_judge_aita,run_social_sycophancy_judge_oeq}.py into
probing/evaluations/baseline_probes/judge/.

These tests only import modules and parse --help output (or verify required
CLI arguments are enforced) -- no anthropic client construction is ever
reached (it's inside main()/library functions, never at import time or
during --help's argparse.parse_args()), no network calls, no
ANTHROPIC_API_KEY required.
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import probing.evaluations.baseline_probes.judge.judge_correctness as judge_correctness
import probing.evaluations.baseline_probes.judge.judge_dataset as judge_dataset
import probing.evaluations.baseline_probes.judge.judge_moral as judge_moral
import probing.evaluations.baseline_probes.judge.judge_praise as judge_praise
import probing.evaluations.baseline_probes.judge.judge_social as judge_social
import probing.evaluations.baseline_probes.judge.judge_syconbench as judge_syconbench
import probing.evaluations.baseline_probes.judge.scoring as scoring
import probing.evaluations.baseline_probes.judge.are_you_sure_correctness_judge as are_you_sure_correctness_judge
import probing.evaluations.baseline_probes.judge.moral_sycophancy_judge as moral_sycophancy_judge
import probing.evaluations.baseline_probes.judge.social_sycophancy_judge as social_sycophancy_judge
import probing.evaluations.baseline_probes.judge.sycophantic_praise_judge as sycophantic_praise_judge
import probing.evaluations.baseline_probes.judge.truthfulqa_verdict_judge as truthfulqa_verdict_judge
import probing.evaluations.baseline_probes.judge.incremental_judge_moral_flip as incremental_judge_moral_flip
import probing.evaluations.baseline_probes.judge.incremental_judge_social as incremental_judge_social
import probing.evaluations.baseline_probes.judge.run_moral_sycophancy_judge_aita as run_moral_sycophancy_judge_aita
import probing.evaluations.baseline_probes.judge.run_social_sycophancy_judge_oeq as run_social_sycophancy_judge_oeq

LIBRARY_MODULES = [
    scoring, are_you_sure_correctness_judge, moral_sycophancy_judge,
    social_sycophancy_judge, sycophantic_praise_judge, truthfulqa_verdict_judge,
]

CLI_MODULES = [
    "probing.evaluations.baseline_probes.judge.judge_correctness",
    "probing.evaluations.baseline_probes.judge.judge_dataset",
    "probing.evaluations.baseline_probes.judge.judge_moral",
    "probing.evaluations.baseline_probes.judge.judge_praise",
    "probing.evaluations.baseline_probes.judge.judge_social",
    "probing.evaluations.baseline_probes.judge.judge_syconbench",
    "probing.evaluations.baseline_probes.judge.incremental_judge_moral_flip",
    "probing.evaluations.baseline_probes.judge.incremental_judge_social",
    "probing.evaluations.baseline_probes.judge.run_moral_sycophancy_judge_aita",
    "probing.evaluations.baseline_probes.judge.run_social_sycophancy_judge_oeq",
]

REPO_ROOT_MODULES = [
    judge_correctness, judge_dataset, judge_syconbench,
    are_you_sure_correctness_judge, moral_sycophancy_judge, social_sycophancy_judge,
    sycophantic_praise_judge, truthfulqa_verdict_judge,
    incremental_judge_moral_flip, incremental_judge_social,
    run_moral_sycophancy_judge_aita, run_social_sycophancy_judge_oeq,
]

ALL_SOURCE_MODULES = LIBRARY_MODULES + [
    judge_correctness, judge_dataset, judge_moral, judge_praise, judge_social,
    judge_syconbench, incremental_judge_moral_flip, incremental_judge_social,
    run_moral_sycophancy_judge_aita, run_social_sycophancy_judge_oeq,
]


def run_help(dotted_module: str):
    return subprocess.run(
        [sys.executable, "-m", dotted_module, "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )


def run_without_required(dotted_module: str, args):
    return subprocess.run(
        [sys.executable, "-m", dotted_module, *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )


class JudgeModuleImportTests(unittest.TestCase):
    """All 16 judge modules import cleanly."""

    def test_library_modules_expose_expected_functions(self):
        self.assertTrue(hasattr(scoring, "score_rows"))
        self.assertTrue(hasattr(are_you_sure_correctness_judge, "judge_correctness_batch"))
        self.assertTrue(hasattr(moral_sycophancy_judge, "judge_verdict"))
        self.assertTrue(hasattr(moral_sycophancy_judge, "iter_flip_pairs"))
        self.assertTrue(hasattr(social_sycophancy_judge, "judge_metric"))
        self.assertTrue(hasattr(social_sycophancy_judge, "iter_dataset_records"))
        self.assertTrue(hasattr(sycophantic_praise_judge, "judge_praise_batch"))
        self.assertTrue(hasattr(truthfulqa_verdict_judge, "judge_truthful_batch"))

    def test_shims_delegate_to_judge_dataset_main(self):
        self.assertIs(judge_moral.main, judge_dataset.main)
        self.assertIs(judge_praise.main, judge_dataset.main)
        self.assertIs(judge_social.main, judge_dataset.main)

    def test_no_stale_pipeline_scripts_or_bare_sibling_imports(self):
        import inspect
        for module in ALL_SOURCE_MODULES:
            source = inspect.getsource(module)
            self.assertNotIn("pipeline_scripts", source, f"{module.__name__} still references pipeline_scripts")
            for stale in ("from social_sycophancy_judge import", "from moral_sycophancy_judge import",
                          "from sycophantic_praise_judge import", "from are_you_sure_correctness_judge import",
                          "from judge_dataset import"):
                self.assertNotIn(stale, source, f"{module.__name__} has a stale bare sibling import: {stale}")

    def test_repo_root_resolves_for_every_module_that_defines_one(self):
        for module in REPO_ROOT_MODULES:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.REPO_ROOT, REPO_ROOT)
                self.assertTrue((module.REPO_ROOT / "probing").is_dir())
                self.assertTrue((module.REPO_ROOT / "utils").is_dir())

    def test_sae_results_defaults_still_resolve_under_repo_root(self):
        # SAE/results/ is a shared, unmoved top-level tree (this extraction
        # never moves anything out of SAE/) -- these two defaults should keep
        # pointing at it correctly from the new probing/evaluations/... location.
        self.assertEqual(
            moral_sycophancy_judge.DEFAULT_INPUT_PATH,
            REPO_ROOT / "SAE" / "results" / "AITA-NTA-FLIP.jsonl",
        )
        self.assertEqual(
            social_sycophancy_judge.DEFAULT_RESULTS_DIR,
            REPO_ROOT / "SAE" / "results",
        )


class JudgeHelpSmokeTests(unittest.TestCase):
    """`--help` parses and exits 0 for every judge module with a CLI, without
    constructing an anthropic client or contacting any network service."""

    def test_help_exits_cleanly_for_every_cli_module(self):
        for dotted in CLI_MODULES:
            with self.subTest(module=dotted):
                result = run_help(dotted)
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("usage", result.stdout.lower())


class JudgeRequiredOutputArgTests(unittest.TestCase):
    """run_social_sycophancy_judge_oeq.py's --output-path used to default into
    tool_calling/tasks/sycophancy's own application-owned results/generations/
    tree; per this task's brief, that default is now a required argument
    instead of a new default under probing/."""

    def test_run_social_sycophancy_judge_oeq_output_path_required(self):
        result = run_without_required(
            "probing.evaluations.baseline_probes.judge.run_social_sycophancy_judge_oeq", []
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output-path", result.stderr)

    def test_run_social_sycophancy_judge_oeq_input_path_default_is_sae_results(self):
        # --input-path reads from the shared, unmoved SAE/results/ tree (not
        # application-owned), so it correctly keeps a default rather than
        # becoming required.
        result = run_help(
            "probing.evaluations.baseline_probes.judge.run_social_sycophancy_judge_oeq"
        )
        self.assertIn("OEQ.jsonl", result.stdout)


if __name__ == "__main__":
    unittest.main()
