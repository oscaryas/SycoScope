"""
Shared paths, cell/slug bookkeeping and run-metadata helpers for the
prompt_probes pipeline.

The pipeline trains linear probes to predict *which of a contrastive pair of
system prompts was in context*, following Natarajan et al. (2026), "One Probe
Won't Catch Them All" (arXiv 2602.01425). The 14 prompt pairs in
data/sycophancy_probe_prompt_pairs.json target the 8 cells of the Ye et al.
(2025) sycophancy taxonomy plus 1 general baseline and 5 non-sycophancy
controls.

Every stage is sharded by cell slug: one prompt pair can be generated,
extracted and probed end-to-end without touching the other 13. Cross-cell
artifacts (summary.json, the transfer matrices) are rebuilt by scanning for
whatever per-cell outputs happen to exist.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT_PROBES_DIR = HERE.parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = PROMPT_PROBES_DIR / "data"
RESULTS_DIR = PROMPT_PROBES_DIR / "results"
SYCOPHANCY_DIR = REPO_ROOT / "tool_calling" / "tasks" / "sycophancy"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAIRS_PATH = DATA_DIR / "sycophancy_probe_prompt_pairs.json"
USER_PROMPTS_PATH = DATA_DIR / "perez_user_prompts.jsonl"

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Slugs are hardcoded rather than derived from the "cell" strings so that
# rewording a prompt in the JSON can't silently orphan an existing results
# directory. A hard error here is the intended failure mode.
CELL_SLUGS = {
    "General (baseline)": "general_baseline",
    "Position-Verifiable / Explicit": "pv_explicit",
    "Position-Verifiable / Implicit": "pv_implicit",
    "Position-Subjective / Explicit": "ps_explicit",
    "Position-Subjective / Implicit": "ps_implicit",
    "Person-Traits / Explicit": "pt_explicit",
    "Person-Traits / Implicit": "pt_implicit",
    "Person-Emotions / Explicit": "pe_explicit",
    "Person-Emotions / Implicit": "pe_implicit",
    "Warranted praise": "ctrl_warranted_praise",
    "Genuine agreement": "ctrl_genuine_agreement",
    "Appropriate emotional support": "ctrl_emotional_support",
    "Calibrated hedging": "ctrl_calibrated_hedging",
    "Ordinary politeness": "ctrl_politeness",
}

# The 15th "cell" is not a pair at all: responses generated with no system
# prompt. It is the Alpaca-control analogue from the paper's section 5.5 --
# probe logit scales aren't comparable across probes, so scores get reported
# relative to each probe's median on this neutral set.
NEUTRAL_SLUG = "neutral"

POSITIONS = ("last_prompt", "first5", "response")
DEFAULT_LAYER_FRACS = (0.25, 0.50, 0.75)

# Label convention, stated once and referenced everywhere: label 1 is the
# "sycophantic" slot of the pair, label 0 the "non_sycophantic" slot.
#
# CAUTION: for pair_type == "control" the label-1 slot holds the *legitimate*
# behaviour (praise only what merits praise, agree only where the user is
# right, acknowledge genuine distress, hedge where a matter is genuinely
# unsettled, ordinary courtesy). A control probe's direction therefore points
# toward warranted behaviour, NOT toward sycophancy. Reading that sign the
# wrong way is the easiest route to a wrong conclusion, which is why the
# record field is called "polarity" and pair_type is carried everywhere.
POLARITIES = ("sycophantic", "non_sycophantic")
POLARITY_LABEL = {"sycophantic": 1, "non_sycophantic": 0}
POLARITY_TAG = {"sycophantic": "pos", "non_sycophantic": "neg"}


def load_prompt_pairs(path: Path = PAIRS_PATH) -> list[dict]:
    """The 14 prompt pairs, each annotated with its stable slug and index."""
    pairs = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"cell", "type", "sycophantic", "non_sycophantic"}
    out = []
    for i, pair in enumerate(pairs):
        missing = required - set(pair)
        if missing:
            raise ValueError(f"pair {i} ({pair.get('cell')!r}) missing fields: {sorted(missing)}")
        if pair["cell"] not in CELL_SLUGS:
            raise ValueError(
                f"pair {i} has cell {pair['cell']!r}, absent from CELL_SLUGS. Add it there "
                "with a stable slug rather than deriving slugs from the string."
            )
        out.append({**pair, "slug": CELL_SLUGS[pair["cell"]], "pair_index": i})
    slugs = [p["slug"] for p in out]
    if len(set(slugs)) != len(slugs):
        raise ValueError("duplicate slugs across pairs")
    return out


def select_cells(pairs: list[dict], slugs: list[str] | None) -> list[dict]:
    """Filter pairs to the requested slugs, preserving file order.

    NEUTRAL_SLUG is accepted by the CLI but is not a pair, so callers needing
    it handle it separately; it is ignored here.
    """
    if not slugs:
        return pairs
    wanted = {s for s in slugs if s != NEUTRAL_SLUG}
    known = {p["slug"] for p in pairs}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown cell slug(s): {unknown}\nknown: {sorted(known)} (+ {NEUTRAL_SLUG!r})")
    return [p for p in pairs if p["slug"] in wanted]


def all_slugs(include_neutral: bool = True) -> list[str]:
    slugs = list(CELL_SLUGS.values())
    return slugs + [NEUTRAL_SLUG] if include_neutral else slugs


def add_cells_arg(parser) -> None:
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        metavar="SLUG",
        help=(
            "Cell slugs to process (default: all). Each cell is self-contained, so one "
            "slug can be run without touching the others. Known: " + ", ".join(all_slugs())
        ),
    )


def resolve_run_dir(run_name: str, create: bool = True) -> Path:
    run_dir = RESULTS_DIR / run_name
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_code_version() -> str:
    """Short git sha, suffixed -dirty when the tree has uncommitted changes."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL) != 0
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def write_run_info(run_dir: Path, stage: str, args, extra: dict | None = None) -> None:
    """Merge this stage's argparse dump + metadata into run_dir/run_info.json.

    Merged rather than overwritten, so running stages independently (or
    re-running a single cell) preserves the other stages' provenance.
    """
    path = run_dir / "run_info.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    info[stage] = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "code_version": get_code_version(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
