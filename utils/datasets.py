"""
Loading and iterating over the SAE/datasets CSVs.

Each CSV has a different schema (see DATASET_SPECS below), so a per-file spec
maps each dataset to the column(s) that hold the text to send to a model,
plus an optional id column.
"""

from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[1] / "SAE" / "datasets"

EXCLUDED_FILES = set()

# filename -> {"id_col": column to use as row id (None = use the dataframe index),
#              "prompt_cols": list of columns to generate a response for}
DATASET_SPECS = {
    "AITA-NTA-FLIP.csv": {"id_col": "id", "prompt_cols": ["original_post", "flipped_story"]},
    "AITA-NTA-OG.csv": {"id_col": "id", "prompt_cols": ["original_post"]},
    "AITA-YTA.csv": {"id_col": None, "prompt_cols": ["prompt"]},
    "OEQ.csv": {"id_col": None, "prompt_cols": ["prompt"]},
    "SS.csv": {"id_col": None, "prompt_cols": ["sentence"]},
}


def list_dataset_files() -> list[str]:
    """CSV filenames in SAE/datasets that have a known spec."""
    files = []
    for path in sorted(DATASETS_DIR.glob("*.csv")):
        if path.name in EXCLUDED_FILES:
            continue
        if path.name not in DATASET_SPECS:
            print(f"warning: no dataset spec for {path.name}, skipping")
            continue
        files.append(path.name)
    return files


def load_dataset(filename: str) -> pd.DataFrame:
    """Load a dataset CSV, treating its leading unnamed column as the index."""
    return pd.read_csv(DATASETS_DIR / filename, index_col=0)


def iter_prompts(filename: str):
    """
    Yield {"dataset", "row_id", "prompt_col", "text"} for every non-empty
    prompt cell in the given dataset, per its spec.
    """
    spec = DATASET_SPECS[filename]
    df = load_dataset(filename)
    dataset_name = Path(filename).stem
    id_col = spec["id_col"]

    for row_id, row in df.iterrows():
        record_id = row[id_col] if id_col else row_id
        for prompt_col in spec["prompt_cols"]:
            text = row.get(prompt_col)
            if pd.isna(text) or not str(text).strip():
                continue
            yield {
                "dataset": dataset_name,
                "row_id": record_id,
                "prompt_col": prompt_col,
                "text": str(text).strip(),
            }


def iter_all_prompts(filenames: list[str] | None = None):
    """Yield prompt records across multiple datasets (default: all known, non-excluded)."""
    for filename in filenames or list_dataset_files():
        yield from iter_prompts(filename)
