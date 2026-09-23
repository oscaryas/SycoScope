from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.datasets import DATASET_SPECS, DATASETS_DIR, list_dataset_files


class DatasetsDirExtractionTests(unittest.TestCase):
    """utils/datasets.py::DATASETS_DIR must point at the shared
    probing/data/source/elephant/ location (the five ELEPHANT CSVs extracted
    from SAE/datasets/), not at SAE/datasets/ itself -- SAE/ will not exist
    on main once cleanup finishes."""

    def test_datasets_dir_points_at_shared_source_location(self):
        self.assertEqual(DATASETS_DIR.parts[-4:], ("probing", "data", "source", "elephant"))

    def test_datasets_dir_exists_and_has_the_five_csvs(self):
        self.assertTrue(DATASETS_DIR.is_dir())
        found = list_dataset_files()
        self.assertEqual(set(found), set(DATASET_SPECS))

    def test_list_dataset_files_matches_known_specs(self):
        for filename in list_dataset_files():
            self.assertIn(filename, DATASET_SPECS)


if __name__ == "__main__":
    unittest.main()
