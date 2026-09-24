#!/usr/bin/env python3
import sys
from probing.evaluations.judge.judge_dataset import main

if __name__ == "__main__":
    sys.argv.extend(["--dataset-type", "moral"])
    main()
