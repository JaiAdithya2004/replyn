"""
run_all.py
----------
Runs the whole pipeline end-to-end:
    1. build the dataset from real data   (generate_dataset.py)
    2. generate AI replies                (generate_replies.py)
    3. evaluate them                      (evaluate.py)

Run:  python run_all.py
"""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["generate_dataset.py", "generate_replies.py", "evaluate.py"]


def main():
    for script in STEPS:
        print("\n" + "#" * 64)
        print(f"# RUNNING {script}")
        print("#" * 64)
        result = subprocess.run([sys.executable, os.path.join(HERE, script)])
        if result.returncode != 0:
            sys.exit(f"Step failed: {script} (exit {result.returncode})")
    print("\nAll steps complete. See evaluation.json for full results.")


if __name__ == "__main__":
    main()
