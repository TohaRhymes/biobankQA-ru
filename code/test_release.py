"""Acceptance checks for the reduced code-and-results release."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from verify_release import ROOT, verify


class ReleaseAcceptance(unittest.TestCase):
    def test_code_results_and_figures_are_available(self):
        for rel in ("benchmark/scripts/run_direct.py", "benchmark/scripts/run_rag.py",
                    "benchmark/rag/build_index.py", "prompts/iterations/iter3.md",
                    "code/reproduce.py", "expected/mcq.csv", "results/reproduced/mcq.csv",
                    "figures/article/Fig_1.png", "results/reproduced/figure5.png"):
            with self.subTest(path=rel):
                self.assertTrue((ROOT / rel).is_file())

    def test_release_integrity(self):
        self.assertGreater(verify(), 50)

    def test_modified_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "candidate"
            shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            target = copy / "expected/mcq.csv"
            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaises(AssertionError):
                verify(copy)

    def test_returned_private_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "candidate"
            shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            (copy / "data").mkdir()
            (copy / "data/tasks.jsonl").write_text("{}\n")
            with self.assertRaises(AssertionError):
                verify(copy)

    def test_reproduction_explains_withheld_inputs(self):
        result = subprocess.run([sys.executable, str(ROOT / "code/reproduce.py")],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("withheld evaluation inputs", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
