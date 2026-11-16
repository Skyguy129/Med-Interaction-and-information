"""Repository-wide checks for the rules in the brief: what must not exist, and what must be documented."""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "Python", "test_evidence_files", "__pycache__", "experiments", "node_modules"}


def repo_files(*suffixes, skip=SKIP_DIRS):
    """Files under the repository with these suffixes, without walking into big folders like .venv."""
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            path = Path(folder) / name
            if path.suffix in suffixes:
                yield path


def build(*parts):
    """Assemble a pattern from pieces so this file does not contain the words it forbids elsewhere."""
    return "".join(parts)


class TestRejectedTechniquesAreAbsent(unittest.TestCase):
    def test_no_splitting_keyword_weighting_or_cutoff_scoring_outside_experiments(self):
        pattern = re.compile(build("chun", "k|tf-?", "idf|thres", "hold"), re.IGNORECASE)
        offenders = []
        for path in repo_files(".py", ".md", ".txt", ".cfg", ".toml", ".yml"):
            if path.name == "test_hygiene.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(offenders, [])

    def test_no_urgent_or_not_classifier(self):
        pattern = re.compile(r"urgent[_ -]?(or[_ -]?not|classif)", re.IGNORECASE)
        offenders = [str(p.relative_to(ROOT)) for p in repo_files(".py") if p.name != "test_hygiene.py"
                     and pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
        self.assertEqual(offenders, [])


class TestNoUserInterfaceCode(unittest.TestCase):
    def test_no_html_css_javascript_or_react_files(self):
        found = [str(p.relative_to(ROOT)) for p in repo_files(".html", ".htm", ".css", ".scss", ".js", ".jsx", ".ts", ".tsx", ".vue")]
        self.assertEqual(found, [])

    def test_python_code_does_not_emit_markup(self):
        pattern = re.compile(r"<(?:html|div|script|body)\b", re.IGNORECASE)
        offenders = [str(p.relative_to(ROOT)) for p in repo_files(".py")
                     if p.name != "test_hygiene.py" and pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
        self.assertEqual(offenders, [])


class TestConventions(unittest.TestCase):
    def test_experiment_source_labels_are_not_used_in_code(self):
        pattern = re.compile(r"""["'`]S[1-5]["'`]|\bS[1-5]\s*=""")
        offenders = []
        for path in list((ROOT / "ddi_backend").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
            if path.name == "test_hygiene.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(offenders, [])

    def test_no_api_key_in_the_repository(self):
        pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
        offenders = [str(p.relative_to(ROOT)) for p in repo_files(".py", ".md", ".json", ".txt")
                     if p.name not in {"hp.json"} and pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
        self.assertEqual(offenders, [])

    def test_generated_files_are_not_tracked(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for needle in ("medications.db", "agent_state.db", ".venv"):
            self.assertIn(needle, ignore)

    def test_notebooks_live_only_in_experiments(self):
        outside = [str(p.relative_to(ROOT)) for p in repo_files(".ipynb")]
        self.assertEqual(outside, [])
        self.assertGreaterEqual(len(list((ROOT / "experiments").glob("*.ipynb"))), 8)

    def test_required_layout_exists(self):
        for name in ("ingest", "resolver", "eligibility", "rules", "retrieval", "llm", "agent", "models", "api", "cli"):
            self.assertTrue((ROOT / "ddi_backend" / f"{name}.py").is_file(), name)
        for folder in ("data", "experiments", "eval", "tests"):
            self.assertTrue((ROOT / folder).is_dir(), folder)

    def test_core_loads_without_the_optional_tiers(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import ddi_backend.checker, ddi_backend.agent, ddi_backend.retrieval, ddi_backend.cli;"
            "bad = [m for m in ('numpy', 'sklearn', 'sentence_transformers', 'torch', 'fastapi', 'openai') if m in sys.modules];"
            "sys.exit(1 if bad else 0)"
        )
        result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestReadmeDocumentsTheDecisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_each_tool_decision_is_explained(self):
        for heading in ("**Adopted**", "**Modified**", "**Deferred**", "**Rejected**"):
            self.assertIn(heading, self.readme)
        for topic in ("knowledge graph", "eligibility filtering", "Embedding neighbours", "neural network", "RAG"):
            self.assertIn(topic, self.readme)

    def test_it_says_what_the_data_cannot_do_and_that_a1_to_a6_are_undefined(self):
        self.assertIn("not available in our data", self.readme)
        self.assertIn("A1 through A6", self.readme)
        self.assertIn("never defines them", self.readme)

    def test_it_carries_the_disclaimer_and_the_two_rules(self):
        self.assertIn("Not medical advice", self.readme)
        self.assertIn("never presented as", self.readme)
        self.assertIn("Major", self.readme)


if __name__ == "__main__":
    unittest.main()
