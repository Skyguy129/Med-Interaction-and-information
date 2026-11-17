# Module docstring: repository-wide checks for what the brief says must not exist and what must be documented.
"""Repository-wide checks for the rules in the brief: what must not exist, and what must be documented."""

import os  # os.walk to walk the repository folders
import re  # regular expressions for the forbidden-pattern checks
import subprocess  # runs a clean Python process for the optional-tier import test
import sys  # sys.executable and sys.modules, used in the import test
import unittest  # standard-library test framework
from pathlib import Path  # path objects

ROOT = Path(__file__).resolve().parent.parent  # the repository root (this file lives in tests/)
SKIP_DIRS = {".git", ".venv", "Python", "test_evidence_files", "__pycache__", "experiments", "node_modules"}  # folders never scanned


def repo_files(*suffixes, skip=SKIP_DIRS):  # generator: every file in the repo with one of the given suffixes
    # Docstring below says it avoids walking into big folders such as the virtual environment.
    """Files under the repository with these suffixes, without walking into big folders like .venv."""
    for folder, dirs, files in os.walk(ROOT):  # visit every folder under the root
        dirs[:] = [d for d in dirs if d not in skip]  # prune skipped folders in place so os.walk does not enter them
        for name in files:  # each file in the current folder
            path = Path(folder) / name  # full path
            if path.suffix in suffixes:  # keep only the wanted file types
                yield path


def build(*parts):  # joins string pieces into one pattern
    # Docstring below: lets this file avoid containing the words it forbids elsewhere.
    """Assemble a pattern from pieces so this file does not contain the words it forbids elsewhere."""
    return "".join(parts)  # concatenate the pieces


class TestRejectedTechniquesAreAbsent(unittest.TestCase):  # checks that the rejected techniques are not in the code
    def test_no_splitting_keyword_weighting_or_cutoff_scoring_outside_experiments(self):  # three rejected ideas
        # Pattern for the rejected splitting, keyword-weighting and cut-off-scoring words, built from pieces, case-insensitive.
        pattern = re.compile(build("chun", "k|tf-?", "idf|thres", "hold"), re.IGNORECASE)
        offenders = []  # collects file:line locations that match
        for path in repo_files(".py", ".md", ".txt", ".cfg", ".toml", ".yml"):  # every text-like repo file
            if path.name == "test_hygiene.py":  # this file is exempt
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):  # each line with its number
                if pattern.search(line):  # the line contains a forbidden word
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")  # record where
        self.assertEqual(offenders, [])  # there must be none

    def test_no_urgent_or_not_classifier(self):  # the Meeting 6 triage-style classifier idea was rejected
        pattern = re.compile(r"urgent[_ -]?(or[_ -]?not|classif)", re.IGNORECASE)  # forbidden phrase variants
        # Python files (other than this one) whose text matches the pattern.
        offenders = [str(p.relative_to(ROOT)) for p in repo_files(".py") if p.name != "test_hygiene.py"
                     and pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
        self.assertEqual(offenders, [])  # none may match


class TestNoUserInterfaceCode(unittest.TestCase):  # the brief says no user-interface code in this repository
    def test_no_html_css_javascript_or_react_files(self):  # no front-end file types
        found = [str(p.relative_to(ROOT)) for p in repo_files(".html", ".htm", ".css", ".scss", ".js", ".jsx", ".ts", ".tsx", ".vue")]  # any front-end files
        self.assertEqual(found, [])  # there must be none

    def test_python_code_does_not_emit_markup(self):  # Python must not build web page markup either
        pattern = re.compile(r"<(?:html|div|script|body)\b", re.IGNORECASE)  # opening tags of common page elements
        # Python files (other than this one) that contain such a tag.
        offenders = [str(p.relative_to(ROOT)) for p in repo_files(".py")
                     if p.name != "test_hygiene.py" and pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
        self.assertEqual(offenders, [])  # none may match


class TestConventions(unittest.TestCase):  # tests repository conventions
    def test_experiment_source_labels_are_not_used_in_code(self):  # code uses stable source keys, not the experiment labels
        pattern = re.compile(r"""["'`]S[1-5]["'`]|\bS[1-5]\s*=""")  # a quoted label or an assignment to one
        offenders = []  # collects file:line locations that match
        for path in list((ROOT / "ddi_backend").glob("*.py")) + list((ROOT / "tests").glob("*.py")):  # the package and the tests
            if path.name == "test_hygiene.py":  # this file is exempt
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):  # each line with its number
                if pattern.search(line):  # the line uses one of the labels
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")  # record where
        self.assertEqual(offenders, [])  # there must be none

    def test_no_api_key_in_the_repository(self):  # a real key must never be committed
        pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")  # the shape of an OpenAI-style key
        # Text files (except the big hp.json) that contain something shaped like a key.
        offenders = [str(p.relative_to(ROOT)) for p in repo_files(".py", ".md", ".json", ".txt")
                     if p.name not in {"hp.json"} and pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
        self.assertEqual(offenders, [])  # none may match

    def test_generated_files_are_not_tracked(self):  # generated files must be git-ignored
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")  # the ignore file's text
        for needle in ("medications.db", "agent_state.db", ".venv"):  # generated database, state file and virtual environment
            self.assertIn(needle, ignore)  # each one is listed

    def test_notebooks_live_only_in_experiments(self):  # class notebooks stay archived in experiments/
        outside = [str(p.relative_to(ROOT)) for p in repo_files(".ipynb")]  # notebooks found outside the skipped folders (experiments is skipped)
        self.assertEqual(outside, [])  # there must be none
        self.assertGreaterEqual(len(list((ROOT / "experiments").glob("*.ipynb"))), 8)  # and at least eight are in experiments

    def test_required_layout_exists(self):  # the brief's required files and folders are present
        for name in ("ingest", "resolver", "eligibility", "rules", "retrieval", "llm", "agent", "models", "api", "cli"):  # required modules
            self.assertTrue((ROOT / "ddi_backend" / f"{name}.py").is_file(), name)  # each is a file
        for folder in ("data", "experiments", "eval", "tests"):  # required folders
            self.assertTrue((ROOT / folder).is_dir(), folder)  # each is a directory

    def test_core_loads_without_the_optional_tiers(self):  # importing the core must not pull in tier two or three libraries
        code = (  # a small program run in a fresh Python process; its pieces are joined into one string
            "import sys; sys.path.insert(0, sys.argv[1]);"  # make the repository importable
            "import ddi_backend.checker, ddi_backend.agent, ddi_backend.retrieval, ddi_backend.cli;"  # import the core modules
            "bad = [m for m in ('numpy', 'sklearn', 'sentence_transformers', 'torch', 'fastapi', 'openai') if m in sys.modules];"  # optional libraries that got loaded
            "sys.exit(1 if bad else 0)"  # fail if any optional library was imported
        )
        result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True, text=True)  # run it in isolated mode
        self.assertEqual(result.returncode, 0, result.stderr)  # success means the core is self-contained


class TestReadmeDocumentsTheDecisions(unittest.TestCase):  # tests that the README records the required decisions
    @classmethod
    def setUpClass(cls):  # runs once for the class
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")  # load the README text

    def test_each_tool_decision_is_explained(self):  # adopted, modified, deferred and rejected tools are covered
        for heading in ("**Adopted**", "**Modified**", "**Deferred**", "**Rejected**"):  # the four decision headings
            self.assertIn(heading, self.readme)  # each is present
        for topic in ("knowledge graph", "eligibility filtering", "Embedding neighbours", "neural network", "RAG"):  # the tools discussed
            self.assertIn(topic, self.readme)  # each is mentioned

    def test_it_says_what_the_data_cannot_do_and_that_a1_to_a6_are_undefined(self):  # honest limits are stated
        self.assertIn("not available in our data", self.readme)  # drug-level fields are not in the data
        self.assertIn("A1 through A6", self.readme)  # the undefined failure codes are mentioned
        self.assertIn("never defines them", self.readme)  # and said to be undefined

    def test_it_carries_the_disclaimer_and_the_two_rules(self):  # the disclaimer and the two main rules are present
        self.assertIn("Not medical advice", self.readme)  # the disclaimer
        self.assertIn("never presented as", self.readme)  # rule one: not found is never presented as safe
        self.assertIn("Major", self.readme)  # rule two: Major interactions are the priority


if __name__ == "__main__":  # only when the file is run directly
    unittest.main()  # run the tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Repository-wide guard tests rather than tests of one module. They scan the repo to confirm the
# rejected techniques (the splitting, keyword-weighting and cut-off-scoring ideas and the Meeting 6 triage-style
# classifier) are absent outside experiments/, that no front-end code or markup exists, that experiment labels and API
# keys are not in the code, that generated files are git-ignored, that notebooks live only in experiments/, that the
# required layout exists, that the core imports without the optional tier libraries, and that the README documents the
# Adopted / Modified / Deferred / Rejected decisions and the key limits.
# ASSIGNMENT TOOL: Not one tool itself; it enforces the assignment's decisions across all of them (the rejected RAG
# and cut-off-scoring lab, the deferred neural network, the adopted tools' tier separation).
# ---------------------------------------------------------------------------
