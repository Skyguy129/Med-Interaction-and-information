"""Exact drug-name resolution. Everything starts here.

A name is normalised (case, whitespace, dose strings such as "10 mg") and matched
exactly, ignoring case, against the drugs table. We have no brand-name data, so
"Advil" is ``unknown_drug`` with did-you-mean suggestions. Typos and abbreviations
are never auto-resolved: the person confirms, or an entry is added to the
optional alias file (``data/aliases.json``, empty for now).
"""

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ddi_backend import load_data

RESOLVED = "resolved"
AMBIGUOUS_NAME = "ambiguous_name"
UNKNOWN_DRUG = "unknown_drug"

DEFAULT_ALIASES = load_data.DATA_DIR / "aliases.json"

# "10 mg", "2.5mg", "500 mcg", "40 units", "5%", "10 mg/ml"
_DOSE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|µg|g|kg|ml|l|iu|units?|meq|%)"
    r"(?:\s*/\s*(?:ml|l|kg|dl|day|hr|h))?(?=\W|$)",
    re.IGNORECASE,
)
_WORDS = re.compile(r"\s+")
_ABBREVIATION = re.compile(r"[A-Z][A-Z0-9]{1,4}")


def normalize_name(text):
    """Return (name, dose_text): the drug name without dose strings, and the dose as typed.

    Doses are echoed back to the caller and never interpreted or recommended.
    """
    text = (text or "").strip()
    doses = [m.group(0).strip() for m in _DOSE.finditer(text)]
    name = _DOSE.sub(" ", text)
    name = _WORDS.sub(" ", name).strip(" ,;:-")
    return name, (" ".join(doses) or None)


@dataclass
class Resolution:
    status: str
    query: str
    name: str = None            # canonical drug name when resolved
    drug_id: int = None
    dose: str = None
    via: str = None             # "exact" or "alias"
    candidates: list = field(default_factory=list)  # did-you-mean names; never auto-chosen
    question: str = None        # clarifying question when the name is not resolved

    @property
    def resolved(self):
        return self.status == RESOLVED

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items() if value not in (None, [])}


def load_aliases(path=None):
    """Alias file: {"advil": "Ibuprofen"}. Missing or empty means no aliases."""
    path = Path(path) if path is not None else DEFAULT_ALIASES
    if not path.is_file():
        return {}
    data = load_data.read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected an object mapping alias -> drug name")
    return {str(k).strip().casefold(): str(v).strip() for k, v in data.items()}


class Resolver:
    def __init__(self, drugs, aliases=None, suggester=None):
        """``drugs`` is [(drug_id, name), ...]. ``suggester(text) -> [names]`` is an optional
        extra source of did-you-mean names (embeddings); it can only add candidates."""
        self._by_key = {name.casefold(): (drug_id, name) for drug_id, name in drugs}
        self._names = [name for _, name in sorted(drugs, key=lambda d: d[1].casefold())]
        self._aliases = {k.casefold(): v for k, v in (aliases or {}).items()}
        self._suggester = suggester

    @classmethod
    def from_connection(cls, conn, aliases=None, suggester=None):
        rows = conn.execute("SELECT drug_id, name FROM drugs").fetchall()
        return cls(rows, aliases=aliases, suggester=suggester)

    def resolve(self, text):
        name, dose = normalize_name(text)
        if not name:
            return Resolution(UNKNOWN_DRUG, text or "", dose=dose,
                              question="Which medication do you mean?")
        key = name.casefold()
        if key in self._by_key:
            drug_id, canonical = self._by_key[key]
            return Resolution(RESOLVED, name, canonical, drug_id, dose, via="exact")
        alias_target = self._aliases.get(key)
        if alias_target and alias_target.casefold() in self._by_key:
            drug_id, canonical = self._by_key[alias_target.casefold()]
            return Resolution(RESOLVED, name, canonical, drug_id, dose, via="alias")

        candidates = self.did_you_mean(name)
        contained = self._containing(name)
        if _ABBREVIATION.fullmatch(name):
            question = (f"'{name}' is a short abbreviation, which can mean more than one thing. "
                        "Which medication do you mean?")
            return Resolution(AMBIGUOUS_NAME, name, dose=dose, candidates=candidates, question=question)
        if len(contained) > 1:
            question = f"'{name}' could be more than one drug in our data. Which do you mean?"
            return Resolution(AMBIGUOUS_NAME, name, dose=dose, candidates=contained, question=question)
        if candidates:
            question = (f"'{name}' is not a drug name in our data. Did you mean "
                        f"{', '.join(candidates)}? Please confirm; we do not guess.")
        else:
            question = f"'{name}' is not a drug name in our data."
        return Resolution(UNKNOWN_DRUG, name, dose=dose, candidates=candidates, question=question)

    def _containing(self, text):
        """Names that contain the text as a whole word or word prefix (e.g. 'valpro' -> 2 drugs)."""
        key = text.casefold()
        if len(key) < 3:
            return []
        hits = []
        for name in self._names:
            if any(word.startswith(key) for word in re.split(r"[\s/(),]+", name.casefold())):
                hits.append(name)
        return hits

    def did_you_mean(self, text, limit=5):
        key = text.casefold()
        found = []
        for name in self._containing(text):
            found.append(name)
        for match in difflib.get_close_matches(key, list(self._by_key), n=limit, cutoff=0.6):
            canonical = self._by_key[match][1]
            if canonical not in found:
                found.append(canonical)
        if self._suggester is not None:
            for name in self._suggester(text):
                if name in self._names and name not in found:
                    found.append(name)
        return found[:limit]
