# Module docstring: explains that drug names are matched exactly (ignoring case) and never guessed
"""Exact drug-name resolution. Everything starts here.

A name is normalised (case, whitespace, dose strings such as "10 mg") and matched
exactly, ignoring case, against the drugs table. We have no brand-name data, so
"Advil" is ``unknown_drug`` with did-you-mean suggestions. Typos and abbreviations
are never auto-resolved: the person confirms, or an entry is added to the
optional alias file (``data/aliases.json``, empty for now).
"""

import difflib  # gives close-match suggestions for the did-you-mean list
import json  # (imported for JSON handling; reading goes through load_data.read_json)
import re  # regular expressions for dose stripping and word splitting
from dataclasses import dataclass, field  # build the Resolution result type
from pathlib import Path  # path handling for the alias file

from ddi_backend import load_data  # supplies DATA_DIR and the JSON reader

RESOLVED = "resolved"  # status: the name matched a drug exactly
AMBIGUOUS_NAME = "ambiguous_name"  # status: the name could mean more than one drug
UNKNOWN_DRUG = "unknown_drug"  # status: the name is not in our data

DEFAULT_ALIASES = load_data.DATA_DIR / "aliases.json"  # optional alias file (empty for now)

# "10 mg", "2.5mg", "500 mcg", "40 units", "5%", "10 mg/ml"
_DOSE = re.compile(  # pattern that finds dose text inside a typed name
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|µg|g|kg|ml|l|iu|units?|meq|%)"  # a number followed by a unit
    r"(?:\s*/\s*(?:ml|l|kg|dl|day|hr|h))?(?=\W|$)",  # optional "per" part such as /ml, then a word boundary
    re.IGNORECASE,  # units may be typed in any case
)
_WORDS = re.compile(r"\s+")  # runs of whitespace, to collapse to one space
_ABBREVIATION = re.compile(r"[A-Z][A-Z0-9]{1,4}")  # short all-caps tokens such as "ASA" or "HCTZ"


def normalize_name(text):  # split a typed medication into a clean name and its dose text
    """Return (name, dose_text): the drug name without dose strings, and the dose as typed.

    Doses are echoed back to the caller and never interpreted or recommended.
    """
    text = (text or "").strip()  # treat None as empty and trim the ends
    doses = [m.group(0).strip() for m in _DOSE.finditer(text)]  # collect every dose string exactly as typed
    name = _DOSE.sub(" ", text)  # remove the doses from the name
    name = _WORDS.sub(" ", name).strip(" ,;:-")  # collapse spaces and trim leftover punctuation
    return name, (" ".join(doses) or None)  # name plus the joined doses, or None when there were none


@dataclass  # plain mutable result object
class Resolution:
    status: str  # resolved / ambiguous_name / unknown_drug
    query: str  # what the user typed
    name: str = None            # canonical drug name when resolved
    drug_id: int = None  # database id when resolved
    dose: str = None  # dose text echoed back, never evaluated
    via: str = None             # "exact" or "alias"
    candidates: list = field(default_factory=list)  # did-you-mean names; never auto-chosen
    question: str = None        # clarifying question when the name is not resolved

    @property  # read like an attribute
    def resolved(self):  # convenience flag
        return self.status == RESOLVED  # True only for exact (or alias) matches

    def to_dict(self):  # JSON-friendly form
        return {key: value for key, value in self.__dict__.items() if value not in (None, [])}  # drop empty fields


def load_aliases(path=None):  # read the optional alias file
    """Alias file: {"advil": "Ibuprofen"}. Missing or empty means no aliases."""
    path = Path(path) if path is not None else DEFAULT_ALIASES  # default location unless overridden
    if not path.is_file():  # no file means no aliases
        return {}  # empty mapping
    data = load_data.read_json(path)  # parse the JSON
    if not isinstance(data, dict):  # the file must be an object
        raise ValueError(f"{path.name}: expected an object mapping alias -> drug name")
    return {str(k).strip().casefold(): str(v).strip() for k, v in data.items()}  # normalise keys to lower case


class Resolver:  # turns typed names into resolved drugs
    def __init__(self, drugs, aliases=None, suggester=None):  # build the lookup tables
        """``drugs`` is [(drug_id, name), ...]. ``suggester(text) -> [names]`` is an optional
        extra source of did-you-mean names (embeddings); it can only add candidates."""
        self._by_key = {name.casefold(): (drug_id, name) for drug_id, name in drugs}  # lower-case name -> (id, canonical)
        self._names = [name for _, name in sorted(drugs, key=lambda d: d[1].casefold())]  # all names, alphabetical
        self._aliases = {k.casefold(): v for k, v in (aliases or {}).items()}  # lower-cased alias table
        self._suggester = suggester  # optional extra suggestion source

    @classmethod  # alternate constructor
    def from_connection(cls, conn, aliases=None, suggester=None):  # build a Resolver straight from the database
        rows = conn.execute("SELECT drug_id, name FROM drugs").fetchall()  # read every drug
        return cls(rows, aliases=aliases, suggester=suggester)  # construct with them

    def lookup_terms(self):  # vocabulary used to spot drug names inside a sentence
        """{normalised text: drug name or alias} for finding drug mentions inside a sentence."""
        terms = {re.sub(r"[^a-z0-9]+", " ", key).strip(): name for key, (_, name) in self._by_key.items()}  # punctuation-free names
        for alias in self._aliases:  # add the aliases too
            terms[re.sub(r"[^a-z0-9]+", " ", alias).strip()] = alias  # alias maps to itself; resolve() maps it onward
        return terms  # the finished term table

    def resolve(self, text):  # the main entry: resolve one typed name
        name, dose = normalize_name(text)  # clean the input and pull out the dose
        if not name:  # nothing left after cleaning
            return Resolution(UNKNOWN_DRUG, text or "", dose=dose,
                              question="Which medication do you mean?")  # ask instead of guessing
        key = name.casefold()  # case-insensitive key
        if key in self._by_key:  # exact match on a drug name
            drug_id, canonical = self._by_key[key]  # fetch id and official spelling
            return Resolution(RESOLVED, name, canonical, drug_id, dose, via="exact")  # resolved exactly
        alias_target = self._aliases.get(key)  # is it a listed alias?
        if alias_target and alias_target.casefold() in self._by_key:  # alias points to a real drug
            drug_id, canonical = self._by_key[alias_target.casefold()]  # fetch the target drug
            return Resolution(RESOLVED, name, canonical, drug_id, dose, via="alias")  # resolved via alias

        candidates = self.did_you_mean(name)  # suggestions for a name we could not match
        contained = self._containing(name)  # drugs whose names contain the text as a word or prefix
        if _ABBREVIATION.fullmatch(name):  # short all-caps text is ambiguous by nature
            question = (f"'{name}' is a short abbreviation, which can mean more than one thing. "
                        "Which medication do you mean?")  # ask the person to confirm
            return Resolution(AMBIGUOUS_NAME, name, dose=dose, candidates=candidates, question=question)
        if len(contained) > 1:  # the text fits several drugs
            question = f"'{name}' could be more than one drug in our data. Which do you mean?"  # ask which one
            return Resolution(AMBIGUOUS_NAME, name, dose=dose, candidates=contained, question=question)
        if candidates:  # we have suggestions but will not auto-pick
            question = (f"'{name}' is not a drug name in our data. Did you mean "
                        f"{', '.join(candidates)}? Please confirm; we do not guess.")  # offer them for confirmation
        else:  # nothing even close
            question = f"'{name}' is not a drug name in our data."  # plain unknown message
        return Resolution(UNKNOWN_DRUG, name, dose=dose, candidates=candidates, question=question)

    def _containing(self, text):  # find names that contain the text as a word or word prefix
        """Names that contain the text as a whole word or word prefix (e.g. 'valpro' -> 2 drugs)."""
        key = text.casefold()  # case-insensitive comparison
        if len(key) < 3:  # too short to be a meaningful prefix
            return []  # report nothing
        hits = []  # names that match
        for name in self._names:  # scan every drug name
            if any(word.startswith(key) for word in re.split(r"[\s/(),]+", name.casefold())):  # any word starts with the text
                hits.append(name)  # keep this name
        return hits  # the matching names

    def did_you_mean(self, text, limit=5):  # build the suggestion list for an unknown name
        key = text.casefold()  # case-insensitive key
        found = []  # suggestions, in order
        for name in self._containing(text):  # first, names containing the text
            found.append(name)  # add each
        for match in difflib.get_close_matches(key, list(self._by_key), n=limit, cutoff=0.6):  # then spelling-similar names
            canonical = self._by_key[match][1]  # convert back to the official spelling
            if canonical not in found:  # avoid duplicates
                found.append(canonical)  # add it
        if self._suggester is not None:  # optional embedding suggester
            for name in self._suggester(text):  # it can only add candidates
                if name in self._names and name not in found:  # must be a real drug and new
                    found.append(name)  # add it
        return found[:limit]  # cap the list length

# ---------------------------------------------------------------------------
# FILE SUMMARY: The first step of every request. It cleans a typed medication (case, spaces, dose text), matches it
# exactly against the drugs table, optionally through an alias file, and otherwise reports unknown_drug or ambiguous_name
# with did-you-mean suggestions and a clarifying question. It never auto-corrects typos or abbreviations.
# ASSIGNMENT TOOL: supporting infrastructure for the knowledge graph and rule-based core (exact name resolution before
# lookup); its optional suggester hook is where embeddings play their limited role of adding candidates only.
# ---------------------------------------------------------------------------
