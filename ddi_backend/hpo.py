"""Symptom words -> HPO terms -> words to look for in interaction text.

HPO is a vocabulary, not interaction evidence: it says nothing about which drugs
cause which phenotypes. It is used only to expand a symptom word (through term
labels, synonyms including layperson synonyms, and direct child terms) into
phrases to search for in the effect and mechanism text of the interaction rows.
The interaction answer always rests on that text; the HPO term only helped find it.
"""

import re

from ddi_backend import load_data

HPO_TABLES = {"hpo_terms", "hpo_synonyms", "hpo_edges"}
MAX_SEED_TERMS = 25
MIN_PHRASE_LENGTH = 4
# Single words that would match far too much if searched for on their own.
GENERIC_WORDS = {
    "abnormal", "abnormality", "abnormalities", "increased", "decreased", "sign", "symptom", "syndrome",
    "disease", "disorder", "finding", "onset", "episode", "episodes", "other", "unspecified", "type",
    "mild", "severe", "acute", "chronic", "recurrent", "tendency",
}
PHENOTYPE_NOTE = "Phenotype records from the Human Phenotype Ontology (vocabulary, not interaction evidence)."


def available(conn):
    """True when the connection has the HPO tables (symptom mapping can switch on)."""
    if conn is None:
        return False
    try:
        return HPO_TABLES <= set(load_data.table_names(conn))
    except Exception:
        return False


def _contains_word(text, word):
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text.casefold()) is not None


def seed_terms(conn, word, limit=MAX_SEED_TERMS):
    """Non-obsolete HPO terms whose label or a synonym is the word, else contains it as a whole word."""
    key = word.casefold()
    exact = conn.execute(
        """SELECT DISTINCT t.hp_id, t.label FROM hpo_terms t
           LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
           WHERE t.is_obsolete = 0 AND (LOWER(t.label) = ? OR LOWER(s.synonym) = ?)
           ORDER BY LENGTH(t.label), t.label""",
        (key, key),
    ).fetchall()
    if exact:
        return exact[:limit]
    like = f"%{key}%"
    rows = conn.execute(
        """SELECT DISTINCT t.hp_id, t.label, s.synonym FROM hpo_terms t
           LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
           WHERE t.is_obsolete = 0 AND (LOWER(t.label) LIKE ? OR LOWER(s.synonym) LIKE ?)
           ORDER BY LENGTH(t.label), t.label""",
        (like, like),
    ).fetchall()
    seen, terms = set(), []
    for hp_id, label, synonym in rows:
        if hp_id in seen or not (_contains_word(label, key) or (synonym and _contains_word(synonym, key))):
            continue
        seen.add(hp_id)
        terms.append((hp_id, label))
    return terms[:limit]


def is_term(conn, word):
    """True when the word is exactly the label or a synonym of a non-obsolete HPO term."""
    key = word.casefold()
    return conn.execute(
        """SELECT 1 FROM hpo_terms t LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
           WHERE t.is_obsolete = 0 AND (LOWER(t.label) = ? OR LOWER(s.synonym) = ?) LIMIT 1""",
        (key, key),
    ).fetchone() is not None


def documents(conn):
    """[(hp_id, "label; synonym; ...")] for every non-obsolete term, for the separate HPO embedding index."""
    synonyms = {}
    for hp_id, synonym in conn.execute("SELECT hp_id, synonym FROM hpo_synonyms ORDER BY synonym_id"):
        synonyms.setdefault(hp_id, []).append(synonym)
    rows = conn.execute("SELECT hp_id, label FROM hpo_terms WHERE is_obsolete = 0 ORDER BY hp_id").fetchall()
    return [(hp_id, "; ".join([label] + synonyms.get(hp_id, []))) for hp_id, label in rows]


def expand_symptom(conn, word, extra_seed_ids=()):
    """Expand a symptom word into searchable phrases, each tagged with the HPO term that supplied it.

    ``extra_seed_ids`` are HPO IDs proposed by the embedding index for a word that matched no
    label or synonym; they are expanded the same way (candidates, not decisions).
    Returns {"word", "terms": [phenotype records], "phrases": [{"phrase", "hp_id", "label", "via"}]}.
    """
    terms, phrases, seen = [], [], set()

    def add(phrase, hp_id, label, via):
        phrase = " ".join(phrase.casefold().split())
        if len(phrase) < MIN_PHRASE_LENGTH or phrase in GENERIC_WORDS or phrase in seen:
            return
        seen.add(phrase)
        phrases.append({"phrase": phrase, "hp_id": hp_id, "label": label, "via": via})

    seeds = list(seed_terms(conn, word))
    for hp_id in extra_seed_ids:
        row = conn.execute("SELECT hp_id, label FROM hpo_terms WHERE hp_id = ? AND is_obsolete = 0", (hp_id,)).fetchone()
        if row and row not in seeds:
            seeds.append(row)
    for hp_id, label in seeds:
        definition, comment = conn.execute(
            "SELECT definition, comment FROM hpo_terms WHERE hp_id = ?", (hp_id,)).fetchone()
        terms.append({"id": f"hpo:{hp_id}", "source": "Human Phenotype Ontology", "kind": "phenotype record",
                      "hp_id": hp_id, "label": label, "definition": definition, "comment": comment})
        add(label, hp_id, label, "term label")
        for synonym, synonym_type in conn.execute(
                "SELECT synonym, synonym_type FROM hpo_synonyms WHERE hp_id = ?", (hp_id,)):
            add(synonym, hp_id, label, "layperson synonym" if synonym_type == "layperson" else "synonym")
        for (child_label,) in conn.execute(
                """SELECT t.label FROM hpo_edges e JOIN hpo_terms t ON t.hp_id = e.child_id
                   WHERE e.parent_id = ? AND t.is_obsolete = 0""", (hp_id,)):
            add(child_label, hp_id, label, "child term")
    return {"word": word, "terms": terms, "phrases": phrases}


def find_in_text(records, word, expansion=None):
    """Interaction records whose effect or mechanism text contains the word or an expansion phrase.

    Returns [(record, matched_on, phrase, hpo_info_or_None)] with Major first. ``matched_on`` says
    which text matched; ``hpo_info`` is set only when the match came through an HPO phrase.
    """
    candidates = [(word.casefold(), None)]
    for phrase in (expansion or {}).get("phrases", []):
        candidates.append((phrase["phrase"], phrase))
    matches = []
    for record in records:
        for field, label in (("effect", "effect text"), ("mechanism", "mechanism text")):
            text = record.get(field) or ""
            hit = next(((p, info) for p, info in candidates if _contains_word(text, p)), None)
            if hit:
                phrase, info = hit
                helper = None if info is None else {"hp_id": info["hp_id"], "label": info["label"], "via": info["via"]}
                matches.append((record, label, phrase, helper))
                break
    return matches
