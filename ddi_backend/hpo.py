# Module docstring: describes HPO's limited role (vocabulary used to widen a symptom word, never evidence).
"""Symptom words -> HPO terms -> words to look for in interaction text.

HPO is a vocabulary, not interaction evidence: it says nothing about which drugs
cause which phenotypes. It is used only to expand a symptom word (through term
labels, synonyms including layperson synonyms, and direct child terms) into
phrases to search for in the effect and mechanism text of the interaction rows.
The interaction answer always rests on that text; the HPO term only helped find it.
"""

import re  # whole-word matching with regular expressions

from ddi_backend import load_data  # table_names() helper for checking which tables exist

HPO_TABLES = {"hpo_terms", "hpo_synonyms", "hpo_edges"}  # the three tables symptom mapping needs
MAX_SEED_TERMS = 25  # cap on how many HPO terms one symptom word may start from
MIN_PHRASE_LENGTH = 4  # phrases shorter than this are too vague to search for
# Single words that would match far too much if searched for on their own.
GENERIC_WORDS = {  # phrases that are never used as search words
    "abnormal", "abnormality", "abnormalities", "increased", "decreased", "sign", "symptom", "syndrome",  # vague medical words
    "disease", "disorder", "finding", "onset", "episode", "episodes", "other", "unspecified", "type",  # more vague words
    "mild", "severe", "acute", "chronic", "recurrent", "tendency",  # severity and timing words
}
PHENOTYPE_NOTE = "Phenotype records from the Human Phenotype Ontology (vocabulary, not interaction evidence)."  # label shown beside HPO records


def available(conn):  # is symptom mapping possible with this database?
    """True when the connection has the HPO tables (symptom mapping can switch on)."""
    if conn is None:  # no database connection at all
        return False  # symptom mapping is off
    try:  # listing tables might fail
        return HPO_TABLES <= set(load_data.table_names(conn))  # true only if all three HPO tables exist
    except Exception:  # any database problem
        return False  # treat as unavailable rather than crash


def _contains_word(text, word):  # helper: does text contain word as a whole word?
    # Builds a pattern that refuses to match inside a longer word, ignoring case.
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text.casefold()) is not None


def seed_terms(conn, word, limit=MAX_SEED_TERMS):  # find the HPO terms a symptom word starts from
    """Non-obsolete HPO terms whose label or a synonym is the word, else contains it as a whole word."""
    key = word.casefold()  # compare case-insensitively
    # First try an exact match on a term label or a synonym; the SQL string below is left untouched.
    exact = conn.execute(
        """SELECT DISTINCT t.hp_id, t.label FROM hpo_terms t
           LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
           WHERE t.is_obsolete = 0 AND (LOWER(t.label) = ? OR LOWER(s.synonym) = ?)
           ORDER BY LENGTH(t.label), t.label""",
        (key, key),  # the word is passed twice, once per placeholder
    ).fetchall()  # fetch every matching (hp_id, label) row
    if exact:  # an exact hit beats a loose one
        return exact[:limit]  # return at most `limit` terms
    like = f"%{key}%"  # pattern for the looser "contains" search
    # Otherwise look for terms whose label or synonym merely contains the word.
    rows = conn.execute(
        """SELECT DISTINCT t.hp_id, t.label, s.synonym FROM hpo_terms t
           LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
           WHERE t.is_obsolete = 0 AND (LOWER(t.label) LIKE ? OR LOWER(s.synonym) LIKE ?)
           ORDER BY LENGTH(t.label), t.label""",
        (like, like),  # same pattern for both placeholders
    ).fetchall()  # fetch the loose matches
    seen, terms = set(), []  # seen guards against duplicates; terms is the result list
    for hp_id, label, synonym in rows:  # check each loose match
        # Skip repeats and rows where the word is only part of a longer word.
        if hp_id in seen or not (_contains_word(label, key) or (synonym and _contains_word(synonym, key))):
            continue  # not a real whole-word match
        seen.add(hp_id)  # remember this term
        terms.append((hp_id, label))  # keep it
    return terms[:limit]  # cap the list


def is_term(conn, word):  # is the word itself an HPO term name?
    """True when the word is exactly the label or a synonym of a non-obsolete HPO term."""
    key = word.casefold()  # compare case-insensitively
    return conn.execute(  # run a one-row existence query
        """SELECT 1 FROM hpo_terms t LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
           WHERE t.is_obsolete = 0 AND (LOWER(t.label) = ? OR LOWER(s.synonym) = ?) LIMIT 1""",
        (key, key),  # the word for each placeholder
    ).fetchone() is not None  # a returned row means it is a term


def documents(conn):  # text documents for the separate HPO embedding index
    """[(hp_id, "label; synonym; ...")] for every non-obsolete term, for the separate HPO embedding index."""
    synonyms = {}  # hp_id -> list of synonyms
    for hp_id, synonym in conn.execute("SELECT hp_id, synonym FROM hpo_synonyms ORDER BY synonym_id"):  # every synonym in stable order
        synonyms.setdefault(hp_id, []).append(synonym)  # group synonyms under their term
    rows = conn.execute("SELECT hp_id, label FROM hpo_terms WHERE is_obsolete = 0 ORDER BY hp_id").fetchall()  # live terms in a stable order
    return [(hp_id, "; ".join([label] + synonyms.get(hp_id, []))) for hp_id, label in rows]  # label plus synonyms joined into one text


def expand_symptom(conn, word, extra_seed_ids=()):  # turn a symptom word into search phrases
    """Expand a symptom word into searchable phrases, each tagged with the HPO term that supplied it.

    ``extra_seed_ids`` are HPO IDs proposed by the embedding index for a word that matched no
    label or synonym; they are expanded the same way (candidates, not decisions).
    Returns {"word", "terms": [phenotype records], "phrases": [{"phrase", "hp_id", "label", "via"}]}.
    """
    terms, phrases, seen = [], [], set()  # phenotype records, search phrases, and phrases already added

    def add(phrase, hp_id, label, via):  # inner helper that records one search phrase
        phrase = " ".join(phrase.casefold().split())  # lower-case and collapse whitespace
        if len(phrase) < MIN_PHRASE_LENGTH or phrase in GENERIC_WORDS or phrase in seen:  # too short, too vague, or a repeat
            return  # skip it
        seen.add(phrase)  # remember it so it is not added twice
        phrases.append({"phrase": phrase, "hp_id": hp_id, "label": label, "via": via})  # keep it with where it came from

    seeds = list(seed_terms(conn, word))  # start from the terms that match the word
    for hp_id in extra_seed_ids:  # add terms suggested by the embedding index
        # Look the suggested term up, ignoring obsolete ones.
        row = conn.execute("SELECT hp_id, label FROM hpo_terms WHERE hp_id = ? AND is_obsolete = 0", (hp_id,)).fetchone()
        if row and row not in seeds:  # a real, new term
            seeds.append(row)  # treat it like any other seed
    for hp_id, label in seeds:  # expand each seed term
        definition, comment = conn.execute(  # fetch the term's description fields
            "SELECT definition, comment FROM hpo_terms WHERE hp_id = ?", (hp_id,)).fetchone()  # one row per term
        # Build the phenotype record shown to the user, clearly labelled as HPO vocabulary.
        terms.append({"id": f"hpo:{hp_id}", "source": "Human Phenotype Ontology", "kind": "phenotype record",
                      "hp_id": hp_id, "label": label, "definition": definition, "comment": comment})
        add(label, hp_id, label, "term label")  # the term's own label is a search phrase
        for synonym, synonym_type in conn.execute(  # every synonym of the term
                "SELECT synonym, synonym_type FROM hpo_synonyms WHERE hp_id = ?", (hp_id,)):
            add(synonym, hp_id, label, "layperson synonym" if synonym_type == "layperson" else "synonym")  # tag layperson wording separately
        # Direct child terms (more specific symptoms); the SQL string is left untouched.
        for (child_label,) in conn.execute(
                """SELECT t.label FROM hpo_edges e JOIN hpo_terms t ON t.hp_id = e.child_id
                   WHERE e.parent_id = ? AND t.is_obsolete = 0""", (hp_id,)):
            add(child_label, hp_id, label, "child term")  # child labels widen the search
    return {"word": word, "terms": terms, "phrases": phrases}  # everything the caller needs


def find_in_text(records, word, expansion=None):  # search interaction text with the word and its phrases
    """Interaction records whose effect or mechanism text contains the word or an expansion phrase.

    Returns [(record, matched_on, phrase, hpo_info_or_None)] with Major first. ``matched_on`` says
    which text matched; ``hpo_info`` is set only when the match came through an HPO phrase.
    """
    candidates = [(word.casefold(), None)]  # the user's own word comes first, with no HPO info
    for phrase in (expansion or {}).get("phrases", []):  # then every expansion phrase, if there is one
        candidates.append((phrase["phrase"], phrase))  # keep the phrase together with its HPO details
    matches = []  # results
    for record in records:  # check each interaction record
        for field, label in (("effect", "effect text"), ("mechanism", "mechanism text")):  # the only two fields searched
            text = record.get(field) or ""  # the stored text, or empty
            # First candidate (word, then phrases) found as a whole word in this text.
            hit = next(((p, info) for p, info in candidates if _contains_word(text, p)), None)
            if hit:  # something matched
                phrase, info = hit  # split the match into phrase and HPO info
                helper = None if info is None else {"hp_id": info["hp_id"], "label": info["label"], "via": info["via"]}  # HPO detail only if matched through HPO
                matches.append((record, label, phrase, helper))  # record which text matched and how
                break  # one match per record is enough
    return matches  # the caller sorts and presents them


# ---------------------------------------------------------------------------
# FILE SUMMARY: Expands a symptom word into related HPO labels, synonyms (including layperson wording) and child
# terms, then searches the effect and mechanism text of interaction records for those phrases. HPO only helps find
# text; the answer always rests on the interaction record, and HPO terms are returned separately as phenotype records.
# ASSIGNMENT TOOL: Embeddings in a limited role / retrieval support: the symptom-mapping helper next to the
# knowledge-graph answers. HPO is a vocabulary, not evidence, and no piece-splitting retrieval (RAG) is implemented.
# ---------------------------------------------------------------------------
