"""Free-text questions: the Meeting 8 failure types as explicit components.

Order of work, always: exact name resolution first, then deterministic lookup (pairs,
avoid lists, a drug's interactions), and only then embeddings, and only for free text
nothing else could place. Embeddings propose candidate records; they never decide
whether an interaction exists or how severe it is, and no similarity score is shown.

  ambiguity   -> a clarification step: abbreviations and unknown names get a question, not a guess
  negation    -> explicit rules (not, without, avoid, except ...) answer from the graph's avoid list
  context     -> filter by population only where a record carries that scope; otherwise say so
  adversarial -> the avoid-list route above, so "which pain reliever should I not take with
                 warfarin" never ranks acetaminophen first because of similarity
"""

import re

from ddi_backend import hpo
from ddi_backend.common import DISCLAIMER, NOT_AVAILABLE, NOT_SAFE_NOTE, severity_rank
from ddi_backend.resolver import RESOLVED

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

NEGATION_WORDS = {"not", "without", "avoid", "avoiding", "avoided", "except", "excluding", "never", "cannot"}
_CONTRACTION = re.compile(r"\b\w+n't\b|\b\w+n’t\b", re.IGNORECASE)

# Ordinary question vocabulary: never treated as a drug name or a symptom.
STOPWORDS = set("""
a about above after again all also am an and any anything are as at be because been before being below between
both but by can could did do does doing dont down during each either else enough even ever every few find for
from further get give go going had has have having he her here hers him his how i if in into is it its itself
just know like list make many may me might more most much must my myself need next no nor now of off on once
one only onto or other our ours out over own please same shall she should so some something such take taken takes
taking tell than that the their them then there these they thing think this those through to too under until up
upon us use used using very want was we were what when where whether which while who whom whose why will with
within would yes you your yours
cause causes caused causing effect effects interact interacts interacting interaction interactions interactive
drug drugs medication medications medicine medicines pill pills tablet tablets together combine combined
combining combination mix mixing mixed severe severity serious dangerous danger safe safely risk risky risks
mechanism clinical happen happens happening occur occurs alternative alternatives instead better worse
reliever relievers painkiller painkillers analgesic analgesics nsaid nsaids antibiotic antibiotics
antidepressant antidepressants anticoagulant anticoagulants thinner thinners class classes kind type types
symptom symptoms side warning warnings precaution precautions contraindication contraindications
ingredient ingredients food foods diet drink alcohol grapefruit brand brands name names dose doses dosage dosing
much many often long hours days daily weekly times time sure okay explain describe show tell mean means
child children kid kids pediatric paediatric infant infants baby babies newborn pregnant pregnancy
breastfeeding nursing elderly older geriatric kidney renal liver hepatic patient person people
""".split())
STOPWORDS |= NEGATION_WORDS | {"can't", "won't", "shouldn't", "don't", "isn't"}

# A word right after one of these (skipping determiners) is where a drug name is expected.
_POSITION_WORDS = {"take", "taking", "took", "with", "and", "plus", "on", "using", "use", "mix", "mixing",
                   "combine", "combining", "between", "versus", "vs", "alongside", "do", "does", "did",
                   "of", "starting", "start", "add", "adding"}
_DETERMINERS = {"a", "an", "the", "my", "any", "some", "other", "another", "also", "to", "it", "them", "these", "those"}
_ABBREVIATION_OK = {"OK", "FDA", "DDI", "HPO", "OTC", "US", "UK", "ID"}

_TOPICS = [
    (re.compile(r"\bside[- ]?effects?\b|\badverse (?:effects?|reactions?)\b", re.I), "Side effects"),
    (re.compile(r"\bused for\b|\buses\b|\bindications?\b|\bwhat is [\w' .-]+ for\b", re.I), "Uses"),
    (re.compile(r"\bwarnings?\b|\bprecautions?\b", re.I), "Warnings"),
    (re.compile(r"\bcontraindications?\b|\bcontraindicated\b", re.I), "Contraindications"),
    (re.compile(r"\bingredients?\b", re.I), "Ingredients"),
    (re.compile(r"\bfood\b|\bfoods\b|\bdiet\b|\balcohol\b|\bgrapefruit\b|\bmeals?\b", re.I), "Food interactions"),
    (re.compile(r"\bbrand names?\b|\bbrands?\b", re.I), "Brand names"),
]
_DOSE = re.compile(r"\bdos(?:e|es|age|ing)\b|\bhow much\b|\bhow many (?:mg|milligrams|pills|tablets)\b", re.I)
_CONTEXTS = [
    (re.compile(r"\bchild(?:ren)?\b|\bkids?\b|\bpediatric\b|\bpaediatric\b|\binfants?\b|\bbab(?:y|ies)\b|\bnewborns?\b", re.I), "children"),
    (re.compile(r"\bpregnan(?:t|cy)\b", re.I), "pregnancy"),
    (re.compile(r"\bbreast-?feeding\b|\bnursing\b", re.I), "breastfeeding"),
    (re.compile(r"\belderly\b|\bgeriatric\b|\bolder (?:adults?|people)\b", re.I), "older adults"),
    (re.compile(r"\bkidney\b|\brenal\b", re.I), "kidney disease"),
    (re.compile(r"\bliver\b|\bhepatic\b", re.I), "liver disease"),
]

RULE_NEGATION = "negation_detected"
RULE_CLARIFICATION = "clarification_requested"
RULE_POPULATION = "population_context_not_covered"
RULE_NOT_AVAILABLE = "topic_not_available"
RULE_SYMPTOM_MAPPING = "symptom_mapping"
RULE_CANDIDATES_ONLY = "embedding_candidates_only"
RULE_DOSE = "dose_never_recommended"


def detect_negation(text):
    """The negation words found in the question (empty list when there are none)."""
    found = [t.casefold() for t in _TOKEN.findall(text) if t.casefold() in NEGATION_WORDS]
    found += [m.group(0).casefold() for m in _CONTRACTION.finditer(text)]
    return found


def filter_by_population(records, population):
    """Keep records for a population only where a record carries that scope.

    Our data has no population scope, so normally nothing is filtered and a note says why.
    Returns (records, note); note is None when at least one record carried a scope.
    """
    scoped = [r for r in records if r.get("population")]
    if not scoped:
        return list(records), (f"Our data does not cover {population}: no record carries a population "
                               "scope, so nothing was filtered or adjusted for it.")
    wanted = population.casefold()
    return [r for r in records if (r.get("population") or "").casefold() == wanted], None


def context_question(question, patient_context=None):
    """A clarifying question when the question depends on patient context we were not given.

    We never assume the context. ``patient_context`` maps a context name (for example "children") to
    what the user told us; a missing entry produces the question.
    """
    given = patient_context or {}
    for pattern, context in _CONTEXTS:
        if pattern.search(question) and not given.get(context):
            return (f"Your question mentions {context}. Can you tell me more about that (for example the age "
                    f"or condition)? Our data does not cover {context}, so please also ask a pharmacist or "
                    "clinician.")
    return None


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _legacy_record(record):
    keys = ("id", "source", "severity", "drug_a", "drug_b", "mechanism", "effect",
            "safer_alternative", "rationale", "reference")
    return {k: record.get(k) for k in keys}


class QuestionAnswerer:
    def __init__(self, checker, hpo_conn=None, symptom_mapping=True, embeddings=None, hpo_index=None):
        """``hpo_conn`` is a connection with the HPO tables (None switches symptom mapping off).
        ``embeddings`` is an optional EmbeddingIndex of interaction records and ``hpo_index`` a separate
        one of HPO terms (tier two); both only propose candidates."""
        self.checker = checker
        self.hpo_conn = hpo_conn if hpo.available(hpo_conn) else None
        self.symptom_mapping = symptom_mapping
        self.embeddings = embeddings
        self.hpo_index = hpo_index
        self._terms = checker.resolver.lookup_terms()
        # Words that appear in the interaction text itself are ordinary vocabulary, not drug names.
        self._vocabulary = {w for r in checker.graph.records
                            for field in ("effect", "mechanism", "rationale")
                            for w in re.findall(r"[a-z0-9]+", (r.get(field) or "").casefold())}
        self._max_words = max((len(k.split()) for k in self._terms), default=1)

    # ------------------------------------------------------------------
    # Understanding the question
    # ------------------------------------------------------------------
    def mentions(self, question):
        """Drug mentions in order: [{"text", "resolution", "span"}], resolved or not.

        Known names are matched exactly (longest first). Anything else that sits where a drug
        name is expected is unresolved: a short all-caps abbreviation, a capitalised word that
        is not a known drug, or an unknown word right after take/with/and/on ...
        """
        tokens = [(m.group(0), m.start()) for m in _TOKEN.finditer(question)]
        norms = [_norm(t) for t, _ in tokens]
        found, taken, i = [], set(), 0
        while i < len(tokens):
            matched = 0
            for n in range(min(self._max_words, len(tokens) - i), 0, -1):
                if " ".join(norms[i:i + n]) in self._terms:
                    matched = n
                    break
            if matched:
                text = question[tokens[i][1]: tokens[i + matched - 1][1] + len(tokens[i + matched - 1][0])]
                found.append((i, text))
                taken.update(range(i, i + matched))
                i += matched
            else:
                i += 1

        for i, (token, _) in enumerate(tokens):
            if i in taken:
                continue
            lower = token.casefold()
            unknown = False
            if token.isupper() and token.isalpha() and 2 <= len(token) <= 5 and token not in _ABBREVIATION_OK:
                unknown = True
            elif (lower not in STOPWORDS and token.isalpha() and len(token) >= 3
                  and not self._is_vocabulary(lower)):
                if token[0].isupper() and i > 0 and not token.isupper():
                    unknown = True
                else:
                    j = i - 1
                    while j >= 0 and norms[j] in _DETERMINERS:
                        j -= 1
                    unknown = j >= 0 and norms[j] in _POSITION_WORDS
            if unknown:
                found.append((i, token))
                taken.add(i)

        found.sort()
        result, seen = [], set()
        for index, text in found:
            resolution = self.checker.resolver.resolve(text)
            key = resolution.name.casefold() if resolution.resolved else _norm(text)
            if key in seen:
                continue
            seen.add(key)
            result.append({"text": text, "resolution": resolution, "index": index})
        return result, taken, tokens

    def _is_vocabulary(self, word):
        """A word our own data uses as ordinary text (an effect, a mechanism) or as an HPO term."""
        if word in self._vocabulary:
            return True
        return self.hpo_conn is not None and hpo.is_term(self.hpo_conn, word)

    def _symptom_candidates(self, tokens, taken):
        words = []
        for i, (token, _) in enumerate(tokens):
            lower = token.casefold()
            if (i in taken or lower in STOPWORDS or not token.isalpha() or len(token) < 4 or lower in words):
                continue
            words.append(lower)
        return words

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------
    def ask(self, question):
        notes, fired = [], []
        answer = {"question": question, "disclaimer": DISCLAIMER, "notes": notes, "rules_fired": fired}

        for pattern, topic in _TOPICS:
            if pattern.search(question):
                notes.append(f"{topic}: {NOT_AVAILABLE}. Our data holds interactions, not drug monographs.")
                if RULE_NOT_AVAILABLE not in fired:
                    fired.append(RULE_NOT_AVAILABLE)
        for pattern, context in _CONTEXTS:
            if pattern.search(question):
                _, note = filter_by_population([], context)
                notes.append(f"Your question mentions {context}. " + note)
                if RULE_POPULATION not in fired:
                    fired.append(RULE_POPULATION)
        clarifying = context_question(question)
        if clarifying:
            answer["clarifying_question"] = clarifying
        if _DOSE.search(question):
            notes.append("Doses are echoed back but this system never recommends or calculates a dose; "
                         "ask a pharmacist or clinician.")
            fired.append(RULE_DOSE)

        negation = detect_negation(question)
        mentions, taken, tokens = self.mentions(question)
        unresolved = [m for m in mentions if not m["resolution"].resolved]
        clarifications = [
            {"input": m["text"], "status": m["resolution"].status, "question": m["resolution"].question,
             "did_you_mean": m["resolution"].candidates}
            for m in unresolved
        ]
        if clarifications:
            answer["clarifications"] = clarifications
            fired.append(RULE_CLARIFICATION)

        if len(mentions) >= 2:
            self._pair_route(answer, mentions)
        elif len(mentions) == 1:
            mention = mentions[0]
            if not mention["resolution"].resolved:
                answer.update(status="needs_clarification", intent="unresolved")
                notes.append(NOT_SAFE_NOTE)
            elif negation:
                self._avoid_route(answer, mention, negation)
            else:
                self._drug_route(answer, mention, self._symptom_candidates(tokens, taken))
        elif negation:
            question_text = "Which medication do you want to avoid combining with? Please name it."
            answer.update(status="needs_clarification", intent="avoid",
                          clarifications=[{"input": None, "status": "missing_drug", "question": question_text,
                                           "did_you_mean": []}])
            fired.append(RULE_CLARIFICATION)
            fired.append(RULE_NEGATION)
        else:
            self._free_text_route(answer, question, self._symptom_candidates(tokens, taken), negation)

        return answer

    # ---- routes -------------------------------------------------------
    def _pair_route(self, answer, mentions):
        texts = [m["text"] for m in mentions]
        out = self.checker.check(texts[:-1], texts[-1], include_current_pairs=len(texts) > 2)
        answer.update(status="answered", intent="pair", results=out["results"], inputs=out["inputs"],
                      summary=out["summary"])
        answer["notes"].extend(out["notes"])
        for rule in out["rules_fired"]:
            if rule not in answer["rules_fired"]:
                answer["rules_fired"].append(rule)

    def _avoid_route(self, answer, mention, negation):
        out = self.checker.avoid(mention["resolution"].name)
        answer.update(status="answered", intent="avoid", avoid=out, negation_words=sorted(set(negation)))
        answer["rules_fired"].append(RULE_NEGATION)
        answer["rules_fired"].extend(r for r in out["rules_fired"] if r not in answer["rules_fired"])
        answer["notes"].extend(out["notes"])

    def _drug_route(self, answer, mention, symptom_words):
        out = self.checker.drug(mention["resolution"].name)
        interactions = out["interactions"]
        answer.update(status="answered", intent="drug", drug=out["drug"]["name"])
        answer["rules_fired"].append("exact_name_resolution")
        if symptom_words:
            filtered = self._match_symptoms(answer, interactions, symptom_words)
            if filtered:
                interactions = filtered
            else:
                answer["notes"].append("No interaction text for this drug mentions that symptom; "
                                       "all of its documented interactions are listed instead.")
        answer["interactions"] = sorted(interactions, key=lambda r: (severity_rank(r["severity"]), r["id"]))
        if out["excluded_evidence"]:
            answer["excluded_evidence"] = out["excluded_evidence"]
            answer["needs_review"] = True
        answer["not_available"] = out["not_available"]

    def _free_text_route(self, answer, question, symptom_words, negation):
        records = self.checker.eligible_records(self.checker.graph.records)[0]
        if symptom_words:
            matched = self._match_symptoms(answer, records, symptom_words)
            if matched:
                answer.update(status="answered", intent="symptom", interactions=matched)
                return
        if self.embeddings is not None:
            candidates = self.embeddings.candidates(question, k=5)
            eligible, _ = self.checker.eligible_records([c for c in candidates if c is not None])
            if eligible:
                answer.update(status="candidates_only", intent="free_text",
                              candidates=[_legacy_record(r) for r in eligible])
                answer["rules_fired"].append(RULE_CANDIDATES_ONLY)
                answer["notes"].append(
                    "These records look related to your wording. Text similarity finds candidates only; it "
                    "does not decide whether an interaction exists or how severe it is. Ask again with exact "
                    "drug names to get a documented answer.")
                return
        answer.update(status="no_match", intent="none")
        answer["notes"].append(
            "I could not identify a drug from the exact names in our data, so nothing was looked up. "
            "Ask with drug names, for example 'Do warfarin and ibuprofen interact?'. " + NOT_SAFE_NOTE)

    def _match_symptoms(self, answer, records, symptom_words):
        """Records whose effect or mechanism text matches a symptom word, Major first."""
        mapping = self.symptom_mapping and self.hpo_conn is not None
        if not self.symptom_mapping:
            answer["notes"].append("Symptom mapping is off, so only the literal words you typed were matched.")
        by_id, phenotypes = {}, {}
        for word in symptom_words:
            expansion = hpo.expand_symptom(self.hpo_conn, word) if mapping else None
            if mapping and not expansion["terms"] and self.hpo_index is not None:
                # No label or synonym matched: let the HPO embedding index propose candidate terms.
                expansion = hpo.expand_symptom(self.hpo_conn, word, self.hpo_index.nearest_ids(word, k=3))
            for record, matched_on, phrase, helper in hpo.find_in_text(records, word, expansion):
                item = dict(record, matched_on=matched_on, matched_phrase=phrase)
                if helper:
                    item["hpo_term_helped"] = helper
                by_id.setdefault(record["id"], item)
            if expansion and by_id:
                for term in expansion["terms"]:
                    phenotypes.setdefault(term["id"], term)
        if not by_id:
            return []
        answer["rules_fired"].append(RULE_SYMPTOM_MAPPING if mapping else "literal_symptom_match")
        answer["notes"].append(
            "Results matched on the effect text (or the mechanism text) of the interaction records"
            + ("; the HPO terms only helped find them." if mapping else "."))
        if mapping and phenotypes:
            answer["phenotype_records"] = list(phenotypes.values())
            answer["phenotype_note"] = hpo.PHENOTYPE_NOTE
        return sorted(by_id.values(), key=lambda r: (severity_rank(r["severity"]), r["id"]))


def candidate_records(conn, query):
    """Records for a question in the old record shape, for the selector and the offline lookup.

    The same rules apply as in QuestionAnswerer; HPO is not mixed in. Most severe first.
    """
    from ddi_backend.checker import Checker  # imported here: checker pulls in more than this module needs
    checker = Checker(conn)
    answer = QuestionAnswerer(checker, hpo_conn=conn).ask(query)
    records, seen = [], set()

    def add(record):
        if record["id"] not in seen:
            seen.add(record["id"])
            records.append(record)

    if answer.get("intent") == "avoid":
        for entry in answer["avoid"]["avoid"]:
            for record_id in entry["record_ids"]:
                add(next(r for r in checker.graph.records if r["id"] == record_id))
    elif answer.get("intent") == "pair":
        for result in answer["results"]:
            for record in result.get("records", []):
                add(record)
    else:
        for record in answer.get("interactions", []):
            add(record)
    records.sort(key=lambda r: (severity_rank(r["severity"]), r["id"]))
    return [_legacy_record(r) for r in records]
