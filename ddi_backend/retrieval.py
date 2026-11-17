# Module docstring: explains the order of work and the four failure types (Meeting 8) handled as explicit components.
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

import re  # regular expressions for tokenising and topic detection

from ddi_backend import hpo  # symptom word expansion and text search
from ddi_backend.common import DISCLAIMER, NOT_AVAILABLE, NOT_SAFE_NOTE, severity_rank  # fixed wording and the Major-first sort key
from ddi_backend.resolver import RESOLVED  # status value for a successfully matched drug name

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")  # a word, optionally with an apostrophe ending (don't)

NEGATION_WORDS = {"not", "without", "avoid", "avoiding", "avoided", "except", "excluding", "never", "cannot"}  # words that flip the question to "what to avoid"
_CONTRACTION = re.compile(r"\b\w+n't\b|\b\w+n’t\b", re.IGNORECASE)  # negative contractions such as shouldn't (straight or curly apostrophe)

# Ordinary question vocabulary: never treated as a drug name or a symptom.
# One big string of words, split into a set below; no notes can go inside the string.
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
STOPWORDS |= NEGATION_WORDS | {"can't", "won't", "shouldn't", "don't", "isn't"}  # negation words and contractions are also not drug names

# A word right after one of these (skipping determiners) is where a drug name is expected.
_POSITION_WORDS = {"take", "taking", "took", "with", "and", "plus", "on", "using", "use", "mix", "mixing",  # verbs and joiners that precede a drug
                   "combine", "combining", "between", "versus", "vs", "alongside", "do", "does", "did",  # more joiners
                   "of", "starting", "start", "add", "adding"}  # and a few more
_DETERMINERS = {"a", "an", "the", "my", "any", "some", "other", "another", "also", "to", "it", "them", "these", "those"}  # small words skipped when looking back for a position word
_ABBREVIATION_OK = {"OK", "FDA", "DDI", "HPO", "OTC", "US", "UK", "ID"}  # capital-letter words that are not unknown drugs

_TOPICS = [  # question topics our data cannot answer (it holds interactions, not monographs)
    (re.compile(r"\bside[- ]?effects?\b|\badverse (?:effects?|reactions?)\b", re.I), "Side effects"),  # side effects
    (re.compile(r"\bused for\b|\buses\b|\bindications?\b|\bwhat is [\w' .-]+ for\b", re.I), "Uses"),  # what a drug is for
    (re.compile(r"\bwarnings?\b|\bprecautions?\b", re.I), "Warnings"),  # warnings and precautions
    (re.compile(r"\bcontraindications?\b|\bcontraindicated\b", re.I), "Contraindications"),  # contraindications
    (re.compile(r"\bingredients?\b", re.I), "Ingredients"),  # ingredients
    (re.compile(r"\bfood\b|\bfoods\b|\bdiet\b|\balcohol\b|\bgrapefruit\b|\bmeals?\b", re.I), "Food interactions"),  # food and drink
    (re.compile(r"\bbrand names?\b|\bbrands?\b", re.I), "Brand names"),  # brand names
]
_DOSE = re.compile(r"\bdos(?:e|es|age|ing)\b|\bhow much\b|\bhow many (?:mg|milligrams|pills|tablets)\b", re.I)  # dose questions, which are never answered
_CONTEXTS = [  # patient populations our data has no scope for
    (re.compile(r"\bchild(?:ren)?\b|\bkids?\b|\bpediatric\b|\bpaediatric\b|\binfants?\b|\bbab(?:y|ies)\b|\bnewborns?\b", re.I), "children"),  # children
    (re.compile(r"\bpregnan(?:t|cy)\b", re.I), "pregnancy"),  # pregnancy
    (re.compile(r"\bbreast-?feeding\b|\bnursing\b", re.I), "breastfeeding"),  # breastfeeding
    (re.compile(r"\belderly\b|\bgeriatric\b|\bolder (?:adults?|people)\b", re.I), "older adults"),  # older adults
    (re.compile(r"\bkidney\b|\brenal\b", re.I), "kidney disease"),  # kidney
    (re.compile(r"\bliver\b|\bhepatic\b", re.I), "liver disease"),  # liver
]

RULE_NEGATION = "negation_detected"  # rule name: the question contained a negation word
RULE_CLARIFICATION = "clarification_requested"  # rule name: a name needed clarifying
RULE_POPULATION = "population_context_not_covered"  # rule name: a population our data does not cover
RULE_NOT_AVAILABLE = "topic_not_available"  # rule name: a topic our data does not hold
RULE_SYMPTOM_MAPPING = "symptom_mapping"  # rule name: HPO helped find the text match
RULE_CANDIDATES_ONLY = "embedding_candidates_only"  # rule name: only similarity candidates were found
RULE_DOSE = "dose_never_recommended"  # rule name: a dose was asked about


def detect_negation(text):  # find negation words in a question
    """The negation words found in the question (empty list when there are none)."""
    found = [t.casefold() for t in _TOKEN.findall(text) if t.casefold() in NEGATION_WORDS]  # plain negation words
    found += [m.group(0).casefold() for m in _CONTRACTION.finditer(text)]  # plus negative contractions
    return found  # empty when none


def filter_by_population(records, population):  # keep records for a population, where the data allows
    """Keep records for a population only where a record carries that scope.

    Our data has no population scope, so normally nothing is filtered and a note says why.
    Returns (records, note); note is None when at least one record carried a scope.
    """
    scoped = [r for r in records if r.get("population")]  # records that carry a population scope
    if not scoped:  # none do (the normal case)
        return list(records), (f"Our data does not cover {population}: no record carries a population "  # say so instead of guessing
                               "scope, so nothing was filtered or adjusted for it.")  # and filter nothing
    wanted = population.casefold()  # the population to keep
    return [r for r in records if (r.get("population") or "").casefold() == wanted], None  # keep only matching records


def context_question(question, patient_context=None):  # ask for missing patient context
    """A clarifying question when the question depends on patient context we were not given.

    We never assume the context. ``patient_context`` maps a context name (for example "children") to
    what the user told us; a missing entry produces the question.
    """
    given = patient_context or {}  # what the user has told us so far
    for pattern, context in _CONTEXTS:  # each population we know about
        if pattern.search(question) and not given.get(context):  # mentioned but not explained
            return (f"Your question mentions {context}. Can you tell me more about that (for example the age "  # ask rather than assume
                    f"or condition)? Our data does not cover {context}, so please also ask a pharmacist or "  # and say our data lacks it
                    "clinician.")  # point to a professional
    return None  # nothing to ask


def _norm(text):  # normalise text for matching
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()  # lower-case and turn punctuation into single spaces


def _legacy_record(record):  # reduce a record to the old, smaller shape
    keys = ("id", "source", "severity", "drug_a", "drug_b", "mechanism", "effect",  # the fields the selector chat expects
            "safer_alternative", "rationale", "reference")  # remaining fields
    return {k: record.get(k) for k in keys}  # a plain dictionary with just those fields


class QuestionAnswerer:  # answers a free-text question using the checker, HPO and (optionally) embeddings
    def __init__(self, checker, hpo_conn=None, symptom_mapping=True, embeddings=None, hpo_index=None):  # wire up the helpers
        """``hpo_conn`` is a connection with the HPO tables (None switches symptom mapping off).
        ``embeddings`` is an optional EmbeddingIndex of interaction records and ``hpo_index`` a separate
        one of HPO terms (tier two); both only propose candidates."""
        self.checker = checker  # the deterministic main flow
        self.hpo_conn = hpo_conn if hpo.available(hpo_conn) else None  # keep it only if the HPO tables exist
        self.symptom_mapping = symptom_mapping  # user switch for symptom mapping
        self.embeddings = embeddings  # optional record index (tier two)
        self.hpo_index = hpo_index  # optional HPO term index (tier two)
        self._terms = checker.resolver.lookup_terms()  # every known drug name, for exact matching
        # Words that appear in the interaction text itself are ordinary vocabulary, not drug names.
        self._vocabulary = {w for r in checker.graph.records  # every record
                            for field in ("effect", "mechanism", "rationale")  # the free-text fields
                            for w in re.findall(r"[a-z0-9]+", (r.get(field) or "").casefold())}  # each lower-case word
        self._max_words = max((len(k.split()) for k in self._terms), default=1)  # longest drug name, in words

    # ------------------------------------------------------------------
    # Understanding the question
    # ------------------------------------------------------------------
    def mentions(self, question):  # find the drug names in the question
        """Drug mentions in order: [{"text", "resolution", "span"}], resolved or not.

        Known names are matched exactly (longest first). Anything else that sits where a drug
        name is expected is unresolved: a short all-caps abbreviation, a capitalised word that
        is not a known drug, or an unknown word right after take/with/and/on ...
        """
        tokens = [(m.group(0), m.start()) for m in _TOKEN.finditer(question)]  # each word with its start position
        norms = [_norm(t) for t, _ in tokens]  # normalised version of each word
        found, taken, i = [], set(), 0  # mentions found, token positions used, and the scan position
        while i < len(tokens):  # scan left to right
            matched = 0  # how many words matched a known name here
            for n in range(min(self._max_words, len(tokens) - i), 0, -1):  # try the longest name first
                if " ".join(norms[i:i + n]) in self._terms:  # exact known name
                    matched = n  # remember its length
                    break  # longest match wins
            if matched:  # a known name starts here
                text = question[tokens[i][1]: tokens[i + matched - 1][1] + len(tokens[i + matched - 1][0])]  # the original wording from the question
                found.append((i, text))  # record the mention
                taken.update(range(i, i + matched))  # mark those words as used
                i += matched  # skip past the name
            else:  # no known name here
                i += 1  # move on one word

        for i, (token, _) in enumerate(tokens):  # second pass: words that look like unknown drug names
            if i in taken:  # already part of a known name
                continue  # skip
            lower = token.casefold()  # lower-case form
            unknown = False  # assume it is not a drug name
            if token.isupper() and token.isalpha() and 2 <= len(token) <= 5 and token not in _ABBREVIATION_OK:  # a short all-capitals word such as an abbreviation
                unknown = True  # treat as an unresolved name
            elif (lower not in STOPWORDS and token.isalpha() and len(token) >= 3  # a real word that is not ordinary vocabulary
                  and not self._is_vocabulary(lower)):  # and not used as text in our data or HPO
                if token[0].isupper() and i > 0 and not token.isupper():  # capitalised mid-sentence: probably a name
                    unknown = True  # treat as an unresolved name
                else:  # otherwise look at the words before it
                    j = i - 1  # start with the previous word
                    while j >= 0 and norms[j] in _DETERMINERS:  # skip small words like "the"
                        j -= 1  # step back
                    unknown = j >= 0 and norms[j] in _POSITION_WORDS  # a name is expected right after words like "take" or "with"
            if unknown:  # we decided it looks like a drug name
                found.append((i, token))  # keep it as an unresolved mention
                taken.add(i)  # mark it used

        found.sort()  # restore question order
        result, seen = [], set()  # final mentions, and a set to drop repeats
        for index, text in found:  # each mention
            resolution = self.checker.resolver.resolve(text)  # try exact name resolution
            key = resolution.name.casefold() if resolution.resolved else _norm(text)  # repeat detection key
            if key in seen:  # the same drug mentioned twice
                continue  # keep only the first
            seen.add(key)  # remember it
            result.append({"text": text, "resolution": resolution, "index": index})  # keep the mention with its resolution
        return result, taken, tokens  # mentions plus data the caller needs for symptom words

    def _is_vocabulary(self, word):  # is this word ordinary text rather than a drug name?
        """A word our own data uses as ordinary text (an effect, a mechanism) or as an HPO term."""
        if word in self._vocabulary:  # appears in interaction text
            return True  # ordinary word
        return self.hpo_conn is not None and hpo.is_term(self.hpo_conn, word)  # or is an HPO term

    def _symptom_candidates(self, tokens, taken):  # words that might be symptoms
        words = []  # collected lower-case words
        for i, (token, _) in enumerate(tokens):  # each word
            lower = token.casefold()  # lower-case form
            if (i in taken or lower in STOPWORDS or not token.isalpha() or len(token) < 4 or lower in words):  # skip drug names, ordinary words, short words and repeats
                continue  # skip it
            words.append(lower)  # keep it
        return words  # candidate symptom words

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------
    def ask(self, question):  # the main entry point for a question
        notes, fired = [], []  # notes for the user, and the names of the rules that fired
        answer = {"question": question, "disclaimer": DISCLAIMER, "notes": notes, "rules_fired": fired}  # start the answer

        for pattern, topic in _TOPICS:  # topics our data cannot answer
            if pattern.search(question):  # the question mentions one
                notes.append(f"{topic}: {NOT_AVAILABLE}. Our data holds interactions, not drug monographs.")  # say it is not available
                if RULE_NOT_AVAILABLE not in fired:  # record the rule once
                    fired.append(RULE_NOT_AVAILABLE)  # note the rule
        for pattern, context in _CONTEXTS:  # populations our data does not cover
            if pattern.search(question):  # the question mentions one
                _, note = filter_by_population([], context)  # get the standard "not covered" note
                notes.append(f"Your question mentions {context}. " + note)  # add it
                if RULE_POPULATION not in fired:  # record the rule once
                    fired.append(RULE_POPULATION)  # note the rule
        clarifying = context_question(question)  # ask for missing context
        if clarifying:  # there is something to ask
            answer["clarifying_question"] = clarifying  # attach it
        if _DOSE.search(question):  # a dose question
            notes.append("Doses are echoed back but this system never recommends or calculates a dose; "  # explain the limit
                         "ask a pharmacist or clinician.")  # point to a professional
            fired.append(RULE_DOSE)  # note the rule

        negation = detect_negation(question)  # negation words, if any
        mentions, taken, tokens = self.mentions(question)  # drug mentions in the question
        unresolved = [m for m in mentions if not m["resolution"].resolved]  # names we could not match exactly
        clarifications = [  # a question for each unresolved name
            {"input": m["text"], "status": m["resolution"].status, "question": m["resolution"].question,  # what was typed and why it failed
             "did_you_mean": m["resolution"].candidates}  # suggested names (never auto-chosen)
            for m in unresolved  # every unresolved mention
        ]
        if clarifications:  # at least one
            answer["clarifications"] = clarifications  # attach them
            fired.append(RULE_CLARIFICATION)  # note the rule

        if len(mentions) >= 2:  # two or more drugs: a pair question
            self._pair_route(answer, mentions)  # answer through the checker
        elif len(mentions) == 1:  # exactly one drug
            mention = mentions[0]  # the single mention
            if not mention["resolution"].resolved:  # not matched
                answer.update(status="needs_clarification", intent="unresolved")  # ask instead of guessing
                notes.append(NOT_SAFE_NOTE)  # and never imply safety
            elif negation:  # "which should I not take with X"
                self._avoid_route(answer, mention, negation)  # answer from the avoid list
            else:  # a plain question about one drug
                self._drug_route(answer, mention, self._symptom_candidates(tokens, taken))  # list its interactions, optionally by symptom
        elif negation:  # negation but no drug named
            question_text = "Which medication do you want to avoid combining with? Please name it."  # what to ask
            answer.update(status="needs_clarification", intent="avoid",  # ask for the drug
                          clarifications=[{"input": None, "status": "missing_drug", "question": question_text,  # nothing was typed
                                           "did_you_mean": []}])  # no suggestions
            fired.append(RULE_CLARIFICATION)  # note the clarification rule
            fired.append(RULE_NEGATION)  # and the negation rule
        else:  # no drug and no negation
            self._free_text_route(answer, question, self._symptom_candidates(tokens, taken), negation)  # symptoms, then embeddings

        return answer  # the finished answer

    # ---- routes -------------------------------------------------------
    def _pair_route(self, answer, mentions):  # answer a "drug A with drug B" question
        texts = [m["text"] for m in mentions]  # the drug names as typed
        out = self.checker.check(texts[:-1], texts[-1], include_current_pairs=len(texts) > 2)  # earlier names are current drugs, the last is new
        answer.update(status="answered", intent="pair", results=out["results"], inputs=out["inputs"],  # copy the checker's results
                      summary=out["summary"])  # and its summary
        answer["notes"].extend(out["notes"])  # add the checker's notes
        for rule in out["rules_fired"]:  # merge the rules without duplicates
            if rule not in answer["rules_fired"]:  # new rule
                answer["rules_fired"].append(rule)  # add it

    def _avoid_route(self, answer, mention, negation):  # answer "what should I avoid with X"
        out = self.checker.avoid(mention["resolution"].name)  # the graph's avoid list for the drug
        answer.update(status="answered", intent="avoid", avoid=out, negation_words=sorted(set(negation)))  # attach it with the negation words seen
        answer["rules_fired"].append(RULE_NEGATION)  # note the negation rule
        answer["rules_fired"].extend(r for r in out["rules_fired"] if r not in answer["rules_fired"])  # merge the checker's rules
        answer["notes"].extend(out["notes"])  # add its notes

    def _drug_route(self, answer, mention, symptom_words):  # answer a question about one drug
        out = self.checker.drug(mention["resolution"].name)  # everything documented for the drug
        interactions = out["interactions"]  # its interaction rows
        answer.update(status="answered", intent="drug", drug=out["drug"]["name"])  # start the answer
        answer["rules_fired"].append("exact_name_resolution")  # the drug was matched by exact name
        if symptom_words:  # the user also named a symptom
            filtered = self._match_symptoms(answer, interactions, symptom_words)  # keep rows whose text mentions it
            if filtered:  # some matched
                interactions = filtered  # show only those
            else:  # none matched
                answer["notes"].append("No interaction text for this drug mentions that symptom; "  # say so
                                       "all of its documented interactions are listed instead.")  # and list everything
        answer["interactions"] = sorted(interactions, key=lambda r: (severity_rank(r["severity"]), r["id"]))  # Major first, ties by ID
        if out["excluded_evidence"]:  # some evidence failed the eligibility filter
            answer["excluded_evidence"] = out["excluded_evidence"]  # report it
            answer["needs_review"] = True  # and flag for review
        answer["not_available"] = out["not_available"]  # list the drug-level facts our data does not hold

    def _free_text_route(self, answer, question, symptom_words, negation):  # last-resort route for free text
        records = self.checker.eligible_records(self.checker.graph.records)[0]  # all records that pass the eligibility filter
        if symptom_words:  # try symptom words first
            matched = self._match_symptoms(answer, records, symptom_words)  # text match, with optional HPO help
            if matched:  # found something
                answer.update(status="answered", intent="symptom", interactions=matched)  # answer with those rows
                return  # done
        if self.embeddings is not None:  # tier two is on
            candidates = self.embeddings.candidates(question, k=5)  # the five nearest records
            eligible, _ = self.checker.eligible_records([c for c in candidates if c is not None])  # still pass them through the filter
            if eligible:  # some survived
                answer.update(status="candidates_only", intent="free_text",  # not a documented answer
                              candidates=[_legacy_record(r) for r in eligible])  # records shown as candidates
                answer["rules_fired"].append(RULE_CANDIDATES_ONLY)  # note the rule
                answer["notes"].append(  # explain the limit
                    "These records look related to your wording. Text similarity finds candidates only; it "
                    "does not decide whether an interaction exists or how severe it is. Ask again with exact "
                    "drug names to get a documented answer.")
                return  # done
        answer.update(status="no_match", intent="none")  # nothing could place the question
        answer["notes"].append(  # tell the user how to ask
            "I could not identify a drug from the exact names in our data, so nothing was looked up. "
            "Ask with drug names, for example 'Do warfarin and ibuprofen interact?'. " + NOT_SAFE_NOTE)  # and never imply safety

    def _match_symptoms(self, answer, records, symptom_words):  # find records whose text matches symptom words
        """Records whose effect or mechanism text matches a symptom word, Major first."""
        mapping = self.symptom_mapping and self.hpo_conn is not None  # is HPO expansion in use?
        if not self.symptom_mapping:  # user turned it off
            answer["notes"].append("Symptom mapping is off, so only the literal words you typed were matched.")  # say so
        by_id, phenotypes = {}, {}  # matched records by ID, and HPO term records by ID
        for word in symptom_words:  # each symptom word
            expansion = hpo.expand_symptom(self.hpo_conn, word) if mapping else None  # widen it through HPO when allowed
            if mapping and not expansion["terms"] and self.hpo_index is not None:  # HPO knew nothing about the word
                # No label or synonym matched: let the HPO embedding index propose candidate terms.
                expansion = hpo.expand_symptom(self.hpo_conn, word, self.hpo_index.nearest_ids(word, k=3))  # try the three nearest terms
            for record, matched_on, phrase, helper in hpo.find_in_text(records, word, expansion):  # search the interaction text
                item = dict(record, matched_on=matched_on, matched_phrase=phrase)  # copy the record and say what matched
                if helper:  # the match came through an HPO phrase
                    item["hpo_term_helped"] = helper  # say which term helped
                by_id.setdefault(record["id"], item)  # keep the first match per record
            if expansion and by_id:  # something matched and HPO was used
                for term in expansion["terms"]:  # its phenotype records
                    phenotypes.setdefault(term["id"], term)  # keep each once
        if not by_id:  # nothing matched
            return []  # empty result
        answer["rules_fired"].append(RULE_SYMPTOM_MAPPING if mapping else "literal_symptom_match")  # note which matching was used
        answer["notes"].append(  # explain what the match rests on
            "Results matched on the effect text (or the mechanism text) of the interaction records"
            + ("; the HPO terms only helped find them." if mapping else "."))  # mention HPO only when it was used
        if mapping and phenotypes:  # HPO records exist
            answer["phenotype_records"] = list(phenotypes.values())  # shown separately from interactions
            answer["phenotype_note"] = hpo.PHENOTYPE_NOTE  # labelled as vocabulary, not evidence
        return sorted(by_id.values(), key=lambda r: (severity_rank(r["severity"]), r["id"]))  # Major first, ties by ID


def candidate_records(conn, query):  # records for a question, in the old shape
    """Records for a question in the old record shape, for the selector and the offline lookup.

    The same rules apply as in QuestionAnswerer; HPO is not mixed in. Most severe first.
    """
    from ddi_backend.checker import Checker  # imported here: checker pulls in more than this module needs
    checker = Checker(conn)  # the deterministic main flow
    answer = QuestionAnswerer(checker, hpo_conn=conn).ask(query)  # answer the question with the normal rules
    records, seen = [], set()  # collected records and the IDs already added

    def add(record):  # inner helper: add a record once
        if record["id"] not in seen:  # not added yet
            seen.add(record["id"])  # remember it
            records.append(record)  # keep it

    if answer.get("intent") == "avoid":  # an avoid-list answer
        for entry in answer["avoid"]["avoid"]:  # each drug to avoid
            for record_id in entry["record_ids"]:  # its record IDs
                add(next(r for r in checker.graph.records if r["id"] == record_id))  # look up the stored record
    elif answer.get("intent") == "pair":  # a pair answer
        for result in answer["results"]:  # each pair result
            for record in result.get("records", []):  # its records
                add(record)  # keep them
    else:  # a drug, symptom or other answer
        for record in answer.get("interactions", []):  # its interaction rows
            add(record)  # keep them
    records.sort(key=lambda r: (severity_rank(r["severity"]), r["id"]))  # Major first, ties by ID
    return [_legacy_record(r) for r in records]  # reduce to the old shape


# ---------------------------------------------------------------------------
# FILE SUMMARY: Answers free-text questions in a fixed order: exact drug-name resolution, then deterministic lookups
# (pair check, avoid list, one drug's interactions), then symptom words matched in effect and mechanism text, and only
# last embedding candidates. It handles ambiguity, negation, missing context and adversarial wording as explicit rules.
# ASSIGNMENT TOOL: Knowledge graph with rule-based reasoning (the question router over the graph and checker), with
# embeddings in a limited role as the last fallback and HPO as a symptom-word helper. No piece-splitting retrieval
# (RAG) is implemented; each interaction row is one whole record.
# ---------------------------------------------------------------------------
