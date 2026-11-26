# Drug interaction backend (class prototype)

A backend that checks a new medication against the medications someone already takes and returns what our
data **documents**: severity (Major, Moderate or Minor), mechanism, clinical effect, safer alternatives with
the reason, and the source, in plain language. It also answers free-text questions such as "which drugs should
I avoid with warfarin?" and can hand a documented interaction to a **simulated** pharmacist-review queue after
the user approves it.

> **Class prototype. Not medical advice.** It uses only the public or synthetic data in this repository, never
> real patient data, and every response says so. Anything simulated is labelled simulated. There is no
> user-interface code here; a frontend will be built later against the HTTP API.

Two rules matter more than anything else:

1. **"Not found in our records" is never presented as "safe".** An unknown drug is `unknown_drug`, a pair with
   no row is `no_documented_interaction` with a note that this does not mean the combination is safe, and
   evidence that fails the eligibility filter is `evidence_ineligible`.
2. **Missing a Major interaction is the worst error.** Pairs are looked up with a direct query, repeated pairs
   headline their highest severity, and results are sorted Major first.

## Quick start (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m ddi_backend.load_data load          # builds medications.db from data\DDI_Database.json and data\hp.json
python -m unittest discover -s tests -t .     # runs the tests

python -m ddi_backend check --current "Warfarin 5 mg" --current Omeprazole --new Ibuprofen
python -m ddi_backend avoid warfarin
python -m ddi_backend ask "Which pain reliever should I not take with warfarin?"
python -m ddi_backend serve                   # HTTP API at http://127.0.0.1:8000/docs
python eval\run_eval.py                       # evaluation report in eval\report.md
```

`medications.db` is generated, so it is not in git. Rebuild it any time with the `load` command; the load is
one transaction and safe to repeat. The application opens it **read-only** and never changes it. Everything the
system writes (the review queue, the request log, cached embeddings) goes in a separate small file,
`agent_state.db`, beside it. Only the standard library is needed for the core; FastAPI is needed for the API,
and numpy and sentence-transformers only for the optional embedding tier.

## What is in the repository

| Path | What it is |
|---|---|
| `ddi_backend/load_data.py` | The existing loader. Schema, `ddi:<id>` / `hpo:<HP id>` IDs, transactional load and repeated pairs are unchanged. |
| `ddi_backend/ingest.py` | Read-only database access, the interaction record shape, and typing of safer alternatives. |
| `ddi_backend/resolver.py` | Exact, case-insensitive name resolution; dose stripping; did-you-mean suggestions; optional alias file. |
| `ddi_backend/graph.py`, `rules.py` | The knowledge graph and the named, tested rules. |
| `ddi_backend/registry.py`, `eligibility.py` | Source registry and the evidence eligibility filter. |
| `ddi_backend/checker.py` | The deterministic main flow: resolve, look up, apply rules. |
| `ddi_backend/retrieval.py`, `hpo.py`, `embeddings.py` | Free-text questions, symptom mapping, and embeddings in their limited role. |
| `ddi_backend/composer.py`, `llm.py`, `llm_chat.py` | Template composer, language model modes, selector chat. |
| `ddi_backend/agent.py`, `state.py` | The guarded agent and its state file. |
| `ddi_backend/cli.py`, `api.py`, `service.py` | Command line, HTTP API, and the shared setup. |
| `ddi_backend/models.py`, `scripts/export_graph.py` | The deferred neural model: interface, graph export, pair-level split. |
| `ddi_backend/evaluation.py`, `eval/run_eval.py` | The evaluation harness. |
| `data/` | `DDI_Database.json`, `hp.json`, `sources_seed.json`, `aliases.json` (empty). |
| `experiments/` | The class notebooks, archived as evidence. Nothing imports them. |
| `tests/` | The test suite. |

## Our data, and what it cannot do

The only data inputs are `DDI_Database.json`, `hp.json` and the `medications.db` built from them. The data holds
**interactions, not drug monographs**: uses, side effects, warnings, contraindications, ingredients, brand names
and food interactions are not in it, so they are answered "not available in our data" and the language model is
never allowed to supply them. No record carries a population scope, so a question about children, pregnancy and
the like gets a statement that our data does not cover it. Each answer cites its `ddi:` ID and the record's own
Reference field.

We have no alias or brand-name data, so a brand name such as Advil is `unknown_drug` with did-you-mean
suggestions; typos and abbreviations are never auto-resolved. `data/aliases.json` is empty and can be filled in.

### Source metadata

Sources use stable keys (`kaggle_ddi`, `hpo`), never the S1 to S5 labels from the experiments. The metadata
in `data/sources_seed.json` is the **team's designation, not a clinical claim**:

| Source | Status | Authority class | Scope | Effective date | Owner |
|---|---|---|---|---|---|
| `kaggle_ddi` | current | `course_designated_dataset` | exact pair | unknown (not guessed) | to be filled in |
| `hpo` | current | `vocabulary` | none | 2026-09-01 (from the version in `hp.json`) | to be filled in |

Approved authority classes are a list in the seed file and start with only `course_designated_dataset`.
"Approved" means the project designated the source as its interaction source; it does not mean clinically
validated. `vocabulary` is not approved as evidence, and a source with scope `none` is never counted as
interaction evidence. Code never fills in missing metadata.

## Decisions from our experiments

These follow section 4 of the brief; the evidence is in `experiments/`.

**Adopted**
- **Knowledge graph with rule-based reasoning**: the core engine and the source of every factual answer. Medications
  are nodes and interactions are edges; the rules are severity precedence, repeated-pair conflict detection and
  the alternative cross-check, and every response lists which rules fired.
- **Metadata-aware eligibility filtering**, with one safeguard added for the over-filtering we saw (the Kaggle
  source was excluded in four of five cases): when the only evidence for a pair is ineligible the status is
  `evidence_ineligible`, never "no interaction".
- **Guarded tool use with a simulated review queue**, ported from the agent notebook.
- **The language model**, as the reader of our database and the writer of the plain-language response, under the
  safeguards below.
- **The embedding test from the Colab work**, but only in the limited role below.

**Modified**
- **Embedding neighbours (Meeting 7) and semantic-search failures (Meeting 8).** Similarity may only find
  candidates; it never decides whether an interaction exists or how severe it is, and no score is shown as
  confidence. The four failure types each became a component: ambiguity (a clarification step), negation and
  adversarial questions (explicit rules that answer from the graph's avoid list), and context (population filter
  only where a record carries that scope, otherwise a statement that we do not cover it). The warfarin question
  ("Which pain reliever should I not take with warfarin?") now returns warfarin's avoid list by severity;
  similarity ranked acetaminophen first because it ignored "not". Using the real model, the nearest records for that
  question put Warfarin + Acetaminophen second even though the word "not" was in it.

**Deferred**
- **The neural network and graph neural network (Meetings 10 and 11).** Only an interface (`InteractionPredictor`),
  an offline graph export and a pair-level train/test split exist. Nothing is trained and nothing is on the user
  path. A future prediction must be labelled "predicted, not documented", and a prototype should start with a
  simple scikit-learn baseline.

**Rejected**
- **RAG** (splitting records into pieces, retrieving pieces, generating an answer from them): in our test only
  three of five cases retrieved relevant evidence, only one of five answers was grounded, only one of five citations
  lined up, and an absent pair got an invented answer. Here one interaction row is always one whole record, and a
  repository test checks that no retrieval code of that kind exists outside `experiments/`.
- **The Meeting 6 cut-off-score lab.** No cut-off scoring and no urgent-or-not classifier exist anywhere.

## How a question is answered

Order of work, always: exact name resolution first, then deterministic lookup (pairs, avoid lists, a drug's
interactions), and only then embeddings, and only for free text nothing else could place. Everything that comes back
passes the eligibility filter. HPO is used only to expand a symptom word (term labels, synonyms including
layperson synonyms, and child terms) into words to look for in the interaction **effect and mechanism text**; the
answer says it matched on that text, with the HPO term only helping to find it. Interaction and severity answers
read only the `drugs` and `interactions` tables, and `--no-symptom-mapping` turns symptom mapping off.

Safer alternatives are stored with a type, `alternative_drug` or `management_action` ("Reduce warfarin dose by
30-50%" is an action, not a drug), and are shown as written with their source. The system never generates or
calculates dose advice; doses you enter are echoed and never evaluated. Each suggested alternative is
cross-checked against both drugs of the pair and everything you take; if it has its own documented interaction it
is shown with a warning and flagged for review (for example ddi:1 suggests acetaminophen, but warfarin with
acetaminophen is itself documented as ddi:69).

### Statuses

Every pair result is one of `documented_interaction`, `no_documented_interaction`, `evidence_ineligible`,
`ambiguous_name` or `unknown_drug`. A repeated pair (our data lists 8 pairs twice with different severities, for
example clopidogrel with omeprazole as Major and Moderate) returns every row, headlines the highest severity, sets
`severity_conflict` and routes the pair to review.

## The three tiers

| Tier | Needs | What it adds |
|---|---|---|
| 1 | `medications.db` | The deterministic core, the guarded agent, the CLI and the API. |
| 2 | numpy, sentence-transformers | Embeddings (`all-MiniLM-L6-v2`, pinned revision, cosine similarity) for candidates, did-you-mean names and symptom-word candidates. One record is one document; HPO has its own index; vectors are computed once and cached beside `agent_state.db`. |
| 3 | `ANTHROPIC_API_KEY` | Claude explanations and record selection. `ANTHROPIC_MODEL` defaults to `claude-sonnet-4-5`; no key is in code. |

Each tier has its own tests, and the absence or failure of tier two or three never breaks tier one. If the embedding
model cannot be downloaded or loaded, retrieval switches itself off and the system falls back to exact names plus
`difflib` suggestions. Embeddings are off by default on the command line (`--embeddings auto` or `DDI_EMBEDDINGS=auto`
turns them on).

## The language model

Two modes, behind a small Anthropic Messages API provider using `urllib`.

- **Selector** (`llm_chat.py`, unchanged): the model sees only the question and the candidate records and returns
  `{"record_ids": [...]}`. The program checks the key set, the types and that every ID is a candidate, then prints
  the stored rows only. Invalid JSON or an unknown ID shows nothing, and no candidates means no API call.
- **Explain** (`llm.py`): writes a plain-language response from the eligible records for the requested pair. The
  model gets no browsing and no tools, sees only the supplied records, and must return structured output in which
  every claim points to a record ID. A validator compares severity, drug names, mechanism and effect with the record
  and rejects any new fact (a severity, drug name, number or content word that is not in the records); a failed check
  falls back to template text. Absent evidence produces "not documented in our data", and missing patient context
  produces a clarifying question, never an assumption. The generated text is always shown next to its source record.

With no key the system is deterministic-only, and that mode is complete: the template composer builds the response from
the severity, mechanism, effect, rationale, safer alternative and reference fields. `context_level` (`names_only` or
`full_context`) and `detail_level` (`brief`, `standard`, `detailed`) are settings, for the CLI (`--context`, `--detail`)
and the API (`options`).

## The guarded agent and review queue

The only write is `stage_interaction_review`, which stages a documented interaction into the simulated queue
`interaction_review_queue`, and only the user can authorise it. The controller checks, in order: `missing_evidence`,
`tool_disabled`, `tool_not_allowed`, `out_of_scope`, `approval_required`, `duplicate_request_id`; only then is the tool
called, and the tool repeats its own checks. Success is `staged_in_simulation`; a denial writes nothing; an error reports
the failure without claiming success, and a retry needs the error fixed and approval given again. Each queue note holds
`evidence_id`, `drug_pair`, `request_id` and the purpose "clinician review only". The notebook's cases one to five (and
5a/5b) are tests, and `python -m ddi_backend trace <request_id>` prints the attempted call, the gate result, the tool
result, the state change and whether the action actually happened.

The orchestrator is deterministic and imports no language model code. Only an interaction that passes the eligibility
filter counts as "documented", so an `evidence_ineligible` pair cannot be staged (it is reported as `missing_evidence`).

**About the codes.** The dossier refers to failure codes A1 through A6 but never defines them, so this project does
**not** assign A-numbers: the six gate names above are used as the codes until the team defines them. The R1, R2, R3,
G1, C1 and E1 codes are defined in the notebooks and are used in the evaluation report.

## HTTP API

`python -m ddi_backend serve`, then open `/docs`. There is no user-interface code. Set `DDI_CORS_ORIGINS` when a
frontend is served from another origin.

| Route | What it does |
|---|---|
| `POST /check` | Body: `current` (names or `{name, dose, details}`), `new`, `options` (`include_current_pairs`, `detail_level`, `context_level`, `explain`). One result per pair with status, severity, mechanism, effect, alternatives, rules fired and source. |
| `GET /drugs/{name}` | The drug's name and the interactions it appears in. Drug-level fields are "not available in our data". |
| `GET /avoid/{drug}` | Every drug with a Major or Moderate edge, ordered by severity, with a note that classes are not in our data. |
| `POST /ask` | A free-text question. |
| `POST /review-queue/requests` | `drug_pair`, `request_id` and `approved` are all required. 201 when staged; 403, 404 or 409 when a gate stops it, and nothing is written. |
| `GET /review-queue` | The simulated queue. |
| `GET /trace/{request_id}` | The trace for one request. |
| `GET /health` | Which tiers are available. |

## Evaluation

`python eval\run_eval.py` runs the same pairs (every documented pair plus a seeded sample of pairs with no row) through
the backend and writes `eval/report.md` with missed known interactions, false interactions, wrong severity, unsupported
claims, **Major recall on its own**, a severity confusion matrix, the handout and Meeting 8 questions, and how often the
filter wrongly excludes an eligible record. With `--llm claude` it also runs the four language-model conditions on the
same pairs, changing one variable at a time: names-only vs full-context (synthetic doses and other medications) and brief
vs detailed. Model claims are labelled supported, plausible but unsupported, or contradicted, with R1/G1 codes.

What has been run: the deterministic backend (0 of 172 known interactions missed, 0 of 60 false interactions, 0 wrong
severities, Major recall 100% of 63, filter wrong-exclusion rate 0 of 180) and the harness against a scripted stand-in
model, which is **simulated** and only shows that the harness catches misses, false interactions and wrong severities.
**No real language-model numbers are reported yet**: they need your `ANTHROPIC_API_KEY`.

## Things to know

- The brief says `DDI_Database.json`; the file that arrived was `DDI Database.json`. It now lives at
  `data\DDI_Database.json` (the loader still accepts either name).
- Cabergoline and rucaparib are not in the data (210 drugs, 180 interactions), so that pair is `unknown_drug`, not
  `no_documented_interaction`. The tests also use two drugs that both exist with no row (warfarin and docusate).
- The `ddi:` IDs depend on load order, so fixtures were confirmed against the real database first (ddi:1, 12, 20, 26
  and 73 match the handout).
- The retrieval notebook's lookup, the old keyword search, mixed HPO terms into interaction results and let words like
  "not" and "pain" use up the result slots before the drug name contributed (the warfarin question never reached
  warfarin). It was replaced; `search_hpo_records` returns phenotype records separately.
- An alternative such as "Acetaminophen (Paracetamol)" lists both names; the synonym is reported as not in our data.
- The state file has no authentication and the API has no users; it is a class prototype.

## Tests

`python -m unittest discover -s tests -t .` runs everything. The embedding tests use a fake embedder, and one
optional test uses the real model when it is installed and downloadable. The original four test files were repaired by
making the code a real package (so imports and `mock.patch` targets name the same modules); the regression tests the
brief lists are all kept: loader transactions and rollback, reload idempotence, foreign-key checks, repeated pairs
reported, a generic prompt returning nothing, a positive limit, a partial database rejected, and formatted output that
uses only present values. One of them changed on purpose: the old test expected an HPO term back from the interaction
search, which is the mixing we removed; it now checks the phenotype search separately.
