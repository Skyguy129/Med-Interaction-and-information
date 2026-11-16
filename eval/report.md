# Evaluation report

Class prototype, not medical advice. The dataset is the ground truth. Doses and other medications in the full-context condition are synthetic.

Pairs: 172 documented, 60 with no documented row (seed 0). Every condition ran the same pairs.

## Backend (deterministic)

| Measure | Value |
|---|---|
| Pairs scored | 232 (errors 0, unparseable 0) |
| Missed known interactions | 0 of 172 |
| False interactions | 0 of 60 |
| Wrong severity | 0 |
| Unsupported claims (plausible but unsupported / contradicted) | 0 (0 / 0) |
| Major recall, exact severity | 100.0% of 63 |
| Major recall, any interaction | 100.0% |
| Failure codes | none |

Severity confusion matrix (rows: dataset, columns: system):

| | Major | Moderate | Minor | None | Unparseable |
|---|---|---|---|---|---|
| Major | 63 | 0 | 0 | 0 | 0 |
| Moderate | 0 | 48 | 0 | 0 | 0 |
| Minor | 0 | 0 | 61 | 0 | 0 |
| None | 0 | 0 | 0 | 60 | 0 |

## Language model

Not run: no language model was configured. Set OPENAI_API_KEY and run `python eval/run_eval.py --llm openai`.

## Handout and Meeting 8 questions

| Case | Result |
|---|---|
| RAG 1 direct question | pass |
| RAG 2 paraphrase | pass |
| RAG 3 severity, effect and mechanism | pass |
| RAG 4 distractor: pain reliever and warfarin | pass |
| RAG 5 absent pair (drugs not in the data) | pass |
| RAG 5 variant: known drugs, no documented row | pass |
| Meeting 7 reworded query, same record | pass |
| Meeting 8 ambiguity: MS | pass |
| Meeting 8 negation | pass |
| Meeting 8 context: children | pass |

## Eligibility filter

On the dataset (180 records) the filter wrongly excluded 0 eligible records (0.0%).

On the labelled suite of made-up sources (11 cases): 0 eligible records wrongly excluded, 0 ineligible records wrongly included.
