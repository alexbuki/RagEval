# ConfRAG metrics for the existing LightRAG pilot

## Run

Keep all four Python files together in `~/RagEval` (back up your old query.py).
Requires Python 3.10+ and `requests`; scoring uses only the standard library.
No index rebuild or server_load.py run is required. Wait until the intended source
files are successfully processed, not merely until pipeline_busy is false.

```bash
python3 query.py
python3 evaluate.py
```

Generation defaults: first 10 physical dataset lines, `local` and `naive`, existing
LightRAG server at http://localhost:9621. It writes
`data/ConfRAG_structured_answers.jsonl` and its `.meta.json` run manifest.
It refuses to overwrite an existing predictions file: use `--output` for another run.
To evaluate that alternative, pass its path with `evaluate.py --answers PATH`.
Logs print before and after each request. Transport failures and malformed JSON are
saved rather than aborting the other questions. There is no model-based repair or
silent retry, which could alter measured behavior. Restarting generation requires a
new output path; it does not resume or mix runs.

Scores go to `data/confrag_scores/summary.json`, `scores.csv`, and `per_question.jsonl`.
For a different output directory, pass `--out-dir PATH`.
The scripts use the server's configured model and retrieval budgets. They record
health configuration and request parameters, disable reranking, and include chunk
content in the same answer response. For controlled repeated experiments, also
freeze model digests, decoding settings, actual prompt/token budgets and the index.
Metadata contains server configuration; review before publicly sharing it.

## What was found in the attachments

- All 20 uploaded predictions are canned “No relevant context found” responses,
  with `llm_generated=false`, not substantive generated answers. They cannot receive
  valid structured ConfRAG scores. See `attached_run_audit/`.
- The first 10 records are valid and yield exactly 97 nonempty webpages under the
  uploaded loader's counter. They have IDs 1,2,3,4,5,7,8,9,11,12.
- Some later physical lines of the uploaded dataset are not independently valid JSON.
  The scripts strictly read only the requested first N lines; they stop with a clear
  error rather than silently skipping corrupted records if you increase N.
- Several uploaded webpages have no gold cluster annotation. The structured prompt
  identifies eligible annotated indices but does NOT reveal their grouping, answer
  text, reasons, or matching keywords.

## Source mapping (critical)

server_load.py assigned global numeric file_source strings starting at 0. Those
numbers are NOT ConfRAG's per-question website indices and NOT LightRAG citation IDs.
common.py replays exactly the first N dataset records, incrementing only for nonempty
content. The query prompt includes a file_path-to-original-index lookup for the
current question; predictions must use original webpage indices.

This assumes the indexed corpus is the same first N records, in the same order,
with the same loader. Do not change N if the index was built differently. The map
and dataset subset SHA-256 are saved in the manifest. Failed/duplicate insertions
and deduplication can affect returned paths; unresolved paths are recorded, not
silently assigned. Cross-question sources in the shared corpus are also recorded
and excluded from the target question's annotated-source evaluation. Gold labels
are NEVER embedded into source content or indexed.

## Official scoring versus experimental protocol

`official_grade.py` is copied from `analysisResult/grade.py` in the uploaded ZIP.
Only `from config import CACHE_SIZE` was replaced by the original value 1024, to
avoid importing upstream API credentials. All scoring functions are unchanged.
Attribution: https://github.com/XaiverYuan/ConfRAG ; repository README states CC BY 4.0.
The original scorer SHA-256 is recorded in provenance.json.

The official repository:
- Samples at most 5 webpages in generateIndex.py, with eligibility constraints.
- Gives the target number of clusters to the model.
- Calls `test({'answer': prediction, 'info': gold_partition}, gold_record)`.
- Uses case-SENSITIVE substring keyword matching (the paper says case-insensitive).
- Uses one-to-one maximum matching: answerScore = matches/sqrt(Npred*Ngold).
- Matches reasons within matched answer pairs, normalizes each by the square root
  of predicted/gold reason counts, then divides their sum by sqrt(Npred*Ngold).
- Assigns NMI=0 for missing/extra/duplicate source indices. It returns 0 for the
  degenerate identical single-cluster case, unlike some standard NMI packages.
- Has an alternate input branch that filters its NMI reference using predicted
  indices. We deliberately use the explicit `info` branch to avoid hiding omissions.

Our DEFAULT protocol is a shared-corpus RAG ADAPTATION using those scoring functions.
It does not reproduce the paper's five-webpage input selection. It uses all gold
clusters as the coverage reference and all gold source indices as the strict NMI
reference. Missing retrieval can therefore make strict NMI zero. This is an end-to-end
penalty, not proof the generator clustered available evidence incorrectly.
`retrieved_subset_nmi` is a separate diagnostic using the gold partition restricted
to annotated sources in returned references. It is only reported for nontrivial
subsets with at least two clusters and at least one multi-source cluster, and no
unknown source paths or unobserved predicted indices. Never call this the original
full-context benchmark NMI. Source-level cluster coverage is only a proxy: a chunk
from a gold source does NOT establish that its contradictory claim survived retrieval.
Graph summaries can also contain information beyond returned chunk snippets.

The default `--cluster-count gold` supplies the original question's gold cluster
count, explicitly a benchmark hint. It tells the model not to invent viewpoints if
retrieval lacks support. Use `--cluster-count unknown` for a natural RAG condition;
report those results separately. No gold answer wording/reasons/group memberships
are supplied. Both settings share the same output schema and metric functions.
Do not compare these scores directly to the paper's headline table without matching
its input selection, count hint, prompts, eligibility, and aggregation protocol.

For an exact benchmark reproduction, use the repo's generate.py + generateIndex.py
selection, the selected full webpage texts, and its saved `info` partition. That is a
separate generator benchmark, not the current graph/vector retrieval comparison.

## Reporting failures

summary.json separates scorable outputs, valid partitions, and failures. It reports
means over scorable outputs, valid-partition-only means, and an explicitly labeled
failure-penalized mean over received outputs. Missing rows are listed and not silently
counted as successes. All-failure input yields null valid means (not invented model
scores). Malformed/empty predictions are failures; malformed partitions retain the
repository's NMI=0 behavior. Source-grounding flags are separate from lexical scores.
Keyword matches can reward negations or miss paraphrases: audit examples manually.
Do not let an evaluator invent answers or clusters from gold annotations.

## Attached-output audit

```bash
python3 evaluate.py --answers data/ConfRAG_answers.jsonl --out-dir data/old_run_audit
```

Your uploaded old run produces 0/20 scorable responses. This says nothing about the
quality of your now-completed index. Run the updated query.py against that index.

## Validation

`python3 test_eval.py` exercises official-score equivalence, source mapping, invalid
partitions, case-sensitive matching, absent generation and mocked HTTP query responses.
No live Hive server was accessible from the development environment, so live inference
still needs to be run on your allocated node.
