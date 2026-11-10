# ConfRAG metrics for the existing LightRAG pilot



| File | Role |
|---|---|
| `query.py` | Queries LightRAG in `local` and `naive` modes and requests structured answers containing viewpoints, source indices, and reasons. |
| `evaluate.py` | Compares generated answers with ConfRAG annotations and calculates answer clustering, answer coverage, and reason coverage. |
| `official_grade.py` | Contains the original ConfRAG scoring functions used by `evaluate.py`. |
| `common.py` | Shared utilities for loading ConfRAG data, mapping LightRAG sources to ConfRAG indices, and parsing structured model output. |
| `test_eval.py` | Runs small tests to verify that parsing, source mapping, and evaluation metrics behave correctly. |
| `provenance.json` | Records where the evaluation code came from and any changes made relative to the official ConfRAG implementation. |
| `attached_run_audit` | Stores audit information from the attached or previous run, such as configurations, inputs, outputs, or validation details. Its exact contents depend on whether it is a file or directory. |

## Run Offline Tests
```bash
cd ~/RagEval/confrag_eval
python3 test_eval.py
```
All tests should pass before running the experiment.

## Run a one-question query test
```bash
rm -f ../data/structured_test.jsonl ../data/structured_test.meta.json

python3 query.py \
  --num-records 1 \
  --modes naive \
  --output ../data/structured_test.jsonl
```
Inspect the generated record:

```bash
python3 - <<'PY'
import json

with open("../data/structured_test.jsonl") as file:
    result = json.loads(file.readline())

print("STATUS:", result["status"])
print("ERROR:", result.get("error"))
print("MODEL OUTPUT:", result["response"]["response"])
print("PARSED:", result.get("prediction"))
PY
```

The test is successful when:

- `STATUS` is `ok`.
- `ERROR` is empty or absent.
- `prediction` contains an `answers` list.
- Each answer contains `answer`, `index`, and `reason`.

The expected prediction structure is:

```json
{
  "answers": [
    {
      "answer": "A distinct viewpoint",
      "index": [1, 3],
      "reason": [
        "A reason supporting this viewpoint"
      ]
    }
  ]
}
```

Current Issue I am having is when I run query it says that for all sources, for local and for naive I get invalid structure. 

