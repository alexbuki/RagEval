
import json
import copy
from pathlib import Path
from official_grade import test

data_path = Path(__file__).resolve().parent.parent / "data" / "ConfRAG.jsonl"

with open(data_path, encoding="utf-8") as f:
    row = json.loads(f.readline())

gold_groups = [a["index"] for a in row["answers"]]

def evaluate(answers):
    prediction = {"answers": answers}
    result = test(
        {"answer": prediction, "info": gold_groups},
        copy.deepcopy(row)
    )
    return (
        round(result["NMI"], 3),
        round(result["answerScore"], 3),
        round(result["reasonScore"], 3)
    )

perfect = [
    {
        "answer": a["answer"],
        "index": a["index"],
        "reason": [r["explain"] for r in a["reason"]]
    }
    for a in row["answers"]
]

no_reasons = copy.deepcopy(perfect)
for a in no_reasons:
    a["reason"] = []

print("Scenario                         Clustering   Answer   Reason")
print("-" * 65)
print("Perfect answer:                ", evaluate(perfect))
print("Missing one viewpoint:         ", evaluate(perfect[:2]))
print("Missing two viewpoints:        ", evaluate(perfect[:1]))
print("Correct answers, no reasons:   ", evaluate(no_reasons))
