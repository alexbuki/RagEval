# For ConfRAG eval metrics: answer coverage, answer clustering, reason coverage
``` bash
python3 confrag_eval/query.py \
  --data data/ConfRAG.jsonl \
  --num-records 50 \
  --modes naive local \
  --cluster-count unknown \
  --output data/structured_answers_50.jsonl
```
--data: ConfRAG dataset used for the evaluation.
--num-records 50: Evaluates the first 50 questions in the dataset.
--modes naive local: Runs both vector retrieval (naive) and graph-based local retrieval (local).
--cluster-count unknown: Requires the model to infer the number of distinct viewpoints from retrieved evidence without revealing the gold cluster count.
--output: Saves the structured predictions used by the evaluation script.


