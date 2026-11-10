#"Structured LightRAG queries. Run from RagEval with the existing index loaded
# Notes: like the regular query.py file but asks ligthrag to organize evidence into answer clusteres required to do Confrag like evaluation
""" Example json structure
{
  "answers": [
    {
      "answer": "One viewpoint",
      "index": [1, 3],
      "reason": ["Supporting reason"]
    }
  ]
}

"""

import argparse
import hashlib
import json
from pathlib import Path
import requests
from common import load_pilot, source_map, parse_prediction, retrieved_indices


def instruction(row, mapping, cluster_count):
    gold_ids = sorted({i for a in row['answers'] for i in a['index']})
    lookup = {k: v['index'] for k, v in mapping.items()
              if v['question_id'] == row['id']}
    count_rule = (f'Target cluster count: {len(row["answers"])} (benchmark-provided count). '
                  'Try to produce that many distinct supported viewpoints, but never invent '
                  'a viewpoint to reach the count.' if cluster_count == 'gold' else
                  'Choose the number of distinct viewpoints supported by the supplied evidence.')
    return f'''Return only a valid JSON object with this structure:
{{"answers":[{{"answer":"One viewpoint", "index":[1,3], "reason":["Specific supporting reason"]}}]}}
Organize the retrieved evidence into distinct viewpoints answering the question.
{count_rule}
Each index is an ORIGINAL ConfRAG webpage index, NOT a citation/reference number,
chunk ID, file_path, or graph entity ID.
Lookup from retrieved reference file_path to original webpage index:
{json.dumps(lookup, sort_keys=True)}
Benchmark-annotated indices eligible for this task: {json.dumps(gold_ids)}.
Use only eligible webpages actually represented in the retrieved evidence.
Ignore retrieved webpages belonging to other questions or without benchmark annotation.
The lookup is metadata, not evidence: do not assign unseen sources merely because
an index appears in it. Do not infer missing source positions from model memory.
Group eligible retrieved webpages into a non-overlapping partition, with each used
webpage index appearing exactly once. Multiple chunks from one webpage share its index.
Each answer must summarize one distinct viewpoint. Keep supporting reasons attached
to that viewpoint, as a list of concise strings grounded in its retrieved sources.
Do not invent missing details or collapse incompatible viewpoints into a compromise.
If there is no usable evidence, return {{"answers":[]}}.
No markdown fences, commentary, or citations outside the JSON object.'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', default='http://localhost:9621')
    p.add_argument('--data', default='../data/ConfRAG.jsonl')
    p.add_argument('--output', default='../data/ConfRAG_structured_answers.jsonl')
    p.add_argument('--num-records', type=int, default=10)
    p.add_argument('--modes', nargs='+', choices=['local','naive'], default=['local','naive'])
    p.add_argument('--cluster-count', choices=['gold','unknown'], default='gold')
    p.add_argument('--timeout', type=int, default=1500)
    args = p.parse_args()
    rows = load_pilot(args.data, args.num_records)
    mapping = source_map(rows)
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f'{output} exists. Use --output with a new name; existing results are preserved.')
    output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    health = session.get(args.server+'/health', timeout=30)
    health.raise_for_status()
    health = health.json()
    if health.get('pipeline_busy') or health.get('pipeline_active'):
        raise SystemExit('Indexing is still active. Wait for completion before querying.')
    # Health idle alone does NOT certify all indexed documents succeeded.
    metadata = {'protocol':'confrag_metrics_on_shared_corpus_rag',
                'cluster_count':args.cluster_count, 'dataset_subset_sha256':fingerprint,
                'num_records':args.num_records, 'source_map':mapping,
                'configuration':health.get('configuration',{}),
                'core_version':health.get('core_version'),
                'note':'Confirm all intended source documents are successfully indexed.'}
    output.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with output.open('x', encoding='utf-8') as f:
        for row in rows:
            for mode in args.modes:
                print(f'Starting id={row["id"]} mode={mode}', flush=True)
                body = {
                    'query': row['question'],
                    'mode': mode,
                    'response_type': 'valid JSON only',
                    'include_references': True,
                    'include_chunk_content': True,
                    'enable_rerank': False,
                    'disable_user_prompt_prefix': True,
                    'user_prompt': instruction(row, mapping, args.cluster_count),
                }
                result = {'id':row['id'], 'question':row['question'], 'mode':mode,
                          'request':body, 'cluster_count_condition':args.cluster_count}
                try:
                    resp = session.post(args.server+'/query', json=body, timeout=(15,args.timeout))
                    resp.raise_for_status()
                    payload = resp.json()
                    result['response'] = payload
                    indices, unknown, other = retrieved_indices(payload,row,mapping)
                    result.update(retrieved_source_indices=indices,
                                  unknown_source_paths=unknown, off_question_source_paths=other)
                    if payload.get('llm_generated') is False:
                        result['status'] = 'no_generation'
                    else:
                        try:
                            result['prediction'] = parse_prediction(payload.get('response',''))
                            result['status'] = 'ok'
                        except (ValueError,TypeError) as e:
                            result.update(status='invalid_structure',error=str(e))
                except (requests.RequestException,ValueError) as e:
                    result.update(status='request_error',error=str(e))
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
                f.flush()
                print(f'Finished id={row["id"]} mode={mode}: {result["status"]}',flush=True)
    print(f'Saved {output}',flush=True)


if __name__ == '__main__':
    main()
