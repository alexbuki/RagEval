"""Score structured predictions using the uploaded ConfRAG repository's functions."""
import argparse
import copy
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from common import load_pilot, source_map, parse_prediction, retrieved_indices
from official_grade import test as official_test


def score_one(row, sample, mapping):
    result = {'id':sample['id'], 'mode':sample['mode']}
    payload = sample.get('response', {})
    if not isinstance(payload, dict):
        payload = {'response':payload}
    indices, unknown, other = retrieved_indices(payload,row,mapping)
    groups = [a['index'] for a in row['answers']]
    gold_ids = {i for group in groups for i in group}
    observed_groups = [[i for i in group if i in indices] for group in groups]
    observed_groups = [group for group in observed_groups if group]
    result.update(retrieved_source_indices=indices,unknown_source_paths=unknown,
                  off_question_source_paths=other,
                  source_cluster_coverage=len(observed_groups)/len(groups) if groups and not unknown else None,
                  answer_clustering=None,answer_coverage=None,reason_coverage=None,
                  retrieved_subset_nmi=None)
    try:
        if sample.get('status') == 'request_error':
            raise ValueError('request_error: '+sample.get('error',''))
        if payload.get('llm_generated') is False:
            raise ValueError('no_generation: server returned a canned response')
        pred = parse_prediction(json.dumps(sample['prediction']) if 'prediction' in sample
                                else payload.get('response',''))
        if not pred['answers']:
            raise ValueError('empty_prediction')
        # Supply info explicitly. Never derive the scoring universe from predictions.
        scores = official_test({'answer':pred,'info':groups}, copy.deepcopy(row))
        result.update(status='scored',partition_status=scores['badPartition'],
                      answer_clustering=scores['NMI'],answer_coverage=scores['answerScore'],
                      reason_coverage=scores['reasonScore'],matching=scores['match'])
        predicted_ids = {i for a in pred['answers'] for i in a['index']}
        result['unretrieved_predicted_indices'] = sorted(predicted_ids-set(indices))
        result['non_gold_predicted_indices'] = sorted(predicted_ids-gold_ids)
        # Diagnostic only: conditional on observed, annotated sources. Exclude trivial
        # all-singleton/one-group cases; do not label this the official benchmark NMI.
        if not unknown and len(observed_groups)>1 and sum(map(len,observed_groups))>len(observed_groups):
            if predicted_ids <= set(indices) & gold_ids:
                conditional = official_test({'answer':pred,'info':observed_groups},copy.deepcopy(row))
                result['retrieved_subset_nmi'] = conditional['NMI']
                result['retrieved_subset_partition_status'] = conditional['badPartition']
    except (ValueError,TypeError,KeyError,ZeroDivisionError) as e:
        result.update(status='unscorable',error=str(e))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='./data/ConfRAG.jsonl')
    p.add_argument('--answers', default='./data/ConfRAG_structured_answers.jsonl')
    p.add_argument('--num-records', type=int, default=10)
    p.add_argument('--out-dir', default='./data/confrag_scores')
    args=p.parse_args()
    rows=load_pilot(args.data,args.num_records)
    by_id={r['id']:r for r in rows};mapping=source_map(rows)
    samples=[];seen=set()
    with open(args.answers,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            s=json.loads(line);key=(s['id'],s['mode'])
            if key in seen:raise SystemExit(f'Duplicate prediction {key}; select one run.')
            if s['id'] not in by_id:raise SystemExit(f'Unknown question {s["id"]}')
            if s.get('question') != by_id[s['id']]['question']:raise SystemExit('Question/id mismatch')
            seen.add(key);samples.append(s)
    results=[score_one(by_id[s['id']],s,mapping) for s in samples]
    grouped=defaultdict(list)
    for r in results:grouped[r['mode']].append(r)
    summary={'protocol':'Repository scoring functions; shared-corpus RAG adaptation, not exact benchmark reproduction.',
             'matching':'Case-sensitive, as implemented in uploaded grade.py',
             'coverage_reference':'All gold answer/reason clusters of each question',
             'nmi_reference':'Full gold source partition; missing sources invalidate the partition',
             'modes':{}}
    for mode,rs in grouped.items():
        valid=[r for r in rs if r['status']=='scored']
        detail={'expected_questions':len(rows),'received_outputs':len(rs),
                'missing_question_ids':sorted(set(by_id)-{r['id'] for r in rs}),
                'scorable_outputs':len(valid),'unscorable_outputs':len(rs)-len(valid),
                'valid_partition_rate_over_received':sum(r.get('partition_status')=='Normal' for r in rs)/len(rs),
                'metrics':{}}
        for metric in ['answer_clustering','answer_coverage','reason_coverage','retrieved_subset_nmi']:
            vals=[r[metric] for r in rs if r[metric] is not None]
            item={'n':len(vals),'mean':statistics.mean(vals) if vals else None,
                  'population_std':statistics.pstdev(vals) if vals else None}
            # Explicit penalty view prevents failed requests from disappearing from averages.
            if metric!='retrieved_subset_nmi':
                item['failure_penalized_mean_over_received']=sum(vals)/len(rs)
                normal=[r[metric] for r in valid if r.get('partition_status')=='Normal']
                item['valid_partition_only_mean']=statistics.mean(normal) if normal else None
                item['valid_partition_only_n']=len(normal)
            detail['metrics'][metric]=item
        summary['modes'][mode]=detail
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'per_question.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results),encoding='utf-8')
    with (out/'scores.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['id','mode','status','partition_status','answer_clustering','answer_coverage','reason_coverage','retrieved_subset_nmi','source_cluster_coverage','error']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
