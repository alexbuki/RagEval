import contextlib
import copy
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import official_grade
# Offline tests mock HTTP; permit this suite without requests installed.
try:
    import requests
except ModuleNotFoundError:
    import types
    sys.modules['requests'] = types.SimpleNamespace(Session=None, RequestException=RuntimeError)
import query
from common import parse_prediction,source_map,retrieved_indices
from evaluate import score_one


GOLD={'id':1,'question':'Which year?', 'websites':[
    {'index':1,'content':'2011'}, {'index':3,'content':'2011'},
    {'index':7,'content':'2013'}, {'index':8,'content':''}],
    'answers':[
        {'answer':'2011','index':[1,3],'answer_judge_keyword':['2011'],
         'reason':[{'reason_judge_keyword':['record A']}]},
        {'answer':'2013','index':[7],'answer_judge_keyword':['2013'],
         'reason':[{'reason_judge_keyword':['record B']}]}]}
PRED={'answers':[
    {'answer':'2011','index':[1,3],'reason':['record A']},
    {'answer':'2013','index':[7],'reason':['record B']}]}


def sample(pred=PRED):
    return {'id':1,'question':'Which year?','mode':'local',
            'response':{'response':json.dumps(pred),'llm_generated':True,
                        'references':[{'file_path':str(i),'content':['text']} for i in range(3)]}}


class Checks(unittest.TestCase):
    def test_perfect_matches_official(self):
        expected=official_grade.test({'answer':PRED,'info':[[1,3],[7]]},copy.deepcopy(GOLD))
        got=score_one(GOLD,sample(),source_map([GOLD]))
        for a,b in [('answer_clustering','NMI'),('answer_coverage','answerScore'),('reason_coverage','reasonScore')]:
            self.assertEqual(got[a],expected[b]);self.assertAlmostEqual(got[a],1)
        self.assertAlmostEqual(got['retrieved_subset_nmi'],1)

    def test_missing_does_not_shrink_gold(self):
        p=copy.deepcopy(PRED);p['answers'][0]['index']=[1]
        got=score_one(GOLD,sample(p),source_map([GOLD]))
        self.assertEqual(got['partition_status'],'Missing Elements')
        self.assertEqual(got['answer_clustering'],0)
        self.assertEqual(got['answer_coverage'],1)

    def test_duplicate(self):
        p=copy.deepcopy(PRED);p['answers'][1]['index']=[3,7]
        self.assertEqual(score_one(GOLD,sample(p),source_map([GOLD]))['partition_status'],'Duplicate Elements')

    def test_missing_viewpoint_normalization(self):
        p={'answers':[PRED['answers'][0]]}
        r=score_one(GOLD,sample(p),source_map([GOLD]))
        self.assertAlmostEqual(r['answer_coverage'],1/math.sqrt(2))
        self.assertAlmostEqual(r['reason_coverage'],1/math.sqrt(2))

    def test_repository_case_sensitive(self):
        self.assertEqual(official_grade.compare_answer(['YES'],[['yes']])[0],0)

    def test_canned_not_scored(self):
        s=sample();s['response']={'response':'No relevant context found for the query.','llm_generated':False}
        r=score_one(GOLD,s,source_map([GOLD]));self.assertEqual(r['status'],'unscorable')
        self.assertIsNone(r['answer_clustering'])

    def test_malformed(self):
        s=sample();s['response']['response']='Plain prose'
        self.assertEqual(score_one(GOLD,s,source_map([GOLD]))['status'],'unscorable')
        with self.assertRaises(ValueError):parse_prediction('{"answers":[{"answer":"a","index":["1"],"reason":[]}]}')

    def test_mapping(self):
        m=source_map([GOLD]);self.assertEqual([x['index'] for x in m.values()],[1,3,7])
        m['3']={'question_id':2,'index':1}
        found,unknown,other=retrieved_indices({'references':[{'file_path':'/inputs/1'},{'file_path':'3'},{'file_path':'nonsense'}]},GOLD,m)
        self.assertEqual(found,[3]);self.assertEqual(unknown,['nonsense']);self.assertEqual(other,['3'])

    def test_mock_query_pipeline(self):
        class Response:
            def __init__(self,data):self.data=data
            def raise_for_status(self):pass
            def json(self):return self.data
        class Session:
            def get(self,*a,**k):return Response({'pipeline_busy':False,'configuration':{}})
            def post(self,*a,**k):
                body=k['json'];assert body['include_chunk_content'];assert 'user_prompt' in body
                assert 'answer_judge_keyword' not in body['user_prompt']
                return Response(sample()['response'])
        with tempfile.TemporaryDirectory() as d:
            inp=Path(d)/'data.jsonl';out=Path(d)/'result.jsonl'
            inp.write_text(json.dumps(GOLD)+'\n')
            args=['query.py','--data',str(inp),'--output',str(out),'--num-records','1']
            with patch.object(sys,'argv',args),patch.object(query.requests,'Session',Session),contextlib.redirect_stdout(io.StringIO()):
                query.main()
            rows=[json.loads(s) for s in out.read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertTrue(all(r['status']=='ok' for r in rows))
            self.assertTrue(out.with_suffix('.meta.json').exists())
            self.assertEqual(rows[0]['retrieved_source_indices'],[1,3,7])
            with patch.object(sys,'argv',args),self.assertRaises(SystemExit):query.main()


if __name__=='__main__': unittest.main()
