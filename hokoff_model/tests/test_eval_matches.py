import json
import unittest
from copy import deepcopy
from eval_hokoff_matches import ROOT, DEFAULT_CONTRACT, schedule, result_name, summarize
from hokoff_model.ppo_decks import opponent_deck_config


class EvalMatchTests(unittest.TestCase):
    def test_mirrored_unique_legal_decks_balanced_sides(self):
        contract=json.loads(DEFAULT_CONTRACT.read_text())
        pool=opponent_deck_config('random',contract['encoder'])
        fixture=json.loads((ROOT/'hokoff_model/fixtures/hog_2_6.json').read_text())
        before=deepcopy(fixture);plan=schedule(fixture,pool,100,123)
        self.assertEqual(plan,schedule(fixture,pool,100,123))
        self.assertEqual(fixture,before)
        self.assertEqual(sum(r['candidate_side']==0 for r in plan),50)
        decks=set()
        for row in plan:
            battle=row['replay']['battle']
            self.assertEqual(battle['deck0'],battle['deck1'])
            cards=[c['d'] for c in battle['deck0']['sp']]
            self.assertEqual(len(set(cards)),8)
            self.assertLessEqual(len(set(cards)&set(pool['champion_ids'])),1)
            self.assertTrue(set(cards)<=set(pool['card_ids']))
            decks.add(tuple(sorted(cards)))
        self.assertEqual(len(decks),100)

    def test_outcomes_and_denominators(self):
        term=dict(terminated=True,truncated=False,outcome='side0_win')
        self.assertEqual(result_name(term,0),'win')
        self.assertEqual(result_name(term,1),'loss')
        with self.assertRaises(ValueError):result_name(dict(term,truncated=True),0)
        with self.assertRaises(ValueError):result_name(dict(term,outcome='unknown'),0)
        result=summarize([dict(result='win',candidate_side=0),dict(result='draw',candidate_side=1)])
        self.assertEqual(result['win_rate'],.5);self.assertEqual(result['score_rate'],.75)


if __name__=='__main__':unittest.main()
