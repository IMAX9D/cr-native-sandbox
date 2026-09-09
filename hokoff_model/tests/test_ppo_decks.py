from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from hokoff_model.ppo_decks import opponent_deck_config,episode_replay


class DeckTests(unittest.TestCase):
    def setUp(self):
        self.fixture=dict(rndSeed=1,cmd=[],battle={f'deck{s}':dict(sp=[dict(d=20+i,l=10) for i in range(8)],sc=[1]) for s in (0,1)})
        self.config=dict(mode='random',sampling='uniform_unique_8_max_one_champion_v1',card_ids=list(range(1,13)),champion_ids=[11,12])

    def test_randomizes_only_opponent_and_preserves_levels_and_fixture(self):
        original=deepcopy(self.fixture);seen=set()
        for side in (0,1):
            for i in range(100):
                replay=episode_replay(self.fixture,seed=42,episode_id=i,learner_side=side,config=self.config)
                self.assertEqual(replay['battle'][f'deck{side}'],original['battle'][f'deck{side}'])
                cards=replay['battle'][f'deck{1-side}']['sp'];ids=[c['d'] for c in cards]
                self.assertEqual(len(ids),len(set(ids)));self.assertLessEqual(len(set(ids)&{11,12}),1)
                self.assertTrue(all(c['l']==10 and set(c)=={'d','l'} for c in cards))
                self.assertEqual(replay['rndSeed'],42+i);seen.add(tuple(ids))
        self.assertGreater(len(seen),50);self.assertEqual(self.fixture,original)

    def test_replay_does_not_depend_on_call_order_or_touch_global_rng(self):
        import random
        rng=random.getstate()
        first={i:episode_replay(self.fixture,seed=8,episode_id=i,learner_side=i%2,config=self.config) for i in range(12)}
        for i in reversed(range(12)):
            self.assertEqual(first[i],episode_replay(self.fixture,seed=8,episode_id=i,learner_side=i%2,config=self.config))
        self.assertEqual(random.getstate(),rng)

    def test_frozen_pool_and_old_fixed_mode_survive_resume(self):
        self.assertEqual(opponent_deck_config(None,None,saved=self.config),self.config)
        fixed=opponent_deck_config(None,None,saved=dict(mode='fixed'))
        self.assertEqual(episode_replay(self.fixture,seed=9,episode_id=3,learner_side=1,config=fixed)['battle'],self.fixture['battle'])
        with self.assertRaisesRegex(ValueError,'resume'):opponent_deck_config('random',None,saved=dict(mode='fixed'))

    def test_pool_excludes_unknown_nonstandard_and_unsupported_skill_cards(self):
        rows={i:dict(card_id=i,standard_1v1=True,active_ability=None,rarity='Common') for i in range(1,14)}
        rows[11]['standard_1v1']=False;rows[12]['active_ability']='unsupported';rows[10]['rarity']='Champion'
        encoder=SimpleNamespace(card_id_to_token={i:i for i in range(1,13)},ability_id_to_token={})
        with patch('hokoff_model.ppo_decks.catalog',return_value=rows),patch('hokoff_model.ppo_decks.NativeObservationEncoder.from_manifest',return_value=encoder):
            config=opponent_deck_config(None,{})
        self.assertEqual(config['card_ids'],list(range(1,11)));self.assertEqual(config['champion_ids'],[10])

    def test_known_spell_with_unknown_spawned_forms_is_excluded(self):
        rows={i:dict(card_id=i,standard_1v1=True,active_ability=None,rarity='Common') for i in range(1,9)}
        rows[28000025]=dict(card_id=28000025,standard_1v1=True,active_ability=None,rarity='Legendary')
        encoder=SimpleNamespace(card_id_to_token={i:i for i in rows},ability_id_to_token={})
        with patch('hokoff_model.ppo_decks.catalog',return_value=rows),patch('hokoff_model.ppo_decks.NativeObservationEncoder.from_manifest',return_value=encoder):
            config=opponent_deck_config(None,{})
            self.assertNotIn(28000025,config['card_ids'])
            self.assertEqual(config['excluded_spawned_forms'],{'28000025':[26000104,26000105]})
            old=dict(config,card_ids=list(rows))
            with self.assertRaisesRegex(ValueError,'unsupported spawned'):opponent_deck_config(None,{},saved=old)

    def test_both_random_draws_are_independent_and_role_stable(self):
        original=deepcopy(self.fixture)
        for i in range(30):
            side=i%2
            both=episode_replay(self.fixture,seed=42,episode_id=i,learner_side=side,config=self.config,learner_config=self.config)
            opponent_only=episode_replay(self.fixture,seed=42,episode_id=i,learner_side=side,config=self.config)
            learner_only=episode_replay(self.fixture,seed=42,episode_id=i,learner_side=side,learner_config=self.config)
            self.assertEqual(both['battle'][f'deck{1-side}'],opponent_only['battle'][f'deck{1-side}'])
            self.assertEqual(both['battle'][f'deck{side}'],learner_only['battle'][f'deck{side}'])
            self.assertEqual(learner_only['battle'][f'deck{1-side}'],original['battle'][f'deck{1-side}'])
            self.assertNotEqual(both['battle']['deck0']['sp'],both['battle']['deck1']['sp'])
            swapped=episode_replay(self.fixture,seed=42,episode_id=i,learner_side=1-side,config=self.config,learner_config=self.config)
            self.assertEqual(both['battle'][f'deck{side}']['sp'],swapped['battle'][f'deck{1-side}']['sp'])
            for s in (0,1):
                cards=both['battle'][f'deck{s}']['sp'];ids=[c['d'] for c in cards]
                self.assertEqual(len(set(ids)),8);self.assertLessEqual(len(set(ids)&{11,12}),1)
                self.assertTrue(all(c['l']==10 and 'el' not in c for c in cards))
        self.assertEqual(self.fixture,original)

    def test_learner_mode_resume_does_not_silently_change_old_experiments(self):
        old=opponent_deck_config(None,None,saved=dict(mode='fixed'),role='learner')
        self.assertEqual(old['mode'],'fixed')
        with self.assertRaisesRegex(ValueError,'resume learner'):
            opponent_deck_config('random',None,saved=old,role='learner')
        self.assertEqual(opponent_deck_config(None,None,saved=self.config,role='learner'),self.config)


if __name__=='__main__':unittest.main()
