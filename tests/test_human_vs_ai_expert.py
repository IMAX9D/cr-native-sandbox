import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from native_core.human_vs_ai import (
    EXPERT_WEIGHTS_KIND, HumanVsAiGui, _ability_inputs, _deck_token_id,
    _load_policy, _native_action, _policy_label,
    _expert_card_or_position_choice,
    _expert_play_probability,
)


class ExpertLivePolicyTests(unittest.TestCase):
    def test_user_selected_heavy_control_form_allocation(self):
        from native_core.card_catalog import validate_deck
        replay=json.loads((Path(__file__).parents[1]/'examples/user-selected-heavy-control.json').read_text())
        expected={19,21,52,93,97,117,154,180}
        manifest=json.loads(Path(r'D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3-current-frontier-v5\compiled\native-bc-v1\manifest.json').read_text())
        vocabulary={value:index for index,value in enumerate(manifest['card_vocabulary'])}
        forms={0:'base',1:'evolution',2:'hero'}
        for side in range(2):
            raw=replay['battle'][f'deck{side}']['sp']
            checked=validate_deck([{'card_id':c['d'],'level':c['l']+1,'form':forms[c.get('el',0)]} for c in raw])
            tokens={vocabulary[next(item for item in manifest['card_vocabulary'] if item.endswith('@'+str(_deck_token_id(c))))] for c in checked}
            self.assertEqual(tokens,expected)

    def test_top_training_deck_fixture_matches_audited_token_set(self):
        from native_core.card_catalog import validate_deck
        path=Path(__file__).parents[1]/'examples/top-training-deck-control.json'
        replay=json.loads(path.read_text())
        expected={18,21,63,95,116,157,164,171}
        manifest=json.loads((Path(r'D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3-current-frontier-v5\compiled\native-bc-v1\manifest.json')).read_text())
        vocabulary={value:index for index,value in enumerate(manifest['card_vocabulary'])}
        for side in range(2):
            raw=replay['battle'][f'deck{side}']['sp']
            forms={0:'base',1:'evolution',2:'hero'}
            checked=validate_deck([{'card_id':c['d'],'level':c['l']+1,'form':forms[c.get('el',0)]} for c in raw])
            tokens={vocabulary[next(item for item in manifest['card_vocabulary'] if item.endswith('@'+str(_deck_token_id(c))))] for c in checked}
            self.assertEqual(tokens,expected)

    def test_rate_multiplier_preserves_default_and_scales_hazard(self):
        rates = torch.tensor([.001,.1,1.,20.],dtype=torch.float64)
        original = -torch.expm1(-rates*.05)
        torch.testing.assert_close(_expert_play_probability(rates,.05),original,rtol=0,atol=0)
        increased = _expert_play_probability(rates,.05,1.5)
        torch.testing.assert_close(increased,1-(1-original).pow(1.5))
        self.assertTrue(bool((increased>original).all()))
        self.assertTrue(bool((increased<1).all()))
        for invalid in [0,-1,float('nan'),float('inf'),11]:
            with self.assertRaises(ValueError): _expert_play_probability(rates,.05,invalid)

    def test_softcap_version_uses_expert_action_path(self):
        gui = HumanVsAiGui.__new__(HumanVsAiGui)
        gui.policy_version = 'expert-v1.2-softcap'
        gui.state = {}
        gui.env = SimpleNamespace(decks={})
        gui.mask_cache = object()
        gui.native_masks = {}
        gui._prepare_ai_masks = lambda: None
        gui._sample_expert = lambda **kwargs: (0, 0, {'expert_path': True})
        with patch('native_core.human_vs_ai.build_action_masks',
                   return_value=(np.zeros(5,dtype=bool),np.zeros((4,576),dtype=bool),[0,1,2,3])):
            action, meta = gui._sample_ai()
        self.assertIsNone(action)
        self.assertTrue(meta['expert_path'])
        self.assertIn('v1.2',_policy_label({'policy_version':gui.policy_version,'iteration':2,'training_step':154674}))

    def test_greedy_placement_keeps_sampling_rng_advance(self):
        probabilities = torch.tensor([.1, .2, .7, 0.])
        sample_rng = torch.Generator().manual_seed(123)
        greedy_rng = torch.Generator().manual_seed(123)
        reference_rng = torch.Generator().manual_seed(123)
        result = _expert_card_or_position_choice(probabilities, sample_rng)
        expected = int(torch.multinomial(probabilities, 1, generator=reference_rng).item())
        self.assertEqual(result, expected)
        self.assertEqual(_expert_card_or_position_choice(probabilities, greedy_rng, 'greedy-placement'), 2)
        self.assertTrue(torch.equal(sample_rng.get_state(), greedy_rng.get_state()))
        with self.assertRaises(ValueError):
            _expert_card_or_position_choice(probabilities, sample_rng, 'invalid')

    def test_queen_hog_preset_is_eight_unique_standard_cards(self):
        from native_core.card_catalog import validate_deck
        path=Path(__file__).parents[1]/'examples/queen-hog-control.json'
        replay=json.loads(path.read_text())
        for side in range(2):
            cards=replay['battle'][f'deck{side}']['sp']
            checked=validate_deck([{'card_id':c['d'],'level':c['l']+1} for c in cards])
            self.assertEqual(len(checked),8)
            self.assertEqual(checked[0]['card_id'],26000072)
            self.assertTrue(all(c['level']==11 for c in checked))

    def test_labels_distinguish_first_epoch_from_partial_checkpoint(self):
        partial = _policy_label({'policy_version':'expert-v1.1', 'iteration':1, 'training_step':46403})
        completed = _policy_label({'policy_version':'expert-v1.1', 'iteration':1, 'training_step':77337})
        self.assertIn('46,403', partial)
        self.assertIn('77,337', completed)
        self.assertNotEqual(partial, completed)
        self.assertNotIn('3%', completed)
        self.assertEqual(_policy_label({'policy_version':'v0.2'}), 'P050')

    def test_wrong_manifest_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'manifest.json').write_text(json.dumps({}))
            payload = {'kind':EXPERT_WEIGHTS_KIND, 'dataset_manifest_sha256':'0'*64}
            with patch('native_core.human_vs_ai.torch.load', return_value=payload):
                with self.assertRaisesRegex(RuntimeError, 'manifest do not match'):
                    _load_policy(root/'unused.pt', device=torch.device('cpu'),
                                 cuda_graph=False, expert_dataset_root=root)

    def test_hero_deck_uses_form_token(self):
        self.assertEqual(_deck_token_id({'card_id':26000000}),26000000)
        self.assertEqual(_deck_token_id({'card_id':26000000,'form_flags':2}),203000000)
        with self.assertRaises(ValueError):
            _deck_token_id({'card_id':26000000,'form_flags':3})

    def test_ability_slots_match_compiler_order_and_retain_cooldowns(self):
        def entity(key, side, available):
            return SimpleNamespace(key=key,side=side,ability_slot=1,
                                   card_id=26000072,ability_available=available)
        state=SimpleNamespace(entities=[entity(9,1,True),entity(2,0,True),entity(7,1,False)])
        candidates,tokens,mask=_ability_inputs(state,1,{26000072:5},4,True)
        self.assertEqual([e.key for e in candidates],[7,9])
        self.assertEqual(tokens,[5,5,0,0])
        self.assertEqual(mask,[False,True,False,False])
        self.assertEqual(_ability_inputs(state,1,{26000072:5},4,False)[2],[False]*4)

    def test_skill_payload_does_not_require_deployment_coordinates(self):
        self.assertEqual(_native_action({'type':'ability','side':1,'entity_id':77,'card_id':26000072}),
                         {'type':'ability','side':1,'entity_id':77})

    def test_policy_can_use_skill_with_no_playable_card(self):
        gui = HumanVsAiGui.__new__(HumanVsAiGui)
        gui.state={'episode':{'commands_allowed':True}}
        gui.env=SimpleNamespace(decks={1:[{'card_id':i+1} for i in range(8)]})
        gui.expert_card_id_to_token={i+1:i+1 for i in range(8)}
        gui.expert_ability_id_to_token={26000072:1}
        gui.expert_revealed_enemy_tokens=[]
        gui.device=torch.device('cpu')
        gui.expert_generator=torch.Generator().manual_seed(7)
        gui.ai_hidden=(torch.zeros(1,1,2),torch.zeros(1,1,2))
        captured={}
        def forward(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(hidden=gui.ai_hidden,rate_logits=torch.tensor([[50.0]]),
                action_kind_logits=torch.zeros(1,1,2),ability_logits=torch.zeros(1,1,16))
        gui.model=SimpleNamespace(config=SimpleNamespace(max_ability_slots=16,lambda_max=1000000),
                                  forward_sequence=forward)
        native=SimpleNamespace(entities=[SimpleNamespace(key=77,side=1,ability_slot=1,
                          card_id=26000072,ability_available=True)])
        actor=SimpleNamespace(own_player=SimpleNamespace(hand=(0,1,2,3),next_deck_index=4),entities=[])
        with patch('native_core.human_vs_ai.normalize_native_state',return_value=native), \
             patch('native_core.human_vs_ai.actor_projection',return_value=actor), \
             patch('native_core.human_vs_ai._grid',return_value=np.zeros((8,32,18),dtype=np.uint8)), \
             patch('native_core.human_vs_ai._public_scalars',return_value=np.zeros(16,dtype=np.float32)):
            _,_,meta=gui._sample_expert(visible_hand=[0,1,2,3],card_mask=np.zeros(4,dtype=bool),
                                       position_masks=np.zeros((4,576),dtype=bool))
        self.assertEqual(meta['action_type'],'ability')
        self.assertEqual(meta['entity_id'],77)
        self.assertEqual(captured['ability_tokens'][0,0,0].item(),1)


if __name__ == '__main__':
    unittest.main()
