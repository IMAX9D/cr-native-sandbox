from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
from hokoff_model.ppo import state_digest
from hokoff_model.ppo_actions import action_heads
from hokoff_model.ppo_opponent import FrozenOpponent,against_opponent_logits,resolve_win_rate_window


def episode(index,*,win=True,draw=False,side=None,opponent=1):
    side=index%2 if side is None else side
    return dict(episode=index,learner_side=side,opponent_number=opponent,
                terminal=dict(terminated=True,truncated=False,
                              outcome='draw' if draw else f'side{side if win else 1-side}_win'))


class OpponentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        self.model=FixedPolicy(FixedConfig(9,3,width=8,hidden_size=16,frame_window=3)).eval()
        self.league=FrozenOpponent(self.model)

    def test_each_side_uses_correct_model_heads(self):
        features=dict(context=torch.randn(2,8),hand_tokens=torch.ones(2,4,dtype=torch.long),
                      ability_tokens=torch.ones(2,16,dtype=torch.long))
        with torch.no_grad():
            self.model.timing.bias.add_(10)
            learner=action_heads(self.model,features)
            rival=action_heads(self.league.model,features)
            output={k:v[:,None] for k,v in learner.items()}
            for side in (0,1):
                mixed=against_opponent_logits(output,features,self.league.model,side)
                for key in learner:
                    torch.testing.assert_close(mixed[key][side],learner[key][side])
                    torch.testing.assert_close(mixed[key][1-side],rival[key][1-side])
                self.assertNotEqual(float(mixed['timing'][1-side]),float(learner['timing'][1-side]))
        self.league.assert_unchanged()

    def test_real_ppo_update_cannot_change_opponent(self):
        from hokoff_model.tests.test_ppo import PPOTests
        case=PPOTests();case.setUp();trainer=case.trainer()
        rival=FrozenOpponent(case.model);before=state_digest(case.model)
        stats=trainer.update(case.data,generator=case.generator)
        self.assertGreater(stats['accepted_updates'],0)
        self.assertNotEqual(state_digest(case.model),before)
        self.assertEqual(state_digest(rival.model),before);rival.assert_unchanged()
        rival.check_body(case.model)

    def test_exactly_95_percent_does_not_promote_but_96_does(self):
        for i in range(100):self.league.record(episode(i,win=i>=5),policy_step=10+i//2)
        self.assertEqual(self.league.statistics()['win_rate'],.95)
        self.assertIsNone(self.league.maybe_promote(self.model,policy_step=60,iteration=50))
        self.league.record(episode(100),policy_step=60)
        record=self.league.maybe_promote(self.model,policy_step=60,iteration=51)
        self.assertEqual(record['filename'],'beat_1.pt');self.assertEqual(record['win_rate'],.96)
        self.assertEqual(record['side0_games'],50);self.assertEqual(record['side1_games'],50)
        self.assertEqual(self.league.number,2);self.assertEqual(self.league.name,'beat_1')
        self.assertEqual(self.league.statistics()['win_rate_games'],0)
        self.assertIsNone(self.league.statistics()['win_rate'])
        with torch.no_grad():self.model.timing.bias.add_(1)
        self.league.assert_unchanged()
        self.assertNotEqual(state_digest(self.model),self.league.model_hash)

    def test_full_window_balanced_sides_and_draw_denominator(self):
        for i in range(99):self.league.record(episode(i),policy_step=0)
        self.assertFalse(self.league.statistics()['promotion_ready'])
        self.league.record(episode(99,draw=True),policy_step=0)
        stats=self.league.statistics()
        self.assertEqual(stats['win_rate'],.99);self.assertEqual(stats['draws'],1)
        unbalanced=FrozenOpponent(self.model)
        for i in range(100):unbalanced.record(episode(i,side=0),policy_step=0)
        self.assertFalse(unbalanced.statistics()['promotion_ready'])

    def test_invalid_and_duplicate_games_not_counted(self):
        self.league.record(episode(0),policy_step=0)
        with self.assertRaises(ValueError):self.league.record(episode(0),policy_step=0)
        with self.assertRaises(ValueError):self.league.record(episode(1,opponent=2),policy_step=0)
        for fields in [dict(terminated=False),dict(truncated=True),dict(outcome='ongoing')]:
            row=episode(1);row['terminal'].update(fields)
            with self.assertRaises(ValueError):self.league.record(row,policy_step=0)
        self.assertEqual(self.league.games,1)

    def test_checkpoint_restores_window_opponent_and_numbered_exports(self):
        from train_hokoff_ppo import save_checkpoint
        from hokoff_model.tests.test_ppo import PPOTests
        case=PPOTests();case.setUp();trainer=case.trainer()
        league=FrozenOpponent(case.model)
        for i in range(98):league.record(episode(i),policy_step=1)
        with TemporaryDirectory() as directory:
            output=Path(directory)
            save_checkpoint(output/'last.pt',trainer,SimpleNamespace(model=case.model),
                            dict(source_step=1,manifest_sha256='test'),case.generator,49,league)
            checkpoint=torch.load(output/'last.pt',weights_only=False)
            restored=FrozenOpponent(case.model);restored.load_state_dict(checkpoint['opponent'])
            self.assertEqual(restored.statistics(),league.statistics())
            self.assertEqual(restored.model_hash,league.model_hash)
            for i in (98,99):restored.record(episode(i),policy_step=2)
            with torch.no_grad():case.model.timing.bias.add_(.1)
            record=restored.maybe_promote(case.model,policy_step=2,iteration=50)
            # Round-trip after promotion, then export from the committed state.
            recovered=FrozenOpponent(case.reference);recovered.load_state_dict(restored.state_dict())
            recovered.export_latest(output,source_contract=dict(manifest_sha256='test'))
            saved=torch.load(output/'beat_1.pt',weights_only=False)
            self.assertEqual(saved['promotion'],record);self.assertEqual(saved['step'],2)
            for k,v in recovered.model.state_dict().items():torch.testing.assert_close(saved['model'][k],v)
            recovered.export_latest(output,source_contract=dict(manifest_sha256='test'))
            (output/'beat_1.pt').unlink()
            recovered.export_latest(output,source_contract=dict(manifest_sha256='test'))
            self.assertTrue((output/'beat_1.pt').exists())
            for i in range(100,200):recovered.record(episode(i,opponent=2),policy_step=3)
            second=recovered.maybe_promote(case.model,policy_step=3,iteration=100)
            self.assertEqual(second['filename'],'beat_2.pt')
            recovered.export_latest(output,source_contract=dict(manifest_sha256='test'))
            self.assertTrue((output/'beat_1.pt').exists());self.assertTrue((output/'beat_2.pt').exists())
            self.assertEqual(recovered.statistics()['opponents_beaten'],2)

    def test_modified_backbone_and_tampered_opponent_are_rejected(self):
        bad=deepcopy(self.model)
        body_name=next(n for n in bad.state_dict() if n.startswith('context.'))
        with torch.no_grad():bad.state_dict()[body_name].add_(1)
        with self.assertRaises(ValueError):self.league.check_body(bad)
        saved=self.league.state_dict();saved['model']['timing.bias'].add_(1)
        with self.assertRaises(RuntimeError):FrozenOpponent(self.model).load_state_dict(saved)

    def test_window_configuration_is_restored_and_validated(self):
        self.assertEqual(resolve_win_rate_window(None),100)
        self.assertEqual(resolve_win_rate_window(None,dict(promotion_window=200)),200)
        for value in (0,1,99,2.5):
            with self.assertRaises(ValueError):resolve_win_rate_window(value)
        with self.assertRaises(ValueError):resolve_win_rate_window(50,dict(promotion_window=100))


if __name__=='__main__':unittest.main()
