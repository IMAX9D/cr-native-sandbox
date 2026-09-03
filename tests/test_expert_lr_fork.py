from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
import torch
from expert_v1.training_v1.fork_run import migrate_lr_checkpoint, create_fork
from expert_v1.training_v1 import train
from expert_v1.training_v1.train import LiveTrainingWindow


class LearningRateForkTests(unittest.TestCase):
    def test_equal_lr_control_requires_explicit_flag_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            args = train.build_parser().parse_args(['--smoke', '--dataset-root', str(root/'data'),
                '--output-root', str(root/'runs'), '--run-id', 'parent', '--epochs', '2',
                '--stop-after-epoch', '1', '--batch-size', '2', '--sequence-length', '16',
                '--burn-in', '4', '--hidden-size', '16', '--card-embedding-size', '8',
                '--spatial-size', '8', '--device', 'cpu', '--workers', '0', '--integrity-workers', '1'])
            source = train.run(args)
            saved = torch.load(source/'checkpoints/latest.pt', weights_only=False)
            with self.assertRaises(ValueError):
                create_fork(source, source/'checkpoints/latest.pt', root/'arms', 'denied',
                            args.learning_rate, saved['global_step'])
            receipt = create_fork(source, source/'checkpoints/latest.pt', root/'arms', 'control',
                                  args.learning_rate, saved['global_step'], allow_equal_learning_rate=True)
            self.assertTrue(receipt['checkpoint_state_verified'])
            control = torch.load(root/'arms/control/checkpoints/latest.pt', weights_only=False)
            self.assertEqual(control['optimizer_state']['param_groups'][0]['lr'], args.learning_rate)
            self.assertTrue(torch.equal(saved['rng']['train_loader_generator'], control['rng']['train_loader_generator']))
            with self.assertRaises(ValueError):
                create_fork(source, source/'checkpoints/latest.pt', root/'arms', 'increase',
                            args.learning_rate*2, saved['global_step'], allow_equal_learning_rate=True)

    def fixture(self):
        return {'run_id':'parent','global_step':123740,
                'model_state':{'weight':torch.tensor([1.25,2.5])},
                'optimizer_state':{'state':{0:{'step':torch.tensor(123740),
                    'exp_avg':torch.tensor([0.2,0.4]),'exp_avg_sq':torch.tensor([0.01,0.02])}},
                    'param_groups':[{'params':[0],'lr':3e-4,'initial_lr':3e-4,'weight_decay':1e-4}]},
                'scheduler_state':{'base_lrs':[3e-4],'_last_lr':[3e-4],'last_epoch':1,'_step_count':2},
                'rng':{'torch':torch.tensor([1,2,3],dtype=torch.uint8)},
                'run_signature_sha256':'old-signature','optimizer_identity_sha256':'old-optimizer'}

    def test_lr_changes_without_reinitializing_weights_or_moments(self):
        original=self.fixture(); before=deepcopy(original)
        result=migrate_lr_checkpoint(original,learning_rate=1e-4,run_id='child',
            signature='new-signature',optimizer_identity='new-optimizer')
        self.assertIs(result['model_state'],original['model_state'])
        self.assertIs(result['optimizer_state']['state'],original['optimizer_state']['state'])
        self.assertIs(result['rng'],original['rng'])
        self.assertEqual(result['global_step'],123740)
        self.assertEqual(result['optimizer_state']['param_groups'][0]['lr'],1e-4)
        self.assertEqual(result['optimizer_state']['param_groups'][0]['initial_lr'],1e-4)
        self.assertEqual(result['scheduler_state']['base_lrs'],[1e-4])
        self.assertEqual(result['scheduler_state']['_last_lr'],[1e-4])
        self.assertEqual(result['scheduler_state']['last_epoch'],1)
        self.assertEqual(original['optimizer_state']['param_groups'],before['optimizer_state']['param_groups'])
        self.assertEqual(original['scheduler_state'],before['scheduler_state'])

    def test_rejects_invalid_rate_and_scheduler(self):
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                migrate_lr_checkpoint(self.fixture(),learning_rate=value,run_id='x',signature='x',optimizer_identity='x')
        bad=self.fixture();bad['scheduler_state']['base_lrs']=[]
        with self.assertRaises(ValueError):
            migrate_lr_checkpoint(bad,learning_rate=1e-4,run_id='x',signature='x',optimizer_identity='x')

    def test_window_is_real_batch_mean_and_resets_cleanly(self):
        window=LiveTrainingWindow()
        window.add({'loss':4.0,'loss_position':3.0,'loss_card':0.8,'gradient_norm':2.0})
        window.add({'loss':22.0,'loss_position':20.0,'loss_card':1.8,'gradient_norm':10.0})
        result=window.summary()
        self.assertEqual(result['window_batches'],2)
        self.assertEqual(result['loss_window_mean'],13.0)
        self.assertEqual(result['loss_position_window_mean'],11.5)
        self.assertEqual(result['gradient_norm_window_mean'],6.0)
        self.assertEqual(result['loss_window_max'],22.0)
        self.assertEqual(result['loss_window_gt10'],1)
        self.assertEqual(result['loss_window_gt20'],1)
        self.assertEqual(LiveTrainingWindow().summary()['window_batches'],0)


if __name__=='__main__':
    unittest.main()
