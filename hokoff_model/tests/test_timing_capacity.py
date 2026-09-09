import contextlib
from dataclasses import asdict
import io
import tempfile
from pathlib import Path
import unittest
import torch

from policy_v1.smoke import create_fixture
from policy_v1.data import digest
from policy_v1.train import load_checkpoint
from hokoff_model.decision_data import prepare,DecisionWindows,collate_decisions
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
from hokoff_model.capacity_model import CapacityConfig,initialize_from_source,policy_from_config
from hokoff_model.train_capacity import parser,run
import experiment_hokoff_timing_capacity as experiment


class CapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        create_fixture(self.root/'data',steps=81)
        prepare(self.root/'data',self.root/'cache',sampling='fixed',allow_smoke=True,auxiliary_frame_window=3)
        self.config=FixedConfig(12,4,width=16,hidden_size=32,frame_window=3)
        self.original=FixedPolicy(self.config)
        self.source=self.root/'source.pt'
        torch.save(dict(config=asdict(self.config),model=self.original.state_dict(),step=100,
            contract=dict(decision_cache_sha256=digest(self.root/'cache/index.json'),
                manifest_sha256=digest(self.root/'data/manifest.json'),train_split='validation',val_split='train',
                targets=4,batch_size_per_rank=2,world_size=1,precision='fp32',lr=3e-4,
                weight_decay=1e-4,grad_clip=1.,timing_positive_weight=32)),self.source)
        self.ds=DecisionWindows(self.root/'data',self.root/'cache','validation',targets=4,frame_window=3,sampling='fixed')
        self.batch=collate_decisions([self.ds[0],self.ds[1]])

    def tearDown(self):self.ds.close();self.tmp.cleanup()

    def residual_config(self):
        return CapacityConfig(**dict(asdict(self.config),architecture=CapacityConfig.architecture,timing_hidden_size=16))

    def test_initial_outputs_and_pretrained_parameters_are_identical(self):
        for c in (self.config,self.residual_config()):
            model=initialize_from_source(c,checkpoint=self.source)
            for name,value in self.original.state_dict().items():
                torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
            with torch.no_grad():a=self.original(self.batch);b=model(self.batch)
            for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
        with self.assertRaises(ValueError):policy_from_config(dict(architecture='wrong'))

    def test_residual_only_changes_timing_and_receives_gradients(self):
        model=initialize_from_source(self.residual_config(),checkpoint=self.source)
        before=model(self.batch)
        before['timing'].sum().backward()
        self.assertGreater(float(model.timing_residual[-1].weight.grad.abs().sum()),0)
        self.assertEqual(float(model.timing_residual[0].weight.grad.abs().sum()),0)
        with torch.no_grad():model.timing_residual[-1].weight.fill_(.01)
        after=model(self.batch)
        self.assertFalse(torch.equal(before['timing'],after['timing']))
        for name in before:
            if name!='timing':torch.testing.assert_close(before[name],after[name],rtol=0,atol=0)
        self.assertFalse(model.delay.weight.requires_grad)

    def test_capacity_resume_is_exact(self):
        def train(name,steps,resume=False):
            argv=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),'--run',str(self.root/name),
                '--source-checkpoint',str(self.source),'--variant','residual','--timing-hidden-size','16',
                '--device','cpu','--precision','fp32','--workers','0','--cpu-threads','1','--allow-smoke',
                '--width','16','--hidden-size','32','--frame-window','3','--targets','4','--batch-size','2',
                '--max-steps',str(steps),'--epochs','10','--eval-every','0','--eval-batches','2']
            if resume:argv+=['--resume',str(self.root/name/'last.pt')]
            with contextlib.redirect_stdout(io.StringIO()):run(parser().parse_args(argv))
            return load_checkpoint(self.root/name/'last.pt')
        full=train('full',4);train('resume',2);continued=train('resume',4,True)
        for name,value in full['model'].items():torch.testing.assert_close(value,continued['model'][name],rtol=0,atol=0)
        self.assertGreater(float(full['model']['timing_residual.2.weight'].abs().sum()),0)

    def test_paired_runner_and_both_evaluators(self):
        before=digest(self.source)
        args=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),
            '--source-checkpoint',str(self.source),'--output',str(self.root/'paired'),
            '--steps','2','--eval-every','2','--eval-batches','2','--device','cpu',
            '--workers','0','--cpu-threads','1','--timing-hidden-size','16']
        with contextlib.redirect_stdout(io.StringIO()):result=experiment.main(args)
        self.assertTrue(result['initial_audit']['all_initial_outputs_exactly_equal'])
        self.assertTrue(result['matched_training_progress'])
        self.assertTrue(result['matched_evaluation_windows_and_labels'])
        self.assertEqual(digest(self.source),before)
        self.assertEqual(result['trials']['linear']['cursor'],result['trials']['residual']['cursor'])
        for name in ('linear','residual'):
            ck=load_checkpoint(self.root/'paired'/name/'last.pt')
            self.assertEqual(ck['step'],2);self.assertEqual(ck['contract']['optimizer_initialization'],'fresh')


if __name__=='__main__':unittest.main()
