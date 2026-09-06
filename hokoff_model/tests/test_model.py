import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch
from policy_v1.data import Windows,collate,prepare,EVENT_FIELDS
from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint
from hokoff_model.model import Policy,Config
from hokoff_model.metrics import bc_loss,summarize
from hokoff_model.train import parser,run


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        create_fixture(self.root/'data')
        prepare(self.root/'data',self.root/'cache',allow_smoke=True)
        self.ds=Windows(self.root/'data',self.root/'cache','train',targets=4,frame_window=8,event_window=1)
        self.model=Policy(Config(12,4,width=16,hidden_size=32,frame_window=8))

    def tearDown(self):
        self.tmp.cleanup()

    def slice(self,b,lo,hi):
        return {k:(v if k in EVENT_FIELDS else v[:,lo:hi]) for k,v in b.items()}

    def test_burn_in_streaming_causality_and_label_isolation(self):
        b=collate([self.ds[3],self.ds[4]])
        self.model.eval()
        with torch.no_grad():
            output=self.model(b)
            whole,state=self.model.forward_stream(b)
            valid=b['loss_mask'] & b['frame_mask']
            for k in ['timing','card','position']:
                torch.testing.assert_close(output[k][valid],whole[k][valid],rtol=1e-5,atol=1e-6)
            _,first_state=self.model.forward_stream(self.slice(b,0,5))
            latter,last_state=self.model.forward_stream(self.slice(b,5,b['frame_mask'].shape[1]),first_state)
            torch.testing.assert_close(latter['timing'],whole['timing'][:,5:],rtol=1e-5,atol=1e-6)
            for a,c in zip(state,last_state):torch.testing.assert_close(a,c,rtol=1e-5,atol=1e-6)
            changed={k:v.clone() for k,v in b.items()}
            changed['public_scalars'][:,-1]+=50
            future=self.model(changed)
            torch.testing.assert_close(output['timing'][:,:-1],future['timing'][:,:-1])
            for k in ['play_now','action_kind','card_slot','position','ability_slot','ability_position']:
                changed[k].zero_()
            changed['public_scalars']=b['public_scalars']
            changed['event_ticks'].fill_(999)
            isolated=self.model(changed)
            torch.testing.assert_close(output['timing'],isolated['timing'])

    def test_padding_empty_sides_reset_and_zero_length(self):
        b=collate([self.ds[0],self.ds[3]])
        b['entity_mask'].zero_()
        self.model.eval()
        with torch.no_grad():
            out,state=self.model.forward_stream(b)
            for i in range(2):
                n=int(b['frame_mask'][i].sum())
                single={k:v[i:i+1] for k,v in self.slice(b,0,n).items()}
                solo,s=self.model.forward_stream(single)
                torch.testing.assert_close(out['timing'][i,:n],solo['timing'][0],rtol=1e-5,atol=1e-6)
                for a,c in zip(state,s):torch.testing.assert_close(a[:,i:i+1],c,rtol=1e-5,atol=1e-6)
            reset_out,_=self.model.forward_stream(b,state,torch.ones(2,dtype=torch.bool))
            torch.testing.assert_close(out['timing'],reset_out['timing'])
            padded={k:v.clone() for k,v in b.items()};padded['frame_mask'].zero_()
            _,unchanged=self.model.forward_stream(padded,state)
            for a,c in zip(state,unchanged):torch.testing.assert_close(a,c,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(out['position']).all())

    def test_backward_detaches_burn_in_and_learns_tiny_batch(self):
        b=collate([self.ds[2],self.ds[3]])
        b['public_scalars'].requires_grad_()
        output=self.model(b)
        loss,_=bc_loss(output,b);loss.backward()
        burn=b['frame_mask'] & ~b['loss_mask']
        self.assertEqual(float(b['public_scalars'].grad[burn].abs().sum()),0)
        self.assertGreater(float(b['public_scalars'].grad[~burn].abs().sum()),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        b['public_scalars']=b['public_scalars'].detach()
        opt=torch.optim.Adam(self.model.parameters(),lr=.003)
        initial=float(loss.detach())
        for _ in range(20):
            opt.zero_grad();loss,_=bc_loss(self.model(b),b);loss.backward();opt.step()
        self.assertLess(float(loss.detach()),initial*.8)

    def test_metric_confusion_counts(self):
        b=collate([self.ds[0]])
        b['frame_mask'].fill_(True);b['loss_mask'].fill_(True);b['timing_label_mask'].fill_(True)
        b['play_now'][0]=torch.tensor([True,True,False,False])
        output=self.model(b);output['timing']=torch.tensor([[1.,-1.,1.,-1.]])
        _,stats=bc_loss(output,b);metrics=summarize(stats)
        self.assertEqual(metrics['action_precision'],.5)
        self.assertEqual(metrics['action_recall'],.5)
        self.assertEqual(metrics['predicted_action_rate'],.5)
        self.assertEqual(metrics['actual_action_rate'],.5)

    def test_train_resume_and_evaluate(self):
        def train(name,steps,resume=False):
            argv=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),
                  '--run-dir',str(self.root/name),'--allow-smoke','--device','cpu',
                  '--width','16','--hidden-size','32','--frame-window','8','--targets','4',
                  '--batch-size','2','--workers','0','--cpu-threads','1','--max-steps',str(steps),'--eval-batches','2']
            if resume:argv+=['--resume',str(self.root/name/'last.pt')]
            with contextlib.redirect_stdout(io.StringIO()):run(parser().parse_args(argv))
            return load_checkpoint(self.root/name/'last.pt')
        full=train('full',4);train('resume',2);resumed=train('resume',4,True)
        self.assertEqual(resumed['config']['architecture'],'hokoff_cr_lstm_v1')
        for k in full['model']:torch.testing.assert_close(full['model'][k],resumed['model'][k],rtol=0,atol=0)

    def test_resume_across_epoch_boundary(self):
        def train(name, steps, resume=False):
            argv = ['--data', str(self.root/'data'), '--cache', str(self.root/'cache'),
                    '--run-dir', str(self.root/name), '--allow-smoke', '--device', 'cpu',
                    '--width', '16', '--hidden-size', '32', '--frame-window', '8',
                    '--targets', '4', '--batch-size', '2', '--workers', '0',
                    '--cpu-threads', '1', '--max-steps', str(steps), '--eval-batches', '1']
            if resume:
                argv += ['--resume', str(self.root/name/'last.pt')]
            with contextlib.redirect_stdout(io.StringIO()):
                run(parser().parse_args(argv))
            return load_checkpoint(self.root/name/'last.pt')
        full = train('epoch_full', 8)
        boundary = train('epoch_resumed', 6)
        self.assertEqual(boundary['epoch'], 1)
        self.assertEqual(boundary['next_batch'], 0)
        resumed = train('epoch_resumed', 8, True)
        for key in full['model']:
            torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
