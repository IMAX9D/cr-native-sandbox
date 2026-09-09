import contextlib
import io
import tempfile
from pathlib import Path
import unittest
import torch
from hokoff_model.train_fixed import FixedPolicy, FixedConfig, parser, run
from hokoff_model.spatial_position import SpatialPositionHead
from hokoff_model.metrics import bc_loss
from hokoff_model.decision_data import prepare, DecisionWindows, collate_decisions
from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint


class PositionSkipTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        create_fixture(self.root/'data',steps=81)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.root/'data',self.root/'cache',sampling='fixed',allow_smoke=True,auxiliary_frame_window=3)
        self.ds=DecisionWindows(self.root/'data',self.root/'cache','validation',sampling='fixed',
                               decision_period=4,targets=4,frame_window=3,history_length=4)
        self.batch=collate_decisions([self.ds[0],self.ds[1]])

    def tearDown(self):self.ds.close();self.tmp.cleanup()

    def model(self,skip=4):
        return FixedPolicy(FixedConfig(12,4,width=16,hidden_size=32,frame_window=3,
                                       history_length=4,spatial_type_dim=2,spatial_skip_channels=skip))

    def test_initial_equivalence_and_learning(self):
        torch.manual_seed(3);base=self.model(0)
        torch.manual_seed(3);model=self.model(4)
        for k,v in base.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        before=base(self.batch);out=model(self.batch)
        for k in before:torch.testing.assert_close(out[k],before[k],rtol=0,atol=0)
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        loss,_=bc_loss(out,self.batch);loss.backward()
        self.assertGreater(float(model.position_skip.query[-1].weight.grad.abs().sum()),0)
        opt.step();opt.zero_grad()
        loss,_=bc_loss(model(self.batch),self.batch);loss.backward()
        self.assertGreater(float(model.position_skip.spatial[0].weight.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(model.position_skip.spatial[0].weight.grad).all())

    def test_local_grid_and_card_conditioning(self):
        head=SpatialPositionHead(8,16,4)
        with torch.no_grad():
            for p in head.parameters():p.fill_(.05)
        grid=torch.zeros(1,8,32,18);context=torch.zeros(1,16);hand=torch.zeros(1,4,16)
        out=head(grid,context,hand)
        changed=grid.clone();changed[0,0,12,8]=1
        difference=head(changed,context,hand)-out
        self.assertGreater(float(difference.detach().abs().sum()),0)
        affected=torch.zeros(32,18,dtype=torch.bool);affected[10:15,6:11]=True
        self.assertEqual(float(difference.detach()[...,~affected.flatten()].abs().sum()),0)
        other=hand.clone();other[:,1]=1
        conditioned=head(grid,context,other)
        torch.testing.assert_close(conditioned[:,0],out[:,0])
        self.assertFalse(torch.equal(conditioned[:,1],out[:,1]))

    def test_causality_padding_and_other_heads_unchanged(self):
        model=self.model()
        with torch.no_grad():model.position_skip.query[-1].weight.fill_(.1)
        b={k:v.clone() for k,v in self.batch.items()}
        b['grid'].requires_grad_()
        out=model(b)
        changed={k:v.detach().clone() for k,v in b.items()}
        changed['grid'][:,-1]+=10
        future=model(changed)
        torch.testing.assert_close(out['position'][:,:-1],future['position'][:,:-1])
        changed={k:v.detach().clone() for k,v in b.items()}
        for k in ('play_now','card_slot','position'):changed[k].zero_()
        torch.testing.assert_close(out['position'],model(changed)['position'])
        recurrent=torch.randn(*b['frame_mask'].shape,32)
        enabled=model.heads(recurrent,b)
        model.config.spatial_skip_channels=0
        disabled=model.heads(recurrent,b)
        for k in ('timing','kind','card','ability','ability_position'):
            torch.testing.assert_close(enabled[k],disabled[k],rtol=0,atol=0)
        self.assertFalse(torch.equal(enabled['position'],disabled['position']))
        model.config.spatial_skip_channels=4
        loss,_=bc_loss(out,b);loss.backward()
        burn=b['frame_mask'] & ~b['loss_mask']
        self.assertEqual(float(b['grid'].grad[burn].abs().sum()),0)
        empty={k:v.detach().clone() for k,v in b.items()};empty['frame_mask'].zero_()
        result=model.heads(recurrent,empty)
        self.assertTrue(torch.isfinite(result['position']).all())

    def test_train_resume_evaluate_and_legacy(self):
        def train(name,steps,skip,resume=False):
            args=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),'--run',str(self.root/name),
                  '--device','cpu','--workers','0','--cpu-threads','1','--allow-smoke','--width','16',
                  '--hidden-size','32','--frame-window','3','--targets','4','--batch-size','2',
                  '--history-length','4','--spatial-skip-channels',str(skip),'--max-steps',str(steps),'--eval-batches','1']
            if resume:args+=['--resume',str(self.root/name/'last.pt')]
            with contextlib.redirect_stdout(io.StringIO()):run(parser().parse_args(args))
            return load_checkpoint(self.root/name/'last.pt')
        full=train('full',4,4);train('resume',2,4);resumed=train('resume',4,4,True)
        for k in full['model']:torch.testing.assert_close(full['model'][k],resumed['model'][k],rtol=0,atol=0)
        from hokoff_model.evaluate_fixed import main
        with contextlib.redirect_stdout(io.StringIO()):
            result=main(['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),
                         '--checkpoint',str(self.root/'full/last.pt'),'--output',str(self.root/'eval'),
                         '--device','cpu','--workers','0','--batch-size','2','--batches','1'])
        self.assertIn('timing_ranking',result)
        location=result['conditional_position']
        self.assertEqual(location['count'],result['joint_counts']['deploy_count'])
        self.assertGreaterEqual(location['top5_accuracy'],location['accuracy'])
        self.assertEqual(sum(v['count'] for v in result['conditional_position_by_card_token'].values()),location['count'])
        old=train('old',2,0);old['config'].pop('spatial_skip_channels');torch.save(old,self.root/'old/last.pt')
        train('old',3,0,True)
        with self.assertRaisesRegex(ValueError,'contract differs'):train('old',4,4,True)


if __name__ == '__main__':unittest.main()
