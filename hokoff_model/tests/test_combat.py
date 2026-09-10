import contextlib,io,json,tempfile,copy
from pathlib import Path
from dataclasses import asdict
import unittest
import torch
from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint
from hokoff_model.decision_data import prepare,DecisionWindows,collate_decisions
from hokoff_model.train_fixed import FixedConfig,FixedPolicy,parser,run
from hokoff_model.metrics import bc_loss
from train_hokoff_combat import migrate,TABLE

class CombatTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(23)
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  create_fixture(self.root/'data',steps=81)
  with contextlib.redirect_stdout(io.StringIO()):prepare(self.root/'data',self.root/'cache',sampling='fixed',allow_smoke=True,auxiliary_frame_window=3)
  self.ds=DecisionWindows(self.root/'data',self.root/'cache','validation',sampling='fixed',decision_period=4,targets=4,frame_window=3,history_length=4)
  self.b=collate_decisions([self.ds[0],self.ds[1]])
  self.features=[[0.]*20]+[[float(i)/12]*10+[1.]*10 for i in range(1,12)]
 def tearDown(self):self.ds.close();self.tmp.cleanup()
 def test_migration_preserves_predictions_optimizer_and_learns(self):
  c=FixedConfig(12,4,width=16,hidden_size=32,history_length=4,spatial_skip_channels=4)
  old=FixedPolicy(c);opt=torch.optim.AdamW(old.parameters(),lr=.001)
  loss,_=bc_loss(old(self.b),self.b);loss.backward();opt.step();opt.zero_grad()
  saved=dict(config=asdict(c),model=copy.deepcopy(old.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),step=17)
  before=old(self.b)
  migrated=migrate(saved,self.features);new=FixedPolicy(FixedConfig(**migrated['config']));new.load_state_dict(migrated['model'])
  for k,v in before.items():torch.testing.assert_close(v,new(self.b)[k],rtol=1e-5,atol=1e-6)
  self.assertEqual(migrated['step'],17)
  self.assertEqual(float(new.entity[0].weight[:,-20:].detach().abs().sum()),0)
  newopt=torch.optim.AdamW(new.parameters(),lr=.001);newopt.load_state_dict(migrated['optimizer'])
  momentum=newopt.state[new.entity[0].weight]['exp_avg']
  torch.testing.assert_close(momentum[:,:-20],opt.state[old.entity[0].weight]['exp_avg'])
  self.assertEqual(float(momentum[:,-20:].abs().sum()),0)
  loss,_=bc_loss(new(self.b),self.b);loss.backward()
  self.assertGreater(float(new.entity[0].weight.grad[:,-20:].abs().sum()),0)
  newopt.step();self.assertTrue(torch.isfinite(new.entity[0].weight).all())
 def test_table_and_unknown_fields(self):
  d=json.loads(TABLE.read_text());self.assertEqual(len(d['features']),len(d['card_vocabulary']))
  ix=d['card_vocabulary'].index('musketeer@26000014');v=d['features'][ix]
  self.assertEqual(v[2:4],[1,1]);self.assertEqual(v[7],.5)
  ix=d['card_vocabulary'].index('goblin-gang@26000041')
  self.assertEqual(d['features'][ix][11:],[0.]*9)
  for row in d['features']:
   self.assertEqual(len(row),20)
   for value,known in zip(row[:10],row[10:]):
    if not known:self.assertEqual(value,0)
 def test_training_resume_with_table(self):
  table=self.root/'table.json';manifest=json.loads((self.root/'data/manifest.json').read_text())
  manifest['card_vocabulary']=['PAD']+[str(i) for i in range(1,12)]
  # Fixture manifest lacks vocab: rebuild cache against explicit fixture vocab.
  (self.root/'data/manifest.json').write_text(json.dumps(manifest))
  with contextlib.redirect_stdout(io.StringIO()):prepare(self.root/'data',self.root/'cache2',sampling='fixed',allow_smoke=True,auxiliary_frame_window=3)
  table.write_text(json.dumps(dict(schema='cr_nominal_static_combat_v1',game_version='15.535.29',card_vocabulary=manifest['card_vocabulary'],features=self.features)))
  def train(name,steps,resume=False):
   argv=['--data',str(self.root/'data'),'--cache',str(self.root/'cache2'),'--run',str(self.root/name),'--device','cpu',
         '--allow-smoke','--workers','0','--cpu-threads','1','--width','16','--hidden-size','32','--frame-window','3',
         '--targets','4','--batch-size','2','--history-length','4','--spatial-skip-channels','4',
         '--combat-features-file',str(table),'--max-steps',str(steps),'--eval-batches','1']
   if resume:argv+=['--resume',str(self.root/name/'last.pt')]
   with contextlib.redirect_stdout(io.StringIO()):run(parser().parse_args(argv))
   return load_checkpoint(self.root/name/'last.pt')
  full=train('full',4);train('resume',2);other=train('resume',4,True)
  for k in full['model']:torch.testing.assert_close(full['model'][k],other['model'][k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
