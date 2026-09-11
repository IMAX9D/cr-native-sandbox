import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
from policy_v1.smoke import create_fixture
from policy_v1.data import digest
from policy_v1.train import load_checkpoint
from hokoff_model.decision_data import prepare
from hokoff_model.train_fixed import FixedConfig, FixedPolicy, parser, run
from hokoff_model.weights_restart import prepare_restart, initialize_weights


class WeightsRestartTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        create_fixture(self.root/'data',steps=81)
        self.config=FixedConfig(12,4,width=16,hidden_size=32,frame_window=3,
                               history_length=4,spatial_skip_channels=4)
        model=FixedPolicy(self.config)
        with torch.no_grad(): model.timing.bias.fill_(1.234)
        self.saved=dict(config=asdict(self.config),model=model.state_dict(),step=1037042)
        self.weights=self.root/'release.pt'; torch.save(self.saved,self.weights)
        manifest=json.loads((self.root/'data/manifest.json').read_text())
        manifest['card_vocabulary']=list(range(12))
        manifest['ability_vocabulary']=list(range(4))
        (self.root/'data/manifest.json').write_text(json.dumps(manifest))
        self.contract=self.root/'encoder.json'
        self.contract.write_text(json.dumps(dict(kind='hokoff_match_encoder_contract_v1',
            allowed_weights=[digest(self.weights)],source_manifest_sha256=digest(self.root/'data/manifest.json'),
            encoder={k:manifest[k] for k in ('dimensions','card_vocabulary','ability_vocabulary')})))
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.root/'data',self.root/'cache',sampling='fixed',allow_smoke=True,auxiliary_frame_window=3)

    def tearDown(self): self.tmp.cleanup()

    def test_exact_weights_and_reject_mismatches(self):
        saved,_=prepare_restart(self.weights,self.contract,self.root/'data')
        model=initialize_weights(self.config,saved=saved)
        for k,v in saved['model'].items(): torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(model.timing.weight.requires_grad)
        altered=FixedConfig(**dict(asdict(self.config),decision_period=8))
        with self.assertRaisesRegex(ValueError,'configuration differs'): initialize_weights(altered,saved=saved)
        with (self.root/'data/manifest.json').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'manifest differs'): prepare_restart(self.weights,self.contract,self.root/'data')

    def train(self,name,steps,resume=False):
        args=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),'--run',str(self.root/name),
              '--device','cpu','--workers','0','--cpu-threads','1','--allow-smoke','--width','16',
              '--hidden-size','32','--frame-window','3','--targets','4','--batch-size','2',
              '--history-length','4','--spatial-skip-channels','4','--max-steps',str(steps),'--eval-batches','1']
        args+=['--resume',str(self.root/name/'last.pt')] if resume else ['--init-weights',str(self.weights),'--weights-contract',str(self.contract)]
        with contextlib.redirect_stdout(io.StringIO()): run(parser().parse_args(args))
        return load_checkpoint(self.root/name/'last.pt')

    def test_fresh_optimizer_and_exact_resume(self):
        first=self.train('resumed',1)
        self.assertEqual(first['step'],1)
        self.assertEqual(first['contract']['weights_restart']['source_step'],1037042)
        self.assertTrue(first['optimizer']['state'])
        self.assertTrue(all(int(v['step'])==1 for v in first['optimizer']['state'].values()))
        self.assertFalse(torch.equal(first['model']['entity.0.weight'],self.saved['model']['entity.0.weight']))
        continued=self.train('resumed',2,True)
        full=self.train('full',2)
        for k,v in full['model'].items():torch.testing.assert_close(v,continued['model'][k],rtol=0,atol=0)
        with self.assertRaises(FileExistsError):self.train('full',3)


if __name__=='__main__': unittest.main()
