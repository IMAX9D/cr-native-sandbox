import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from policy_v1.data import Windows, collate, prepare
from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint
from hokoff_model.model import Policy, Config
from hokoff_model.metrics import bc_loss
from hokoff_model.experiments import parser, overfit, compare, select_windows
from hokoff_model.train import parser as training_parser, run as train


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_fixture(self.root/'data')
        prepare(self.root/'data',self.root/'cache',allow_smoke=True)
        self.ds = Windows(self.root/'data',self.root/'cache','train',
                          targets=4,frame_window=8,event_window=1)

    def tearDown(self):
        self.tmp.cleanup()

    def arguments(self, mode):
        return parser(mode).parse_args([
            '--data',str(self.root/'data'),'--cache',str(self.root/'cache'),
            '--run-dir',str(self.root/mode),'--train-split','train','--val-split','validation',
            '--device','cpu','--width','16','--hidden-size','32','--frame-window','8',
            '--targets','4','--cpu-threads','1','--steps','2','--allow-smoke',
        ] + (['--positive-windows','2','--wait-windows','2','--scan-windows','12','--log-every','1']
             if mode=='overfit' else ['--batch-size','2','--workers','0','--eval-batches','2']))

    def test_positive_weight_changes_only_positive_timing_gradients(self):
        selected=select_windows(self.ds,2,2,12,42)
        b=collate([x[1] for x in selected])
        model=Policy(Config(12,4,width=16,hidden_size=32,frame_window=8))
        output={k:v.detach().requires_grad_() for k,v in model(b).items()}
        l1,s1=bc_loss(output,b)
        g1=torch.autograd.grad(l1,(output['timing'],output['card']),retain_graph=True)
        l32,s32=bc_loss(output,b,timing_positive_weight=32)
        g32=torch.autograd.grad(l32,(output['timing'],output['card']))
        expected=g1[0]*torch.where(b['play_now'],32.,1.)
        torch.testing.assert_close(g32[0],expected)
        torch.testing.assert_close(g32[1],g1[1])
        self.assertEqual(s1['timing_unweighted_sum'],s32['timing_unweighted_sum'])
        with self.assertRaises(ValueError):
            bc_loss(output,b,timing_positive_weight=float('nan'))

    def test_selection_is_fixed_and_retains_wait_labels(self):
        a=select_windows(self.ds,2,2,12,42)
        z=select_windows(self.ds,2,2,12,42)
        self.assertEqual([i for i,_ in a],[i for i,_ in z])
        self.assertEqual(len({i for i,_ in a}),4)
        for j,(_,b) in enumerate(a):
            mask=b['loss_mask'] & b['frame_mask'] & b['timing_label_mask']
            self.assertEqual(bool((mask & b['play_now']).any()),j<2)

    def test_overfit_diagnostic_writes_separate_checkpoint(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=overfit(self.arguments('overfit'))
        self.assertGreater(result['actual_actions'],0)
        saved=load_checkpoint(self.root/'overfit/diagnostic.pt')
        self.assertEqual(saved['purpose'],'overfit diagnostic only')
        self.assertFalse((self.root/'overfit/last.pt').exists())

    def test_compare_and_weighted_resume_contract(self):
        args=self.arguments('compare')
        with contextlib.redirect_stdout(io.StringIO()):
            result=compare(args)
        a,b=result['arms']
        self.assertEqual(a['actual_actions'],b['actual_actions'])
        self.assertEqual(a['valid_frames'],b['valid_frames'])
        self.assertEqual(a['checkpoint_step'],b['checkpoint_step'])
        baseline=load_checkpoint(self.root/'compare/baseline/last.pt')
        weighted=load_checkpoint(self.root/'compare/weighted/last.pt')
        self.assertNotIn('timing_positive_weight',baseline['contract'])
        self.assertEqual(weighted['contract']['timing_positive_weight'],32)
        config=json.loads((self.root/'compare/weighted/config.json').read_text())
        argv=[]
        # Rebuild the saved CLI but omit the weight to simulate accidental resume
        # under the old objective. Internal defaults are not CLI options.
        p=training_parser()
        destinations={action.dest:action for action in p._actions}
        for key,value in config['arguments'].items():
            if key in ('heads','layers','event_window','timing_positive_weight') or value is None:
                continue
            if key not in destinations:continue
            flag=destinations[key].option_strings[0]
            if isinstance(value,bool):
                if value:argv.append(flag)
            else:argv += [flag,str(value)]
        argv+=['--resume',str(self.root/'compare/weighted/last.pt')]
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'contract differs'):
            train(p.parse_args(argv))
