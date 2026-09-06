from dataclasses import asdict
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from policy_v1.data import prepare, digest
from policy_v1.smoke import create_fixture
from hokoff_model.model import Config, Policy
from hokoff_model.audit_alignment import summarize_deployments, parser, run


def fixture(change_tick):
    n=12;event=5
    labels=np.arange(n)==event
    hand=np.tile([1,2,3,4],(n,1));hand[change_tick:,1]=9
    scalars=np.zeros((n,16),dtype=np.float32);scalars[:,1]=.6;scalars[change_tick:,1]=.3
    a={'card_label_mask':labels,'kind_label_mask':labels,'action_kind':np.zeros(n,dtype=int),
       'hand_tokens':hand,'card_slot':np.ones(n,dtype=int),'public_scalars':scalars,
       'entity_offsets':np.zeros(n+1,dtype=int),'entity_relations':np.array([],dtype=int),
       'entity_tokens':np.array([],dtype=int)}
    record={'ticks':np.arange(100,100+n),'probabilities':np.where(np.arange(n)>=change_tick,.5,.1),
            'valid':np.ones(n,dtype=bool),'labels':labels.copy()}
    return record,a


class AlignmentTests(unittest.TestCase):
    def test_pre_action_boundary_with_next_tick_effect(self):
        record,a=fixture(6)
        counts,cost,hand,jump,curves,_,examples=summarize_deployments(record,a,0,12,2,1)
        self.assertEqual(counts['isolated_deployments'],1)
        self.assertEqual(counts['at_label_elixir_drop'],0)
        self.assertEqual(counts['next_tick_elixir_drop'],1)
        self.assertEqual(counts['next_tick_hand_change'],1)
        self.assertEqual(cost,{'1':1})
        self.assertEqual(hand,{'1':1})
        self.assertEqual(jump,{'1':1})
        self.assertEqual(examples[0]['label_tick'],105)
        self.assertEqual(examples[0]['selected_token_at_label'],2)

    def test_same_tick_effect_and_nearby_action_exclusion(self):
        record,a=fixture(5)
        result=summarize_deployments(record,a,0,12,2,0)
        self.assertEqual(result[0]['at_label_elixir_drop'],1)
        self.assertEqual(result[1],{'0':1})
        record['labels'][6]=True
        counts,*_=summarize_deployments(record,a,0,12,2,0)
        self.assertEqual(counts['isolated_deployments'],0)
        self.assertEqual(counts['excluded_nearby_own_action'],1)

    def test_invalid_neighborhood_not_treated_as_no_change(self):
        record,a=fixture(6);record['valid'][3]=False
        counts,*_=summarize_deployments(record,a,0,12,2,0)
        self.assertEqual(counts['excluded_boundary_or_invalid'],1)
        self.assertEqual(counts['isolated_deployments'],0)

    def test_read_only_integration(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=create_fixture(root/'data')
            prepare(data,root/'cache',allow_smoke=True)
            config=Config(12,4,width=16,hidden_size=32,frame_window=8)
            checkpoint=root/'last.pt'
            torch.save({'config':asdict(config),'model':Policy(config).state_dict(),'step':1,
                        'contract':{'train_split':'train','val_split':'validation','targets':4,
                                    'manifest_sha256':digest(data/'manifest.json')}},checkpoint)
            before=digest(checkpoint)
            args=parser().parse_args(['--checkpoint',str(checkpoint),'--data',str(data),
                '--cache',str(root/'cache'),'--device','cpu','--workers','0','--sequences','2',
                '--batch-size','2','--radius','1','--cpu-threads','1','--allow-smoke',
                '--output',str(root/'report.json')])
            with contextlib.redirect_stdout(io.StringIO()):summary=run(args)
            self.assertEqual(summary['phase'],'alignment_summary')
            self.assertGreater(summary['counts']['isolated_deployments'],0)
            self.assertEqual(before,digest(checkpoint))
