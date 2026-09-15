from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from data_pipeline.r0_training_smoke import assert_tree_equal,save_checkpoint
from training_r0.actions import ActionSequence
from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig
from training_r0.il_dataset import ILSequence
from training_r0.il_learning import update_imitation_sequence
from training_r0.model import RecurrentState
from training_r0.observation import ObservationBuilder
from training_r0.synthetic import public_frame


def test_checkpoint_publish_roundtrip_without_overwrite(tmp_path):
    p=tmp_path/'checkpoint.pt';payload=dict(model=dict(weight=torch.tensor([1.,2.])),optimizer=dict(step=1))
    sha=save_checkpoint(p,payload)
    assert len(sha)==64
    assert_tree_equal(payload,torch.load(p,weights_only=True))
    with pytest.raises(FileExistsError):save_checkpoint(p,payload)
    assert not list(tmp_path.glob('*.partial-*'))
    with pytest.raises(ValueError,match='tensor'):assert_tree_equal(torch.tensor([1]),torch.tensor([2]))


def test_failed_later_tbptt_chunk_clears_accumulated_gradient(monkeypatch):
    # Fault-injection tests gradient hygiene only, never native acceptance.
    builder=ObservationBuilder(CardVocabulary.from_native(),ModelConfig(allow_incomplete_capture=True))
    first=builder.batch([public_frame(builder,tick=0)])
    observations=(first,replace(first,ticks=(5,)))
    empty=ActionSequence(torch.zeros(1,dtype=torch.long),*(torch.full((1,2),-1,dtype=torch.long) for _ in range(3)))
    sequence=ILSequence(observations,(empty,empty),torch.ones(2,1,dtype=torch.bool),True,'0'*64)

    class FailureModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.config=SimpleNamespace(allow_incomplete_capture=False)
        def initial_state(self,batch):return RecurrentState(torch.zeros(batch,1),torch.zeros(batch,1))
        def forward(self,obs,state,**kwargs):
            if obs.ticks==(5,):
                assert self.weight.grad is not None  # First chunk really backpropagated.
                raise ValueError('injected second chunk failure')
            return SimpleNamespace(logp=-self.weight[None],next_state=state)
    model=FailureModel();optimizer=torch.optim.Adam(model.parameters())
    monkeypatch.setattr('training_r0.il_learning.require_batch',lambda *args:True)
    with pytest.raises(ValueError,match='second chunk'):update_imitation_sequence(model,optimizer,sequence,tbptt_steps=1)
    assert model.weight.grad is None and model.weight.item()==1. and not optimizer.state
