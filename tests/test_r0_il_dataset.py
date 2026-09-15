from dataclasses import replace
import pytest
import torch
from training_r0.il_dataset import ILSequence,save_sequence,load_sequence
from training_r0.il_learning import update_imitation_sequence
from training_r0.actions import ActionSequence
from training_r0.config import ModelConfig
from training_r0.catalog import CardVocabulary
from training_r0.observation import ObservationBuilder
from training_r0.model import R0Policy
from training_r0.synthetic import rich_public_frame


def sequence():
    config=ModelConfig(width=32,layers=1,heads=4,hidden=64,spatial_channels=16,learned_scatter_channels=8,native_graph_layers=1,allow_incomplete_capture=True)
    builder=ObservationBuilder(CardVocabulary.from_native(),config)
    batch=builder.batch([rich_public_frame(builder)])
    observations=tuple(replace(batch,ticks=(i*5,)) for i in range(3))
    actions=tuple(ActionSequence(torch.zeros(1,dtype=torch.long),*(torch.full((1,2),-1,dtype=torch.long) for _ in range(3))) for _ in observations)
    return ILSequence(observations,actions,torch.ones((3,1),dtype=torch.bool),True,'0'*64),builder,config


def test_tensor_columns_roundtrip_without_pickle(tmp_path):
    s,_,_=sequence();p=tmp_path/'sequence.npz';sha=save_sequence(s,p);r=load_sequence(p,expected_sha256=sha)
    assert torch.equal(r.observations[0].semantic.elite.child,s.observations[0].semantic.elite.child)
    assert r.observations[1].ticks==(5,) and r.starts_episode
    assert torch.equal(r.actions[1].candidate_uid,s.actions[1].candidate_uid)
    with pytest.raises(ValueError,match='hash mismatch'):load_sequence(p,expected_sha256='1'*64)


def test_discontinuous_sequence_rejected():
    s,_,_=sequence();s=replace(s,observations=(s.observations[0],replace(s.observations[1],ticks=(9,)),s.observations[2]))
    with pytest.raises(ValueError,match='contiguous'):s.validate()


def test_diagnostic_data_cannot_be_called_accepted_training():
    s,builder,config=sequence();model=R0Policy(builder.vocabulary,config)
    with pytest.raises(ValueError,match='diagnostic model'):update_imitation_sequence(model,torch.optim.Adam(model.parameters()),s)
