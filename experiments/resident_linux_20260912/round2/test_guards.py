"""CPU-only negative tests: special fast path must not accept general training inputs."""
import sys, json
from pathlib import Path
sys.path.insert(0,'D:/Deepseek/outputs/hokoff-match-box-20260911/worktree')
import torch
from hokoff_model.match_agent import load_release
from dense_forward import install
from optimized_agent import MatchAgent

torch.set_num_threads(2)
root=Path('D:/Deepseek/outputs/resident-linux-20260912/model319051')
model,encoder,_=load_release(root/'inference.pt',root/'encoder-contract.json',device='cpu')
before={k:v.clone() for k,v in model.state_dict().items()}
install(model)
cases=[{'frame_mask':torch.zeros(1,1,dtype=torch.bool)},
       {'frame_mask':torch.ones(1,2,dtype=torch.bool)},
       {'frame_mask':torch.ones(1,1,dtype=torch.bool),'loss_mask':torch.ones(1,1,dtype=torch.bool)}]
for value in (-1.,float('nan'),float('inf')):
    b={'frame_mask':torch.ones(1,1,dtype=torch.bool)}
    for key in ('history_card','history_position','history_age','history_mask','history_known'):
        b[key]=torch.zeros(1,1,2,4)
    b['history_age'].fill_(value);cases.append(b)
passed=0
for b in cases:
    try:model.forward_stream(b)
    except ValueError:passed+=1
    else:raise AssertionError('invalid online input accepted')
model.train()
try:model.forward_stream({'frame_mask':torch.ones(1,1,dtype=torch.bool)})
except ValueError:passed+=1
else:raise AssertionError('training mode accepted')
model.eval();agent=MatchAgent(model,encoder)
raw={'tick':12};agent._prepared_frame=(raw,object());agent.reset(12)
try:agent.finish(({},None,12),{},(),raw,[],None,batch_validated=True)
except ValueError:passed+=1
else:raise AssertionError('stale frame accepted after reset')
assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
print(json.dumps(dict(negative_guards=passed,weights_unchanged=True)))
