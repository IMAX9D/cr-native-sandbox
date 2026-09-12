"""Restore canary heads/value on the exact source model; never publish or overwrite it."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from learner_canary import BASE,head,digest
import torch
from hokoff_model.match_agent import load_release
torch.set_num_threads(1)
folder=ROOT/'learner-audited'
paths=sorted(folder.glob('update-*.pt'));assert paths
model,_,identity=load_release(BASE/'model319051/inference.pt',BASE/'model319051/encoder-contract.json',device='cpu')
before=digest(model.state_dict());checks=[]
def finite(x):
    if isinstance(x,torch.Tensor):
        if x.is_floating_point():assert bool(torch.isfinite(x).all())
    elif isinstance(x,dict):
        for y in x.values():finite(y)
    elif isinstance(x,(list,tuple)):
        for y in x:finite(y)
for p in paths:
    saved=torch.load(p,map_location='cpu',weights_only=True);finite(saved)
    assert saved['kind']=='isolated_head_ppo_canary_v1'
    assert saved['source_identity']['checkpoint_sha256']==identity['checkpoint_sha256']
    assert saved['frozen_backbone_sha256']==digest(model.state_dict(),lambda k:not head(k))
    assert set(saved['actor_heads'])=={k for k in model.state_dict() if head(k)}
    state=model.state_dict();state.update(saved['actor_heads']);model.load_state_dict(state,strict=True)
    assert saved['frozen_backbone_sha256']==digest(model.state_dict(),lambda k:not head(k))
    checks.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),version=saved['version']))
assert digest(model.state_dict())!=before
result=dict(passed=True,source_unchanged=hashlib.sha256((BASE/'model319051/inference.pt').read_bytes()).hexdigest()==identity['checkpoint_sha256'],checkpoints=checks)
(ROOT/'checkpoint-verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
