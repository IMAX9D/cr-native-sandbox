"""A single frozen opponent, rolling game results and numbered promotions."""
from copy import deepcopy
from dataclasses import asdict
import torch
from .ppo import state_digest
from .ppo_actions import HEAD_NAMES,action_heads

OPPONENT_MODE='fixed_snapshot_until_win_rate_promotion_v1'


def resolve_win_rate_window(requested,contract=None):
    saved=None if contract is None else contract['promotion_window']
    value=(100 if saved is None else saved) if requested is None else requested
    if not isinstance(value,int) or value<2 or value%2:
        raise ValueError('win rate window must be an even integer >=2 for balanced sides')
    if saved is not None and value!=saved:raise ValueError('resume win rate window differs')
    return value


def against_opponent_logits(output,features,opponent,learner_side):
    """Reuse the identical frozen encoder/LSTM; replace only opponent heads.

    Each side still has its own recurrent sequence in the two-row batch.
    FrozenOpponent checks body equality when loading or promoting a snapshot.
    """
    logits={k:output[k][:,0] for k in ('timing','kind','card','position','ability')}
    if opponent is None:return logits
    other=1-learner_side
    opponent_logits=action_heads(opponent,{k:v[other:other+1] for k,v in features.items()})
    return {k:torch.cat((v[:1],opponent_logits[k]),0) if learner_side==0
            else torch.cat((opponent_logits[k],v[1:]),0) for k,v in logits.items()}


class FrozenOpponent:
    threshold=.95

    def __init__(self,base,window=100):
        self.window=resolve_win_rate_window(window)
        self.model=deepcopy(base).eval().requires_grad_(False)
        self.number=1;self.games=0;self.results=[];self.promotions=[]
        self.model_hash=state_digest(self.model)

    @property
    def name(self):return 'base' if self.number==1 else f'beat_{self.number-1}'

    def check_body(self,actor):
        other=actor.state_dict()
        for name,value in self.model.state_dict().items():
            if name.split('.')[0] not in HEAD_NAMES and not torch.equal(value,other[name]):
                raise ValueError('opponent and learner frozen body differ: '+name)

    def assert_unchanged(self):
        if state_digest(self.model)!=self.model_hash:raise RuntimeError('frozen opponent changed without promotion')
        if any(p.requires_grad for p in self.model.parameters()):raise RuntimeError('opponent must have no trainable parameters')

    def record(self,episode,*,policy_step):
        terminal=episode['terminal'];side=episode['learner_side']
        if side not in (0,1) or not terminal.get('terminated') or terminal.get('truncated'):
            raise ValueError('win rate requires resolved full games and valid learner side')
        outcome=terminal['outcome']
        if outcome not in ('side0_win','side1_win','draw'):
            raise ValueError('unrecognized terminal outcome: '+str(outcome))
        if episode.get('opponent_number',self.number)!=self.number:
            raise ValueError('game belongs to a different opponent')
        if self.results and episode['episode']<=self.results[-1]['episode']:
            raise ValueError('game already counted or episode order changed')
        result='draw' if outcome=='draw' else 'win' if outcome==f'side{side}_win' else 'loss'
        self.results.append(dict(episode=episode['episode'],learner_side=side,result=result,policy_step=policy_step))
        self.results=self.results[-self.window:];self.games+=1

    def statistics(self):
        wins=sum(r['result']=='win' for r in self.results)
        draws=sum(r['result']=='draw' for r in self.results)
        sides=[sum(r['learner_side']==s for r in self.results) for s in (0,1)]
        n=len(self.results)
        return dict(opponent_number=self.number,opponent_name=self.name,opponents_beaten=len(self.promotions),
                    games_vs_opponent=self.games,win_rate_games=n,win_rate_window=self.window,
                    wins=wins,losses=n-wins-draws,draws=draws,win_rate=wins/n if n else None,
                    side0_games=sides[0],side1_games=sides[1],promotion_threshold=self.threshold,
                    promotion_ready=n==self.window and sides[0]==sides[1] and wins/n>self.threshold)

    def maybe_promote(self,actor,*,policy_step,iteration):
        stats=self.statistics()
        if not stats['promotion_ready']:return None
        self.assert_unchanged();self.check_body(actor)
        # Snapshot the policy that played this batch, before its next PPO update.
        self.model.load_state_dict(actor.state_dict());self.model.eval().requires_grad_(False)
        self.model_hash=state_digest(self.model)
        record=dict(**stats,filename=f'beat_{self.number}.pt',snapshot_step=policy_step,
                    iteration=iteration,model_sha256=self.model_hash,
                    first_episode=self.results[0]['episode'],last_episode=self.results[-1]['episode'],
                    evidence='rolling_training_games',results=deepcopy(self.results))
        self.promotions.append(record);self.number+=1;self.games=0;self.results=[]
        return deepcopy(record)

    def state_dict(self):
        self.assert_unchanged()
        return dict(window=self.window,threshold=self.threshold,number=self.number,games=self.games,
                    results=deepcopy(self.results),promotions=deepcopy(self.promotions),
                    model={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
                    model_sha256=self.model_hash)

    def load_state_dict(self,saved):
        if saved['window']!=self.window or saved['threshold']!=self.threshold:
            raise ValueError('opponent promotion settings changed')
        # Check body against the original IL before admitting saved opponent heads.
        original=deepcopy(self.model)
        self.model.load_state_dict(saved['model']);self.check_body(original)
        self.model_hash=saved['model_sha256'];self.assert_unchanged()
        self.number=saved['number'];self.games=saved['games']
        self.results=deepcopy(saved['results']);self.promotions=deepcopy(saved['promotions'])
        if self.number!=len(self.promotions)+1 or len(self.results)>self.window or self.games<len(self.results):
            raise ValueError('invalid saved opponent history')
        if self.promotions and self.promotions[-1]['model_sha256']!=self.model_hash:
            raise ValueError('saved opponent does not match its promotion')

    def export_latest(self,output,*,source_contract):
        """Call after last.pt commits; resume can rebuild a missing latest export."""
        if not self.promotions:return
        from policy_v1.train import load_checkpoint
        record=self.promotions[-1];path=output/record['filename']
        if path.exists():
            saved=load_checkpoint(path)
            if saved.get('promotion')!=record or any(not torch.equal(v.detach().cpu(),saved['model'][k].cpu())
                                                    for k,v in self.model.state_dict().items()):
                raise ValueError('existing beat snapshot differs: '+str(path))
            return
        value=dict(kind='hokoff_fixed_beat_snapshot_v1',config=asdict(self.model.config),
                   model={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
                   contract=deepcopy(source_contract),step=record['snapshot_step'],promotion=deepcopy(record))
        temp=path.with_suffix('.pt.tmp');torch.save(value,temp);temp.replace(path)
