"""Action-independent observation schedules plus isolated action-only examples."""
import numpy as np

CONTRACT = 'independent_observations_with_action_aux_v2'


def independent_indices(offsets, scalars, delta, exposure, actions, valid, max_delay,
                        *, seed=42, auxiliary=True, frame_window=17):
    if not 1 <= max_delay <= 32767 or frame_window < 1:
        raise ValueError('invalid observation or history limit')
    n=len(actions)
    if len(offsets)<2 or offsets[0]!=0 or offsets[-1]!=n or np.any(np.diff(offsets)<=0):
        raise ValueError('invalid sequence offsets')
    if not np.all(delta==1) or not np.all(exposure==1):
        raise ValueError('requires contiguous one-tick source')
    rng=np.random.default_rng(seed)
    fields={k:[] for k in ('rows','ticks','elapsed','delay','delay_mask','supervision')}
    packed_offsets=[0];owners=[];roles=[]

    def append(rows,ticks,elapsed,delay,known,supervision,owner,role):
        for k,v in zip(fields,(rows,ticks,elapsed,delay,known,supervision)):
            fields[k].append(np.asarray(v))
        packed_offsets.append(packed_offsets[-1]+len(rows));owners.append(owner);roles.append(role)

    for owner,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        lo,hi=int(lo),int(hi)
        recorded=np.rint(scalars[lo:hi,0]*6000).astype(np.int64)
        initial=int(recorded[0]); clock=initial+np.arange(hi-lo)
        if not 0 <= initial < 6000 or not np.array_equal(recorded,np.minimum(clock,6000)):
            raise ValueError('source clock is not unambiguously contiguous')
        v=np.asarray(valid[lo:hi],dtype=bool)
        starts=np.flatnonzero(v & ~np.r_[False,v[:-1]])
        ends=np.flatnonzero(v & ~np.r_[v[1:],False])+1
        for start,end in zip(starts,ends):
            # Draw all scheduling randomness before reading any action label.
            # 75% maximum waits, 25% uniform 1..K: mean 7.125 ticks for K=8.
            count=int(end-start)
            draws=rng.random((count,2))
            gaps=(draws[:,0]*max_delay).astype(np.int64)+1
            gaps[draws[:,1]<.75]=max_delay
            rows=start+np.r_[0,np.cumsum(gaps)]
            rows=rows[rows<end]
            elapsed=np.r_[0,np.diff(rows)]
            events=np.flatnonzero(np.asarray(actions[lo+start:lo+end],dtype=bool))+start
            next_index=np.searchsorted(events,rows,side='right')
            next_event=np.r_[events,end+max_delay][next_index]
            distance=next_event-rows
            delay=np.minimum(distance,max_delay)
            known=(rows+max_delay<end) | ((next_index<len(events)) & (distance<=max_delay))
            append(rows+lo,rows+initial,elapsed,delay,known,np.ones(len(rows),dtype=np.uint8),owner,0)
            if auxiliary:
                # Keep every missed action, with history from the same independent
                # schedule. Only the final action's non-timing heads are supervised.
                for event in np.setdiff1d(events,rows,assume_unique=True):
                    pos=int(np.searchsorted(rows,event))
                    begin=max(0,pos-frame_window+1)
                    history=np.r_[rows[begin:pos],event]
                    dt=np.r_[elapsed[begin:pos],event-rows[pos-1] if pos else 0]
                    label=np.zeros(len(history),dtype=np.uint8);label[-1]=2
                    append(history+lo,history+initial,dt,np.zeros(len(history),dtype=np.int64),
                           np.zeros(len(history),dtype=bool),label,owner,1)
    out={k:np.concatenate(v) if v else np.empty(0,dtype=np.int64) for k,v in fields.items()}
    for k in ('rows','ticks','elapsed','delay'):out[k]=out[k].astype(np.int64)
    out['delay_mask']=out['delay_mask'].astype(bool)
    out['supervision']=out['supervision'].astype(np.uint8)
    out.update(offsets=np.array(packed_offsets,dtype=np.int64),owners=np.array(owners,dtype=np.int64),
               roles=np.array(roles,dtype=np.uint8))
    return out
