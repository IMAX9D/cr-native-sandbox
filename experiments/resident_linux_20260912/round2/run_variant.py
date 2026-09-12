"""Execute an isolated variant using the already certified frozen-policy driver."""
import os, sys
from pathlib import Path
BASE = Path('/root/autodl-tmp/resident-linux-20260912')
sys.path.insert(0, str(BASE))
sys.path.insert(0, '/root/autodl-tmp/bc-cloud-bench-20260911/core')
import policy_test
from fast_history import FastHistory
import optimized_agent

variant = os.environ.get('CR_OPT_VARIANT', 'baseline')
if variant not in ('baseline', 'history', 'state', 'combined', 'dense'):
    raise ValueError('unknown optimization variant')

if variant != 'baseline':
    Parent = optimized_agent.MatchAgent if variant in ('state', 'combined', 'dense') else policy_test.MatchAgent
    class Agent(Parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if variant in ('history', 'combined', 'dense'):
                self.history = FastHistory(self.model.config.history_length)
        def prepare(self, raw_state, decks, *, batch_device=None):
            # This cache is ONLY valid within this synchronous prepare group.
            # It is cleared by collate before any transitions are submitted.
            if variant in ('state', 'combined', 'dense'):
                key = id(raw_state)
                if key not in frame_cache:
                    frame_cache[key] = (raw_state, optimized_agent.normalize_native_state(raw_state))
                cached_raw, normalized = frame_cache[key]
                if cached_raw is not raw_state or normalized.tick != raw_state['tick']:
                    raise ValueError('frame identity changed during synchronous prepare')
                return super().prepare(raw_state, decks, batch_device=batch_device, normalized_state=normalized)
            return super().prepare(raw_state, decks, batch_device=batch_device)
    frame_cache = {}
    original_collate = policy_test.collate
    def collate(batches):
        try:
            return original_collate(batches)
        finally:
            frame_cache.clear()
    policy_test.MatchAgent = Agent
    policy_test.collate = collate

if variant == 'dense':
    from dense_forward import install
    import atexit, json
    original_load = policy_test.load_release
    def load_release(*args, **kwargs):
        model,encoder,identity=original_load(*args, **kwargs)
        install(model,verify=os.environ.get('CR_OPT_VERIFY_DENSE')=='1')
        atexit.register(lambda: print('DENSE_NUMERICS '+json.dumps(model.optimization_counters),flush=True))
        return model,encoder,identity
    policy_test.load_release=load_release

if __name__ == '__main__':
    if '--modes' in sys.argv and any(m!='packed' for m in sys.argv[sys.argv.index('--modes')+1].split(',')):
        raise ValueError('optimization comparison supports packed mode only')
    policy_test.main()
