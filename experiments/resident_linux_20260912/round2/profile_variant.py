"""Reuse the profiling phases with one explicitly selected inference variant."""
import os
import run_variant
import profile_pipeline
profile_pipeline.MatchAgent=run_variant.policy_test.MatchAgent
profile_pipeline.collate=run_variant.policy_test.collate
profile_pipeline.load_release=run_variant.policy_test.load_release
Client=profile_pipeline.Client
class SelectedClient(Client):
    def __init__(self,*args,**kwargs):
        kwargs['port']=int(os.environ.get('CR_OPT_PORT','41431'))
        super().__init__(*args,**kwargs)
profile_pipeline.Client=SelectedClient
if __name__=='__main__':profile_pipeline.main()
