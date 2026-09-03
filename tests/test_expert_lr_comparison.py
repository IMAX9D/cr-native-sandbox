from pathlib import Path
import sys
import unittest
from scripts.finish_expert_lr_comparison import command_for_arm, require_paused


class ComparisonTests(unittest.TestCase):
    def test_command_only_changes_arm_rate_and_runtime_limit(self):
        original = [sys.executable,'-u','-m','expert_v1.training_v1','--run-id','control',
                    '--output-root','old','--learning-rate','0.0001','--epochs','20',
                    '--batch-size','32','--stop-after-epoch','2','--stop-at-step','12']
        result = command_for_arm(original,Path('/experiment/candidate'),5e-5,157674)
        self.assertNotIn('--stop-after-epoch',result)
        self.assertEqual(result[result.index('--stop-at-step')+1],'157674')
        self.assertEqual(result[result.index('--epochs')+1],'20')
        self.assertEqual(result[result.index('--batch-size')+1],'32')
        self.assertEqual(original[-1],'12')
    def test_success_requires_actual_saved_target(self):
        good = {'status':'paused','global_step':157674,'checkpoint':{'status':'saved'}}
        require_paused(good,157674)
        for change in ({'status':'failed'},{'global_step':157675},{'checkpoint':{}}):
            with self.assertRaises(RuntimeError): require_paused({**good,**change},157674)
