import contextlib
import io
import tempfile
from pathlib import Path
import unittest
import torch
from policy_v1.smoke import create_fixture
from hokoff_model.decision_data import prepare,DecisionWindows,collate_decisions
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
from hokoff_model.metrics import bc_loss
from hokoff_model.model import no_grad_burn_in


class BurnAMPTests(unittest.TestCase):
    def exercise(self,device,dtype):
        torch.set_num_threads(1);torch.manual_seed(18)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create_fixture(root/'data',steps=81)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(root/'data',root/'cache',sampling='fixed',allow_smoke=True,auxiliary_frame_window=3)
            ds=DecisionWindows(root/'data',root/'cache','validation',sampling='fixed',
                               decision_period=4,targets=4,frame_window=3,history_length=4)
            batch={k:v.to(device) for k,v in collate_decisions([ds[1],ds[2]]).items()};ds.close()
            self.assertTrue((batch['frame_mask']&~batch['loss_mask']).any())
            features=[[0.]*20]+[[i/12.]*10+[1.]*10 for i in range(1,12)]
            model=FixedPolicy(FixedConfig(12,4,width=16,hidden_size=32,history_length=4,
                                          spatial_skip_channels=4,combat_features=features)).to(device)
            with torch.no_grad():model.entity[0].weight[:,-20:].zero_()
            names=['entity.0.weight','grid.0.weight','scene.0.weight','history_summary.0.weight','lstm.weight_ih_l0']
            gradients=[]
            # The CPU oneDNN build in some environments lacks BF16 RNN backward.
            with torch.backends.mkldnn.flags(enabled=False):
                for cached in (True,False):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device,dtype=dtype,cache_enabled=cached):
                        loss,_=bc_loss(model(batch),batch,timing_positive_weight=32)
                    loss.backward()
                    captured={}
                    for name,p in model.named_parameters():
                        if name in names:
                            self.assertIsNotNone(p.grad,name)
                            self.assertTrue(torch.isfinite(p.grad).all(),name)
                            self.assertGreater(float(p.grad.abs().sum()),0,name)
                            captured[name]=p.grad.detach().clone()
                    gradients.append(captured)
            for name in names:torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=1e-5,atol=1e-6)
            self.assertGreater(float(model.entity[0].weight.grad[:,-20:].abs().sum()),0)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            optimizer.step()
            self.assertGreater(float(model.entity[0].weight[:,-20:].detach().abs().sum()),0)

    def test_cpu_autocast_encoder_gradients_and_new_feature_update(self):
        self.exercise('cpu',torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is unavailable')
    def test_cuda_fp16_encoder_gradients_and_new_feature_update(self):
        self.exercise('cuda',torch.float16)

    def test_cache_and_grad_mode_restored_on_exception(self):
        previous=torch.is_autocast_cache_enabled()
        try:
            for setting in (True,False):
                torch.set_autocast_cache_enabled(setting)
                with self.assertRaisesRegex(RuntimeError,'intentional'):
                    with no_grad_burn_in():
                        self.assertFalse(torch.is_grad_enabled())
                        self.assertFalse(torch.is_autocast_cache_enabled())
                        raise RuntimeError('intentional')
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(torch.is_autocast_cache_enabled(),setting)
        finally:torch.set_autocast_cache_enabled(previous)

if __name__=='__main__':unittest.main()
