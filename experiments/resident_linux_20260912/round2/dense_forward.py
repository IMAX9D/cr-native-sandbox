"""Online-only all-valid T=1 path. Same weights, no precision/decision changes."""
import copy, types
import torch
from hokoff_model.model import Policy


def install(model, *, verify=False):
    model.lstm.flatten_parameters()
    reference = copy.deepcopy(model).eval() if verify else None
    if reference is not None: reference.lstm.flatten_parameters()
    counters = {'calls':0, 'max_logit_error':0., 'max_hidden_error':0.}
    def forward(self, b, state=None, reset=None):
        # Retain explicit runtime guards. Do not use this path for padded training chunks.
        if self.training or b['frame_mask'].ndim != 2 or b['frame_mask'].shape[1] != 1:
            raise ValueError('dense online path requires eval T=1')
        if not bool(b['frame_mask'].all()) or 'loss_mask' in b:
            raise ValueError('dense online path refuses padding/supervision')
        expected=(*b['frame_mask'].shape,2,self.config.history_length)
        for key in ('history_card','history_position','history_age','history_mask','history_known'):
            if key not in b or tuple(b[key].shape)!=expected:raise ValueError('invalid online history: '+key)
        if not bool(torch.isfinite(b['history_age']).all()) or bool((b['history_age']<0).any()):
            raise ValueError('invalid online history age')
        original_state=state
        x=self.encode(b)
        if state is not None and reset is not None:
            state=tuple(s.masked_fill(reset[None,:,None],0) for s in state)
        # All rows contain one valid timestep: no pack/sort/CPU lengths transfer needed.
        recurrent,new_state=self.lstm(x,state)
        output=Policy.heads(self,recurrent,b)
        if self.config.spatial_skip_channels:
            B,T=b['frame_mask'].shape
            grid=b['grid'].reshape(B*T,self.config.grid_channels,32,18)
            if self.config.spatial_type_dim:
                grid=torch.cat((grid,self.spatial_type_grid(b)[:,0].to(grid)),1)
            context=output['context'].reshape(B*T,-1)
            hand=self.cards(b['hand_tokens'].reshape(B*T,4))
            correction=self.position_skip(grid,context,hand)
            output['position']=(output['position'].reshape(B*T,4,576)+correction.to(output['position'].dtype)).reshape(B,T,4,576)
        if reference is not None:
            with torch.inference_mode(): expected_output,expected_hidden=reference.forward_stream(b,original_state,reset=reset)
            for key,value in output.items():
                counters['max_logit_error']=max(counters['max_logit_error'],float((value-expected_output[key]).abs().max()))
                torch.testing.assert_close(value,expected_output[key],rtol=2e-4,atol=1e-4)
            for value,expected_value in zip(new_state,expected_hidden):
                counters['max_hidden_error']=max(counters['max_hidden_error'],float((value-expected_value).abs().max()))
                torch.testing.assert_close(value,expected_value,rtol=2e-4,atol=1e-4)
        counters['calls']+=1
        return output,new_state
    model.forward_stream=types.MethodType(forward,model)
    model.optimization_counters=counters
    return model
