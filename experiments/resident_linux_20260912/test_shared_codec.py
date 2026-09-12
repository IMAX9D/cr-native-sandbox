import json,socket,struct,threading
import torch
from shared_policy import send,receive,upload_grouped
torch.set_num_threads(1)
values={'grid':torch.randn(8,1,8,32,18),'cards':torch.arange(16).reshape(8,1,2),
        'mask':torch.tensor([True,False]),'empty':torch.empty(8,1,0),'hidden_h':torch.randn(1,8,512)}
packed=upload_grouped(values,device='cpu')
for k in values:assert torch.equal(values[k],packed[k]),k
a,b=socket.socketpair()
try:
    t=threading.Thread(target=send,args=(a,{'seq':1,'model':'test'},values));t.start()
    meta,out=receive(b);t.join(timeout=5)
    assert meta=={'seq':1,'model':'test'} and out.keys()==values.keys()
    for k in values:assert torch.equal(values[k],out[k]),k
finally:a.close();b.close()
for header in ({'meta':{},'bytes':0,'tensors':[['x','object',[1],0]]},
               {'meta':{},'bytes':0,'tensors':[['x','float32',[999999999],0]]}):
    a,b=socket.socketpair()
    try:
        raw=json.dumps(header).encode();a.sendall(struct.pack('!I',len(raw))+raw)
        try:receive(b)
        except ValueError:pass
        else:raise AssertionError('malformed tensor accepted')
    finally:a.close();b.close()
print('typed tensor roundtrip and bounds checks passed')
