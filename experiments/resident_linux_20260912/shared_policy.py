"""Loopback typed-tensor inference RPC. No pickle and no implicit RNN state."""
import argparse,concurrent.futures,json,math,queue,signal,socket,struct,sys,threading,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
DTYPES={'float32':np.float32,'int64':np.int64,'bool':np.bool_}
def upload_grouped(values,device='cuda'):
    """Three aligned H2D blocks instead of one transfer per small tensor."""
    groups={};result={}
    for key,value in values.items():groups.setdefault(value.dtype,[]).append((key,value))
    for dtype,items in groups.items():
        chunks=[];layout=[];offset=0
        for key,value in items:
            flat=value.contiguous().reshape(-1);length=flat.numel();chunks.append(flat)
            layout.append((key,value.shape,offset,length));offset+=length
            alignment=512//value.element_size();pad=(-offset)%alignment
            if pad:chunks.append(value.new_zeros((pad,)));offset+=pad
        block=torch.cat(chunks).to(device)
        for key,shape,start,length in layout:result[key]=block[start:start+length].reshape(shape)
    return result
def read_exact(s,n):
    out=bytearray()
    while len(out)<n:
        b=s.recv(n-len(out))
        if not b:raise EOFError('inference connection closed')
        out.extend(b)
    return bytes(out)
def send(s,meta,tensors):
    chunks=[];layout=[];total=0
    for key,t in tensors.items():
        a=t.detach().cpu().contiguous().numpy();dtype=str(a.dtype)
        if dtype not in DTYPES:raise ValueError('unsupported dtype')
        b=a.tobytes();layout.append([key,dtype,list(a.shape),len(b)]);chunks.append(b);total+=len(b)
    if total>32*1024*1024:raise ValueError('payload cap')
    h=json.dumps(dict(meta=meta,tensors=layout,bytes=total),separators=(',',':')).encode()
    if len(h)>65536:raise ValueError('header cap')
    s.sendall(struct.pack('!I',len(h))+h+b''.join(chunks))
def receive(s):
    size=struct.unpack('!I',read_exact(s,4))[0]
    if not 0<size<=65536:raise ValueError('header cap')
    h=json.loads(read_exact(s,size));layout=h['tensors'];total=h['bytes']
    if not isinstance(layout,list) or len(layout)>48 or not 0<=total<=32*1024*1024:raise ValueError('payload cap')
    expected=0;names=set()
    for key,dtype,shape,n in layout:
        if key in names or not isinstance(key,str) or len(key)>80 or dtype not in DTYPES:raise ValueError('field')
        if len(shape)>6 or any(type(x) is not int or x<0 or x>8192 for x in shape):raise ValueError('shape')
        if math.prod(shape)*np.dtype(DTYPES[dtype]).itemsize!=n:raise ValueError('shape/bytes')
        expected+=n;names.add(key)
    if expected!=total:raise ValueError('length')
    body=read_exact(s,total);out={};offset=0
    for key,dtype,shape,n in layout:
        out[key]=torch.from_numpy(np.frombuffer(body[offset:offset+n],dtype=DTYPES[dtype]).copy().reshape(shape));offset+=n
    return h['meta'],out

class RemotePolicy:
    def __init__(self,config,sha,port):
        self.config=config;self.sha=sha;self.seq=0
        deadline=time.monotonic()+30
        while True:
            try:self.socket=socket.create_connection(('127.0.0.1',port),timeout=5);break
            except ConnectionRefusedError:
                if time.monotonic()>=deadline:raise
                time.sleep(.1)
        self.socket.settimeout(90)
        self.socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    def initial_hidden(self,batch_size,*,device=None):
        return tuple(torch.zeros(1,batch_size,self.config.hidden_size,device='cpu') for _ in range(2))
    def forward_stream(self,batch,state):
        self.seq+=1
        send(self.socket,dict(seq=self.seq,model=self.sha),dict(batch,hidden_h=state[0],hidden_c=state[1]))
        meta,out=receive(self.socket)
        if meta.get('error'):raise RuntimeError(meta['error'])
        if meta.get('seq')!=self.seq or meta.get('model')!=self.sha:raise RuntimeError('inference identity mismatch')
        h=out.pop('hidden_h');c=out.pop('hidden_c');return out,(h,c)
    def close(self):self.socket.close()

def main():
    from hokoff_model.match_agent import load_release
    from policy_test import collate
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=41580);p.add_argument('--checkpoint',required=True)
    p.add_argument('--contract',required=True);p.add_argument('--wait-ms',type=float,default=1.)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,identity=load_release(a.checkpoint,contract_path=a.contract,device='cuda');sha=identity['checkpoint_sha256']
    q=queue.Queue(maxsize=64);stop=threading.Event();counts=dict(calls=0,requests=0,views=0,compute_seconds=0.)
    def worker():
        while not stop.is_set():
            try:first=q.get(timeout=.2)
            except queue.Empty:continue
            items=[first];until=time.perf_counter()+a.wait_ms/1000
            while len(items)<16:
                left=until-time.perf_counter()
                if left<=0:break
                try:items.append(q.get(timeout=left))
                except queue.Empty:break
            try:
                started=time.perf_counter();sizes=[x[1]['frame_mask'].shape[0] for x in items]
                batch=collate([x[1] for x in items])
                batch.update(hidden_h=torch.cat([x[2][0] for x in items],1),hidden_c=torch.cat([x[2][1] for x in items],1))
                batch=upload_grouped(batch);hidden=(batch.pop('hidden_h'),batch.pop('hidden_c'))
                with torch.inference_mode():output,h=model.forward_stream(batch,hidden)
                if not bool(torch.stack([torch.isfinite(v).all() for v in [*output.values(),*h]]).all()):raise FloatingPointError('nonfinite shared output')
                views=sum(sizes);shapes={k:v.shape[1:] for k,v in output.items()};widths=[v[0].numel() for v in output.values()]
                rnn_width=h[0].shape[0]*h[0].shape[2]
                packed=torch.cat([v.reshape(views,-1) for v in output.values()]+
                                 [v.transpose(0,1).reshape(views,-1) for v in h],1).cpu()
                pieces=packed.split(widths+[rnn_width,rnn_width],1)
                output={k:v.reshape(views,*shapes[k]) for k,v in zip(output,pieces)}
                h=tuple(v.reshape(views,1,model.config.hidden_size).transpose(0,1) for v in pieces[-2:])
                counts['compute_seconds']+=time.perf_counter()-started;counts['calls']+=1;counts['requests']+=len(items);counts['views']+=sum(sizes)
                offset=0
                for item,n in zip(items,sizes):
                    item[3].set_result({**{k:v[offset:offset+n].clone() for k,v in output.items()},
                                        'hidden_h':h[0][:,offset:offset+n].clone(),'hidden_c':h[1][:,offset:offset+n].clone()});offset+=n
                if counts['calls']%100==0:(ROOT/'shared-policy-stats.json').write_text(json.dumps(counts))
            except BaseException as e:
                for item in items:item[3].set_exception(e)
    def connection(s):
        with s:
            s.settimeout(120);s.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            try:
                while not stop.is_set():
                    meta,values=receive(s)
                    if meta.get('model')!=sha:
                        send(s,dict(seq=meta.get('seq'),model=sha,error='model identity mismatch'),{});return
                    h=values.pop('hidden_h');c=values.pop('hidden_c');b=values['frame_mask'].shape[0]
                    if not 1<=b<=8 or tuple(h.shape)!=(1,b,model.config.hidden_size) or c.shape!=h.shape:raise ValueError('batch/RNN shape')
                    future=concurrent.futures.Future();q.put((meta,values,(h,c),future),timeout=10)
                    try:result=future.result(timeout=90);send(s,dict(seq=meta['seq'],model=sha),result)
                    except Exception as e:send(s,dict(seq=meta['seq'],model=sha,error=str(e)),{});break
            except (OSError,EOFError,ValueError,KeyError):return
    t=threading.Thread(target=worker);t.start()
    def end(signum,frame):raise SystemExit()
    signal.signal(signal.SIGTERM,end)
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('127.0.0.1',a.port));listener.listen(32);listener.settimeout(1)
            (ROOT/'shared-policy-ready.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),port=a.port,identity=identity)))
            print('shared policy ready',flush=True)
            while True:
                try:s,_=listener.accept()
                except socket.timeout:continue
                threading.Thread(target=connection,args=(s,),daemon=True).start()
    finally:stop.set();t.join(timeout=10);(ROOT/'shared-policy-stats.json').write_text(json.dumps(counts))
if __name__=='__main__':main()
