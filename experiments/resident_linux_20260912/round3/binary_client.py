"""Bounded persistent MessagePack response reader; never retries uncertain mutations."""
import json,socket,struct,threading
import msgpack

class BinaryClient:
    def __init__(self,host='127.0.0.1',port=26431,timeout=30,**kwargs):
        self.host,self.port,self.timeout=host,port,timeout;self.sock=None;self.lock=threading.Lock()
        self.bytes_received=0;self.calls=0
    def close(self):
        if self.sock is not None:self.sock.close();self.sock=None
    def _read(self,n):
        parts=[]
        while n:
            b=self.sock.recv(min(n,65536))
            if not b:raise ConnectionError('truncated binary response; mutation not retried')
            parts.append(b);n-=len(b)
        return b''.join(parts)
    def request(self,payload):
        with self.lock:
            try:
                if self.sock is None:
                    self.sock=socket.create_connection((self.host,self.port),self.timeout);self.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                body=json.dumps({**payload,'response_wire':'msgpack-v1'},separators=(',',':'),allow_nan=False).encode()+b'\n'
                if len(body)>32*1024*1024:raise ValueError('request too large')
                self.sock.sendall(body)
                header=self._read(8)
                if header[:4]!=b'CRB1':raise ValueError('binary protocol mismatch')
                size=struct.unpack('!I',header[4:])[0]
                if not 0<size<=64*1024*1024:raise ValueError('binary frame limit')
                raw=self._read(size)
                result=msgpack.unpackb(raw,raw=False,strict_map_key=True,max_str_len=32*1024*1024,max_bin_len=0,max_array_len=262144,max_map_len=65536,max_ext_len=0)
                if not isinstance(result,dict) or result.get('ok') is not True:raise RuntimeError('native binary operation failed: '+str(result))
                self.bytes_received+=size+8;self.calls+=1
                return result
            except BaseException:self.close();raise
