"""Protocol fragmentation, size/magic/EOF/error guards, and no automatic mutation retry."""
import struct,json
from unittest.mock import patch
import msgpack
import binary_client
class Socket:
    def __init__(self,data):self.data=data;self.sent=0;self.closed=False
    def setsockopt(self,*args):pass
    def sendall(self,data):self.sent+=1
    def recv(self,n):chunk=self.data[:min(3,n)];self.data=self.data[len(chunk):];return chunk
    def close(self):self.closed=True
def frame(value):
    data=msgpack.packb(value,use_bin_type=True)
    return b'CRB1'+struct.pack('!I',len(data))+data
valid={'ok':True,'text':'皇室战争','numbers':[-9999999999,-33,-1,0,127,128,65536,1.25],'flags':[True,False,None]}
tests=[(frame(valid),True),(b'BAD!'+struct.pack('!I',1)+b'\x00',False),
       (b'CRB1'+struct.pack('!I',65*1024*1024),False),
       (b'CRB1'+struct.pack('!I',20)+b'\x80',False),(frame({'ok':False,'error':'test'}),False)]
for data,success in tests:
    sock=Socket(data)
    with patch.object(binary_client.socket,'create_connection',return_value=sock):
        c=binary_client.BinaryClient()
        try:
            result=c.request({'op':'resident_batch','entries':[]})
            assert success and result==valid
        except (ValueError,RuntimeError,ConnectionError):assert not success
        finally:c.close()
    assert sock.sent==1 and sock.closed
print(json.dumps(dict(passed=True,tests=len(tests),mutations_never_retried=True)))
