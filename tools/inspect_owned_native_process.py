"""Bounded read-only /proc inspection of an explicitly owned headless worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

EXPECTED='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    command=Path(f'/proc/{a.pid}/cmdline').read_bytes().split(b'\0')
    if b'royale.nativehost.JniHost' not in command or b'serve-direct' not in command or str(a.port).encode() not in command:
        raise ValueError('not the requested owned headless worker')
    maps=Path(f'/proc/{a.pid}/maps').read_text().splitlines();mapping=None
    for line in maps:
        parts=line.split()
        if len(parts)>=6 and parts[2]=='00000000' and parts[-1].endswith('/libg.so') and '/cr-native-direct-' in parts[-1]:mapping=parts;break
    if mapping is None:raise ValueError('no owned libg mapping')
    library=Path(mapping[-1]);sha=hashlib.sha256(library.read_bytes()).hexdigest()
    if sha!=EXPECTED:raise ValueError('unsupported native library')
    base=int(mapping[0].split('-')[0],16);fd=os.open(f'/proc/{a.pid}/mem',os.O_RDONLY)
    def bytes_at(pointer,size):
        b=os.pread(fd,size,pointer)
        if len(b)!=size:raise ValueError('incomplete memory read')
        return b
    def u64(pointer):return struct.unpack('<Q',bytes_at(pointer,8))[0]
    def i32(pointer):return struct.unpack('<i',bytes_at(pointer,4))[0]
    def words(pointer,size):return list(struct.unpack('<'+'i'*(size//4),bytes_at(pointer,size)))
    manager=u64(base+0x1a85978);state=u64(manager+0x20);battle=u64(state+0x90)
    registry=u64(u64(battle+0xa8)+8);collection=u64(registry+0x40)
    count=i32(collection+0x14);array=u64(collection+8);tick=i32(battle+0x60)
    if not 0<=count<=2048:raise ValueError('invalid registry')
    rows=[]
    managers=[]
    if count:
        world=u64(u64(array)+0x10)
        for label,obj in [('world',world),('collection',collection)]:
            vt=u64(obj)
            managers.append(dict(label=label,pointer=hex(obj),vtable=hex(vt-base),functions=[hex(u64(vt+i*8)-base) for i in range(16)]))
    for i in range(count):
        obj=u64(array+i*8);key=i32(obj+8)
        if not 3000000<=key<6000000:continue
        native_data=u64(obj+0x48)
        row=dict(key=key,object=hex(obj),native_data=hex(native_data),native_data_id=i32(native_data+0x40),
                 kind=i32(obj+0x30),side=i32(obj+0x78),object_words=words(obj,0x180),data_words=words(native_data,0x800),components=[])
        n=i32(obj+0x24);components=u64(obj+0x18)
        if not 0<=n<=32:raise ValueError('invalid components')
        for j in range(min(n,4)):
            component=u64(components+j*8)
            if not component:
                row['components'].append(dict(slot=j,present=False));continue
            vt=u64(component)
            row['components'].append(dict(slot=j,vtable_rva=hex(vt-base),owner_matches=u64(component+8)==obj,
                function_rvas=[hex(u64(vt+k*8)-base) for k in range(16)],words=words(component,[0x68,0x220,0x30,0x38][j])))
        rows.append(row)
        if len(rows)>=64:break
    if i32(battle+0x60)!=tick:raise ValueError('worker advanced during read')
    # Exact known component code range only; avoid unrelated process memory.
    code_start=0xfb2000;code_size=0xf000
    code=bytes_at(base+code_start,code_size).hex()
    character_code=bytes_at(base+0xf70000,0x10000).hex()
    candidate_vtables={hex(vt):[hex(u64(base+vt+i*8)-base) for i in range(32)] for vt in (0x1959758,)}
    os.close(fd);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(dict(pid=a.pid,port=a.port,libg_sha256=sha,base=hex(base),tick=tick,entities=rows,
        managers=managers,candidate_vtables=candidate_vtables,component_code=dict(start=code_start,hex=code),character_code=dict(start=0xf70000,hex=character_code)),f)
    print(json.dumps(dict(tick=tick,entities=len(rows),output=str(a.output))))


if __name__=='__main__':main()
