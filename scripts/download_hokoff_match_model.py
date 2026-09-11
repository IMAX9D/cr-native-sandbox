"""Fetch the explicitly pinned public BC model; never execute downloaded code."""
from pathlib import Path
import argparse
import hashlib
import os
import urllib.request

NAME='hokoff-bc-step1037042.pt'
SIZE=19285652
SHA='ff1fe42b76dbec2f68f840fd289aa5dda15357d5ab7ac47d3a1db1cbe6f02eb9'
URL='https://github.com/IMAX9D/cr-native-sandbox/releases/download/bc-weights-step1037042-20260911/'+NAME


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'models'/NAME)
    args=p.parse_args();target=args.output.resolve()
    if target.exists():
        if target.stat().st_size!=SIZE or digest(target)!=SHA:raise ValueError('existing model differs; refusing overwrite')
        print('Pinned model already verified:',target);return
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+'.download-'+str(os.getpid()))
    try:
        with urllib.request.urlopen(URL,timeout=60) as response,temp.open('xb') as output:
            total=0
            while True:
                data=response.read(1024*1024)
                if not data:break
                total+=len(data)
                if total>SIZE:raise ValueError('download exceeds expected model size')
                output.write(data)
        if temp.stat().st_size!=SIZE or digest(temp)!=SHA:raise ValueError('model download checksum mismatch')
        if target.exists():raise FileExistsError('model destination appeared during download')
        temp.rename(target)
    finally:
        if temp.exists():temp.unlink()
    print('Model SHA256 verified:',target)


if __name__=='__main__':main()
