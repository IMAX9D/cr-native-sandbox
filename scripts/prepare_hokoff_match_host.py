"""Install verified, source-bound open-source host binaries (no game assets)."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main():
    source=ROOT/'tools'/'hokoff-match-host'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='hokoff_match_host_overlay_v1':raise ValueError('unknown host manifest')
    for name,expected in manifest['source_hashes'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise ValueError('prebuilt host is stale for current source; rebuild explicitly: '+name)
    for row in manifest['files']:
        path=source/row['source']
        if path.stat().st_size!=row['size'] or sha(path)!=row['sha256']:raise ValueError('host binary checksum mismatch')
    destination=ROOT/'artifacts';destination.mkdir(exist_ok=True)
    for row in manifest['files']:
        target=destination/row['destination']
        if target.exists():
            old=sha(target)
            if old==row['sha256']:continue
            backup=destination/'hokoff-host-backups'/old/target.name
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():shutil.copy2(target,backup)
        temporary=target.with_name(target.name+'.hokoff-tmp')
        if temporary.exists():raise FileExistsError(temporary)
        shutil.copy2(source/row['source'],temporary)
        temporary.replace(target)
    print('Source-bound match host verified and prepared. No game/model assets included.')

if __name__=='__main__':main()
