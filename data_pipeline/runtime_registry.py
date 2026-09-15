"""Inventory the three retained official runtime/resource archives separately."""
import argparse
import json
import os
from pathlib import Path
from .prepare_600k import digest,atomic_json


def inventory():
    project=Path(__file__).resolve().parents[1]
    old=json.loads((project/'bindings/runtime-manifest.json').read_text('utf-8'))
    direct=Path(os.environ.get('CR_NATIVE_RUNTIME_EXPORT', str(project/'local/runtime-export')))
    fp=json.loads((direct/'data/update/fingerprint.json').read_text('utf-8'))
    entries=[dict(id='official-150535029-x86_64-resource-15.535.29',client_version='150535029',abi='x86_64',
        libg=str(direct/'libg.so'),libg_sha256=digest(direct/'libg.so'),resource_version=fp['version'],fingerprint=fp['sha'],
        native_library_hashes={e['name']:e['sha256'] for e in old['native_libs']},
        runtime_backend='linux-bionic',capture_backend_tested=True,exact_historical_intervals=[],
        r0_feature_certificate=None)]
    for day in ('20260908','20260912'):
        folder=project.parent/'outputs'/('kernel-snapshot-160402002-'+day)
        summary=json.loads((folder/'snapshot-summary.json').read_text('utf-8'))
        data=folder/('cr-native-kernel-160402002-arm64-'+day)
        native=data/'native/arm64-v8a'
        entries.append(dict(id='official-160402002-arm64-resource-'+summary['fingerprint_version'],
            client_version='160402002',abi='arm64-v8a',libg=str(native/'libg.so'),libg_sha256=digest(native/'libg.so'),
            resource_version=summary['fingerprint_version'],fingerprint=summary['fingerprint_sha'],
            native_library_hashes={e['name']:e['sha256'] for e in summary['native_libraries']},
            resource_root=str(data),captured_utc=summary['captured_utc'],
            runtime_backend='linux-arm64-port-pending',capture_backend_tested=False,
            exact_historical_intervals=[],r0_feature_certificate=None))
    return dict(schema='cr-runtime-resource-registry.v1',entries=entries,
        note='Archive count is not distinct libg binary count. Do not substitute an older runtime for an unready newer one; capture timestamps are not effective patch timestamps.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=inventory();atomic_json(a.output,result)
    print(json.dumps([dict(id=e['id'],abi=e['abi'],backend=e['runtime_backend']) for e in result['entries']]))
