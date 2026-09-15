import hashlib
import io
import json
import lzma
from pathlib import Path
import sqlite3
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

from data_pipeline.prepare_600k import normalize,run


def payload(tag='AB123',schema=5):
    return dict(battle_tag=tag,schema_version=schema,timestamp=1786175552,
                battle_time_utc='2026-08-08T07:52:32Z',duration_seconds=10,
                team_deck=['x'],opponent_deck=['y'],ability_plays=[],
                card_plays=[dict(time_raw=20,time=1,side='team',marker_index=0,card='miner',x_raw=1000,y_raw=2000,data_i=0)])


class PreparationTests(unittest.TestCase):
    def source(self):return dict(battle_tag='AB123',catalog_timestamp=1786175552)

    def test_normalized_coordinates_and_causal_boundary(self):
        r=normalize(payload(),self.source())
        self.assertEqual(r['events'][0]['native_xy_candidate'],[17000,30000])
        self.assertFalse(r['training_ready']);self.assertIsNone(r['ruleset_id'])
        self.assertEqual(r['issues'],[])

    def test_missing_ticks_not_guessed(self):
        p=payload();del p['card_plays'][0]['time_raw']
        r=normalize(p,self.source())
        self.assertIsNone(r['events'][0]['tick']);self.assertIn('deploy_integer_tick_missing',r['issues'])

    def test_time_conflict(self):
        p=payload();p['timestamp']+=1
        self.assertIn('timestamp_conflicts_with_catalog',normalize(p,self.source())['issues'])

    def test_unknown_ability_not_wait(self):
        p=payload();p['ability_plays']=[dict(time_raw=30,time=1.5,side='opponent',marker_index=1)]
        r=normalize(p,self.source());self.assertEqual(r['events'][1]['identity_status'],'needs_native_resolution')

    def test_missing_coordinates_not_invented(self):
        p=payload( schema=3);del p['card_plays'][0]['data_i']
        r=normalize(p,self.source());self.assertIsNone(r['events'][0]['native_xy_candidate'])

    def fixture(self,root):
        archive=root/'input.tar.xz';catalog=root/'catalog.sqlite3'
        c=sqlite3.connect(catalog)
        c.execute('create table battles(tag text primary key,archive text,member text,sha256 text,size integer,timestamp integer,schema_version integer,source text)')
        with tarfile.open(archive,'w:xz') as t:
            for tag in ['AB123','AB124','AB125']:
                b=json.dumps(payload(tag)).encode();name='battles/AB/'+tag+'.json'
                m=tarfile.TarInfo(name);m.size=len(b);t.addfile(m,io.BytesIO(b))
                c.execute('insert into battles values(?,?,?,?,?,?,?,?)',(tag,archive.name,name,hashlib.sha256(b).hexdigest(),len(b),1786175552,5,'test'))
        c.commit();c.close()
        return SimpleNamespace(archive=archive,catalog=catalog,output=root/'out',expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),limit=2,shard_records=2,min_free_gib=1)

    def test_shards_resume_and_admission(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.fixture(Path(d));r=run(a);self.assertEqual(r['committed_records'],2)
            self.assertFalse(r['source_stage_complete']);a.limit=0
            r=run(a);self.assertEqual(r['committed_records'],3);self.assertTrue(r['source_stage_complete'])
            r=run(a);self.assertEqual(r['added_this_run'],0);self.assertEqual(r['training_ready_records'],0)

    def test_wrong_archive_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.fixture(Path(d));a.expected_sha256='0'*64
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):run(a)

    def test_concatenated_xz_streams(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.fixture(Path(d));raw=lzma.decompress(a.archive.read_bytes())
            a.archive.write_bytes(lzma.compress(raw[:1536])+lzma.compress(raw[1536:]))
            a.expected_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest();a.limit=0
            self.assertEqual(run(a)['committed_records'],3)


if __name__=='__main__':unittest.main()
