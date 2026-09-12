import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from training.league_dashboard import validate, empty_state, demo_state, write_snapshot, SnapshotReader, make_server


class LeagueMonitorTests(unittest.TestCase):
    def test_empty_is_not_fake_evaluation(self):
        d=validate(empty_state())
        self.assertIsNone(d['models'][0]['elo']);self.assertEqual(d['matchups'],[])
    def test_demo_valid_and_explicit(self):
        d=validate(demo_state());self.assertEqual(d['status'],'demo')
        self.assertIn('合成',d['note'])
    def test_invalid_numbers_and_duplicate_pairs(self):
        d=demo_state();d['models'][0]['elo']=float('nan')
        with self.assertRaises(ValueError):validate(d)
        d=demo_state();row=copy.deepcopy(d['matchups'][0]);row['a'],row['b']=row['b'],row['a'];d['matchups'].append(row)
        with self.assertRaises(ValueError):validate(d)
        d=demo_state();d['matchups'][0]['wins']=-1
        with self.assertRaises(ValueError):validate(d)
    def test_lineage_cycles_rejected(self):
        d=demo_state();d['models'][0]['parent']='p4'
        with self.assertRaises(ValueError):validate(d)
    def test_bad_shapes(self):
        for field,value in [('models',{}),('matchups',[1]),('qualifications',[None]),('telemetry',[])]:
            d=demo_state();d[field]=value
            with self.assertRaises(ValueError):validate(d)
    def test_atomic_publication_and_fail_closed_reader(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.json';r=SnapshotReader(p)
            write_snapshot(p,empty_state());self.assertEqual(r.read()['status'],'not_connected')
            write_snapshot(p,demo_state());self.assertEqual(r.read()['status'],'demo')
            p.write_text('{')
            with self.assertRaises(ValueError):r.read()
    def test_server_is_read_only_and_modes_separate(self):
        server=make_server(port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/api/state') as r:
                d=json.load(r);self.assertEqual(d['mode'],'real');self.assertEqual(d['data']['matchups'],[])
            with urlopen(base+'/api/state?demo=1') as r:self.assertEqual(json.load(r)['mode'],'demo')
            with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/api/state',method='POST'))
            self.assertEqual(error.exception.code,405)
            with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/api/state',headers={'Host':'attacker.example'}))
            self.assertEqual(error.exception.code,403)
            with urlopen(base+'/') as r:self.assertIn('克制矩阵',r.read().decode())
        finally:server.shutdown();server.server_close();thread.join(2)
    def test_live_snapshot_changes_and_corruption_visible(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'state.json';write_snapshot(path,empty_state())
            server=make_server(0,path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}/api/state'
            try:
                with urlopen(url) as r:self.assertEqual(json.load(r)['data']['status'],'not_connected')
                d=empty_state();d['run_name']='updated';write_snapshot(path,d)
                with urlopen(url) as r:self.assertEqual(json.load(r)['data']['run_name'],'updated')
                path.write_text('{')
                with self.assertRaises(HTTPError) as error:urlopen(url)
                self.assertEqual(error.exception.code,503)
            finally:server.shutdown();server.server_close();thread.join(2)

if __name__=='__main__':unittest.main()
