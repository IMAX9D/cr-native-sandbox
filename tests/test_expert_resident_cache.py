import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(__import__('os').name == 'posix', 'Linux cache helper')
class ResidentCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('cache_helper', Path(__file__).parents[1]/'scripts/keep_expert_dataset_hot.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_budget_keeps_cgroup_and_host_reserve(self):
        budget = self.module.cache_budget
        self.assertEqual(budget(68, 12, {'limit':92,'used':12,'host_available':500}), 68)
        self.assertEqual(budget(68, 12, {'limit':92,'used':30,'host_available':500}), 50)
        self.assertEqual(budget(68, 12, {'limit':92,'used':12,'host_available':20}), 8)
        self.assertEqual(budget(68, 12, {'limit':92,'used':85,'host_available':500}), 0)

    def test_candidates_are_training_files_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            manifest={'splits':{'train':['shards/train-0']}, 'shard_file_sha256':{
                'shards/train-0/public_scalars.npy':'', 'shards/test-0/public_scalars.npy':'',
                'shards/train-0/shard.json':''}}
            self.assertEqual(self.module.candidate_files(root,manifest,'train'),
                             [root/'shards/train-0/public_scalars.npy'])

    def test_readonly_lock_and_release_leave_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.npy'
            content=b'actual training bytes' * 1024
            path.write_bytes(content)
            cache=self.module.ResidentFiles()
            try:
                cache.add(path,len(content))
                self.assertEqual(cache.bytes,len(content))
                self.assertEqual(path.read_bytes(),content)
            finally:
                cache.release(cache.bytes)
            self.assertEqual(cache.bytes,0)
            self.assertEqual(cache.pages,[])


if __name__ == '__main__':
    unittest.main()
