import copy
import unittest

from scripts.audit_native_worker_costs import compare, libg_mappings, parse_memory, parse_stat


class WorkerCostAuditTests(unittest.TestCase):
    def test_proc_names_with_spaces_and_parentheses(self):
        fields=['S']+['0']*40
        fields[11],fields[12],fields[19],fields[36]='20','5','1234','7'
        row=parse_stat('42 (Jit pool (test)) '+' '.join(fields))
        self.assertEqual(row['name'],'Jit pool (test)')
        self.assertEqual(row['cpu_ticks'],25)
        self.assertEqual(row['start_ticks'],1234)
        self.assertEqual(row['processor'],7)

    def test_live_threads_are_not_counted_as_busy_cores(self):
        first={'pid':42,'monotonic':10,'process':{'start_ticks':1,'cpu_ticks':100},
               'threads':{'1':{'name':'main','start_ticks':2,'cpu_ticks':100},
                          '2':{'name':'GC','start_ticks':3,'cpu_ticks':0}},
               'memory_bytes':{},'libg_mappings':[],'read_failures':0}
        second=copy.deepcopy(first);second['monotonic']=12
        second['process']['cpu_ticks']=120;second['threads']['1']['cpu_ticks']=120
        row=compare(first,second,clock_hz=100)
        self.assertEqual(row['threads_at_end'],2)
        self.assertAlmostEqual(row['process_mean_cpu_cores'],0.1)
        self.assertEqual(row['thread_groups'][1]['cpu_seconds'],0)
        second['process']['start_ticks']=99
        with self.assertRaisesRegex(ValueError,'reused'):compare(first,second,clock_hz=100)

    def test_inode_not_path_or_segment_count_defines_backing_identity(self):
        text='1000-2000 r-xp 0 00:12 41 /slot0/libg.so\n2000-3000 r--p 1000 00:12 41 /slot0/libg.so\n'
        self.assertEqual(len(libg_mappings(text)),1)
        self.assertEqual(libg_mappings(text)[0]['inode'],41)

    def test_memory_units_and_missing_fields(self):
        result=parse_memory('Pss: 10 kB\nPrivate_Dirty: 7 kB\nSize: 999999 kB\n')
        self.assertEqual(result,{'Pss':10240,'Private_Dirty':7168})


if __name__=='__main__':unittest.main()
