import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.audit_expert_scenarios import _latency_bucket, _row_counts, audit_shard


class ScenarioAuditTests(unittest.TestCase):
    def test_row_counts_handles_empty_and_multiple_entities(self):
        offsets=np.array([0,0,2,3],dtype=np.int64)
        selected=np.array([True,False,True])
        np.testing.assert_array_equal(_row_counts(offsets,selected),[0,1,1])

    def test_latency_boundaries(self):
        self.assertEqual(_latency_bucket(0),'lt_10_ticks')
        self.assertEqual(_latency_bucket(10),'lt_20_ticks')
        self.assertEqual(_latency_bucket(640),'ge_640_ticks')

    def test_audit_classifies_pressure_and_response_without_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            arrays={
                'play_now':np.array([0,0,1,0,1],dtype=bool), 'timing_label_mask':np.ones(5,dtype=bool),
                'sample_weight':np.ones(5,dtype=np.float32), 'entity_offsets':np.array([0,0,1,2,3,3]),
                'entity_relations':np.array([1,1,0],dtype=np.uint8), 'entity_positions':np.array([6*18,14*18,20*18],dtype=np.int16),
                'ability_mask':np.zeros((5,2),dtype=bool), 'replay_extent':np.ones(5,dtype=np.uint8),
                'card_label_mask':np.array([0,0,1,0,1],dtype=bool), 'card_slot':np.array([-100,-100,0,-100,0]),
                'hand_tokens':np.tile(np.array([1,0,0,0]),(5,1)), 'position':np.array([-100,-100,9*18,-100,25*18]),
                'sequence_offsets':np.array([0,5],dtype=np.int64),
            }
            for name,value in arrays.items(): np.save(p/(name+'.npy'),value)
            result=audit_shard((str(p),'train',['<PAD>','knight@26000000']))
            self.assertEqual(result['scenes']['critical_tower_pressure']['rows'],1)
            self.assertEqual(result['scenes']['enemy_in_own_half']['rows'],2)
            self.assertEqual(result['scenes']['counterpush']['rows'],1)
            self.assertEqual(result['scenes']['all_valid']['troop_labels'],2)
            self.assertEqual(result['scenes']['all_valid']['card_token_counts'],{'1':2})
            self.assertEqual(result['latency']['critical_tower_pressure']['lt_10_ticks'],1)


if __name__=='__main__':unittest.main()
