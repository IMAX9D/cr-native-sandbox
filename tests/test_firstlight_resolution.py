import unittest
from data_pipeline.firstlight_resolution import select_ability,rejected_window


def e(key,ordinal,**extra):
    return dict(side=0,generation_key=key,creation_ordinal=ordinal,card_id=26000000,
                ability_slot=1,ability_available=True,ability_cooldown_remaining_ms=0,
                ability_charges_remaining=1,**extra)


class ResolutionTests(unittest.TestCase):
    def test_newest_not_pointer_or_key(self):
        r=select_ability([e(999,1),e(2,7)],side=0)
        self.assertEqual(r['selected'],2);self.assertFalse(r['source_identity_confirmed'])
    def test_unaffordable_newest_falls_back(self):
        a=e(2,7);a['ability_available']=False
        self.assertEqual(select_ability([e(1,1),a],side=0)['selected'],1)
    def test_tie_not_arbitrary(self):
        self.assertIsNone(select_ability([e(1,2),e(2,2)],side=0)['selected'])
    def test_zero_unknown_charges_excluded(self):
        for charge in [0,-1,None]:
            a=e(1,1);a['ability_charges_remaining']=charge
            self.assertIsNone(select_ability([a],side=0)['selected'])
    def test_explicit_wrong_side_not_rebound(self):
        self.assertIsNone(select_ability([e(1,1)],side=1,exact_entity_key=1)['selected'])
    def test_identity_filter(self):
        self.assertIsNone(select_ability([e(1,1)],side=0,allowed_card_ids={42})['selected'])
    def test_boundary(self):
        self.assertIsNone(rejected_window(5,first_decision_tick=5))
        self.assertEqual(rejected_window(10,first_decision_tick=5),5)
        self.assertEqual(rejected_window(11,first_decision_tick=5),10)


if __name__=='__main__':unittest.main()
