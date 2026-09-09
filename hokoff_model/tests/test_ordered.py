import unittest
import numpy as np
import torch
from hokoff_model.ordered_data import labels_at
from hokoff_model.ordered_model import OrderedDecoder, ordered_loss, sequence_metrics, continuation_metrics


def arrays(events):
    n=12
    a=dict(timing_label_mask=np.ones(n,bool),play_now=np.zeros(n,bool),
           action_kind=np.zeros(n,int),kind_label_mask=np.ones(n,bool),
           action_kind_mask=np.ones((n,2),bool),card_slot=np.zeros(n,int),position=np.ones(n,int),
           card_label_mask=np.ones(n,bool),position_label_mask=np.ones(n,bool),
           card_mask=np.ones((n,4),bool),hand_tokens=np.tile([1,2,3,4],(n,1)),
           ability_tokens=np.tile([1,0],(n,1)),ability_mask=np.ones((n,2),bool),
           ability_label_mask=np.ones(n,bool),ability_slot=np.zeros(n,int),
           selected_position_mask_rows=np.arange(n),selected_position_mask_packed=np.full((n,72),255,np.uint8))
    for row,kind in events:a['play_now'][row]=True;a['action_kind'][row]=kind
    return a


class OrderedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(5)

    def test_stop_single_and_ordered_two_with_original_offsets(self):
        for events,types,masks,ticks in [([], [0,0],[True,False],[0,0]),
                                        ([(2,0)],[1,0],[True,True],[2,0]),
                                        ([(1,0),(3,1)],[1,2],[True,True],[1,3]),
                                        ([(0,1),(2,0)],[2,1],[True,True],[0,2])]:
            t,reason=labels_at(arrays(events),0,12)
            self.assertIsNone(reason)
            self.assertEqual(t['type'].tolist(),types)
            self.assertEqual(t['mask'].tolist(),masks)
            self.assertEqual(t['tick'].tolist(),ticks)
            for i,k in enumerate(types):
                if k!=1:self.assertEqual(t['position'][i],576)

    def test_unknown_tails_collisions_and_ambiguous_heroes_are_not_stop(self):
        a=arrays([(0,0),(1,0),(2,0)])
        self.assertEqual(labels_at(a,0,12)[1],'more_than_two')
        self.assertIsNone(labels_at(arrays([]),0,3)[0])
        a=arrays([]);a['timing_label_mask'][3]=False
        self.assertIsNone(labels_at(a,0,12)[0])
        a=arrays([(1,1)]);a['ability_tokens'][1]=[1,1]
        self.assertEqual(labels_at(a,0,12)[1],'ambiguous_hero_type')
        a=arrays([(1,0)]);a['card_label_mask'][1]=False
        self.assertIsNone(labels_at(a,0,12)[0])

    def test_legality_is_checked_at_recorded_time_not_advanced(self):
        a=arrays([(2,0)]);a['card_mask'][0]=False
        self.assertIsNotNone(labels_at(a,0,12)[0])
        a['card_mask'][2]=False
        self.assertIsNone(labels_at(a,0,12)[0])

    def targets(self):
        t,_=labels_at(arrays([(0,0),(2,1)]),0,12)
        return {k:torch.from_numpy(v).unsqueeze(0) for k,v in t.items()}

    def test_first_action_does_not_read_future_labels(self):
        m=OrderedDecoder(8,6,3);h=torch.randn(1,8);t=self.targets()
        out,_=m(h,t)
        changed={k:v.clone() for k,v in t.items()};changed['type'][0,1]=0;changed['token'][0,1]=0
        other,_=m(h,changed)
        for k in out[0]:torch.testing.assert_close(out[0][k],other[0][k],rtol=0,atol=0)
        for k in ('type','card','ability','tick'):
            torch.testing.assert_close(out[1][k],other[1][k],rtol=0,atol=0)

    def test_only_ordered_variant_conditions_second_on_first_action(self):
        h=torch.randn(1,8);t=self.targets();changed={k:v.clone() for k,v in t.items()}
        changed['token'][0,0]=2
        for conditioned in (False,True):
            m=OrderedDecoder(8,6,3,conditioned=conditioned)
            a,_=m(h,t);b,_=m(h,changed)
            self.assertEqual(torch.equal(a[1]['type'],b[1]['type']),not conditioned)
            torch.testing.assert_close(a[0]['type'],b[0]['type'],rtol=0,atol=0)

    def test_stop_and_end_of_period_force_termination(self):
        m=OrderedDecoder(8,6,3);h=torch.randn(3,8)
        with torch.no_grad():
            m.kind.weight.zero_();m.kind.bias.copy_(torch.tensor([10.,0.,0.]))
        _,pred=m(h);self.assertTrue((pred['type']==0).all())
        with torch.no_grad():
            m.kind.bias.copy_(torch.tensor([0.,10.,0.]));m.tick.weight.zero_()
            m.tick.bias.copy_(torch.tensor([0.,0.,0.,10.]))
        _,pred=m(h);self.assertTrue((pred['type'][:,0]==1).all());self.assertTrue((pred['type'][:,1]==0).all())

    def test_skill_has_no_position_loss_and_second_receives_gradients(self):
        m=OrderedDecoder(8,6,3);t=self.targets();h=torch.randn(1,8)
        out,_=m(h,t);loss,_=ordered_loss(out,t);loss.backward()
        self.assertGreater(float(m.transition.weight_ih.grad.abs().sum()),0)
        t['type'][:]=2;t['token'][:]=1;t['position'][:]=576
        out,_=m(h,t);loss,stats=ordered_loss(out,t)
        self.assertEqual(stats['position'],0.)
        self.assertTrue(torch.isfinite(loss))

    def test_oracle_metric_does_not_score_the_given_first_action(self):
        t=self.targets();pred={k:v.clone() for k,v in t.items() if k!='mask'}
        pred['token'][0,1]=2
        stats=continuation_metrics(pred,t)
        self.assertEqual(stats['two_action_count'],1)
        self.assertEqual(stats['second_action_exact_accuracy'],0.)
        self.assertIsNone(stats['stop_after_one_accuracy'])

    def test_predicted_prefix_reproduces_free_decode(self):
        m=OrderedDecoder(8,6,3);h=torch.randn(4,8)
        _,free=m(h);prefix={k:v[:,0] for k,v in free.items()}
        _,repeated=m(h,first_action=prefix)
        for k in free:torch.testing.assert_close(free[k],repeated[k],rtol=0,atol=0)

    def test_wait_only_batch_is_finite_and_checkpoint_decodes_without_labels(self):
        m=OrderedDecoder(8,6,3);t,_=labels_at(arrays([]),0,12)
        t={k:torch.from_numpy(v).unsqueeze(0) for k,v in t.items()};h=torch.randn(1,8)
        out,_=m(h,t);loss,stats=ordered_loss(out,t);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(stats['position'],0.)
        copy=OrderedDecoder(8,6,3);copy.load_state_dict(m.state_dict())
        _,a=m(h);_,b=copy(h)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

    def test_skill_position_does_not_affect_sequence_accuracy(self):
        t=self.targets();pred={k:v.clone() for k,v in t.items() if k!='mask'}
        pred['position'][0,1]=123
        self.assertEqual(sequence_metrics(pred,t)['sequence_accuracy'],1.)
        pred['type'][0,1]=0
        self.assertEqual(sequence_metrics(pred,t)['count_2_recall'],0.)


if __name__=='__main__':unittest.main()
