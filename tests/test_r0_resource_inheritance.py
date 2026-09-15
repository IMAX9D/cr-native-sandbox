import pytest
from training_r0.replay_facts import effective_definition, nominal_geometry


def test_native_short_base_and_dynamic_ext_table_alias():
    assert nominal_geometry('CHARACTER.Knight_EV1') == .5
    assert nominal_geometry('EXT.RoyalHog_EV1_Grounded') == .6
    assert nominal_geometry('EXT.BerserkerHero_BearForm') == .45


def test_native_additive_radius_not_stale_inherited_value():
    assert nominal_geometry('AREA_EFFECT_OBJECT.Zap_EV1') == 2.5
    assert nominal_geometry('AREA_EFFECT_OBJECT.Zap_EV1_SpawnAOE_medium') == 3.


def test_shape_and_visual_only_effect_do_not_invent_circular_radius():
    wind=effective_definition('AEO.BabyDragon_EV1_wind_aeo')
    assert wind['Shape']=='BabyDragon_EV1_wind_aeo_shape'
    assert nominal_geometry('AEO.BabyDragon_EV1_wind_aeo') is None
    assert nominal_geometry('AREA_EFFECT_OBJECT.DummyZap_EV1_Visual') is None


def test_unknown_and_cycle_fail_closed():
    assert effective_definition('CHARACTER.NotARealRow') == {}
    with pytest.raises(ValueError, match='cycle'):
        effective_definition('CHARACTER.Knight',('CHARACTER.Knight',))
