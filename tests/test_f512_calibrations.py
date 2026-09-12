"""Registered per-family cp calibrations must not fall through to sigmoid-400."""

from training.labels import CALIBRATIONS, DEFAULT_CP_CALIBRATION, cp_to_u


def test_datav2_families_are_registered():
    for name, scale in (("datav2-gen", 275.06), ("datav2-self", 274.88), ("datav2-tb6", 40.0)):
        assert name in CALIBRATIONS
        assert abs(CALIBRATIONS[name].scale - scale) < 1e-6
        # 400 cp is a decisive win under s=275, not ~0.73 under sigmoid-400
        u = cp_to_u(400.0, name)
        u400 = DEFAULT_CP_CALIBRATION.u(400.0)
        if name != "datav2-tb6":
            assert u > 0.75
            assert abs(u - u400) > 0.05


def test_placeholder_still_exists_for_unversioned_cp():
    assert DEFAULT_CP_CALIBRATION.name in CALIBRATIONS
    assert abs(cp_to_u(0.0, None) - 0.5) < 1e-12
