"""Same seed => identical phi and identical output (spec SS7, M2)."""

import numpy as np
import pytest
import yaml

from utils.swir_degradation import EVAL_SUBSETS, build_pipeline
from utils.swir_io import load_data_profile


@pytest.fixture(scope='module')
def setup():
    profile = load_data_profile()
    cfg = yaml.safe_load(open('configs/swir_base.yaml'))['degradation']
    rng = np.random.default_rng(0)
    hr = rng.random((256, 256)).astype(np.float32)
    return profile, cfg, hr


@pytest.mark.parametrize('out_scale', [1, 2])
def test_same_seed_same_output(setup, out_scale):
    profile, cfg, hr = setup
    pipe = build_pipeline(cfg, profile)
    a, ma = pipe(hr, out_scale, seed=42)
    b, mb = pipe(hr, out_scale, seed=42)
    assert np.array_equal(a, b)
    assert ma == mb


def test_different_seed_differs(setup):
    profile, cfg, hr = setup
    pipe = build_pipeline(cfg, profile)
    a, _ = pipe(hr, 2, seed=1)
    b, _ = pipe(hr, 2, seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize('out_scale,factor', [(1, 1), (2, 2)])
def test_scale_operator_applies_exactly_once(setup, out_scale, factor):
    """D_2 contains exactly one x2 reduction; D_1 contains none (spec SS6)."""
    profile, cfg, hr = setup
    pipe = build_pipeline(cfg, profile)
    lr, meta = pipe(hr, out_scale, seed=7)
    assert lr.shape == (hr.shape[0] // factor, hr.shape[1] // factor)
    assert meta['scale'] == out_scale


def test_forced_flags_pin_components(setup):
    profile, cfg, hr = setup
    pipe = build_pipeline(cfg, profile)
    for name, force in EVAL_SUBSETS.items():
        if force is None:
            continue
        _, meta = pipe(hr, 2, seed=3, force_flags=force)
        assert meta['flags'] == force, (name, meta['flags'])


def test_sim_a_and_sim_b_differ(setup):
    profile, cfg, hr = setup
    cfg_b = yaml.safe_load(open('configs/sim_b.yaml'))['degradation']
    assert cfg['optics']['kernel_families'] != cfg_b['optics']['kernel_families']
    assert cfg['noise']['version'] != cfg_b['noise']['version']
    assert cfg['backscatter']['t_field'] != cfg_b['backscatter']['t_field']

    a, _ = build_pipeline(cfg, profile)(hr, 2, seed=11)
    b, _ = build_pipeline(cfg_b, profile)(hr, 2, seed=11)
    assert not np.array_equal(a, b)


def test_unknown_flag_rejected(setup):
    profile, cfg, hr = setup
    with pytest.raises(ValueError):
        build_pipeline(cfg, profile)(hr, 2, seed=1, force_flags={'sharpen': True})
