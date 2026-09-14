"""D-1: FreModule must follow inp_channels (spec SS5)."""

import torch

from net.swir_adair_sr import SWIRAdaIRSR


def test_fre_in_channels_follow_inp_channels():
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1)
    for f in (m.fre1, m.fre2, m.fre3):
        assert f.conv1.in_channels == 1
    _ = m(torch.randn(1, 1, 128, 128), out_scale=2)   # must not raise


def test_three_channel_variant_still_builds():
    m = SWIRAdaIRSR(inp_channels=3, out_channels=3)
    for f in (m.fre1, m.fre2, m.fre3):
        assert f.conv1.in_channels == 3
    assert m(torch.randn(1, 3, 64, 64), out_scale=1).shape == (1, 3, 64, 64)


def test_dead_upstream_modules_not_carried_over():
    """Upstream `conv` / `score_gen` are unused; they must not be re-created."""
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1)
    for f in (m.fre1, m.fre2, m.fre3):
        assert not hasattr(f, 'score_gen')
        assert not hasattr(f, 'conv')
