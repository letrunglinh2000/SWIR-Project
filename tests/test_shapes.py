"""M1 shape contract (spec SS16)."""

import pytest
import torch

from net.swir_adair_sr import SWIRAdaIRSR


@pytest.fixture(scope='module')
def model():
    return SWIRAdaIRSR(inp_channels=1, out_channels=1)


@pytest.mark.parametrize('h,w', [(128, 128), (256, 320)])
def test_scale2_doubles(model, h, w):
    assert model(torch.randn(1, 1, h, w), out_scale=2).shape == (1, 1, 2 * h, 2 * w)


@pytest.mark.parametrize('h,w', [(128, 128), (256, 320)])
def test_scale1_preserves(model, h, w):
    assert model(torch.randn(1, 1, h, w), out_scale=1).shape == (1, 1, h, w)


def test_no_hardcoded_widths():
    """INV-4: head input width is derived from dim, not a literal 96."""
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1, dim=32)
    assert m.head_in_channels == 64
    assert m(torch.randn(1, 1, 64, 64), out_scale=2).shape == (1, 1, 128, 128)


def test_unbuilt_scale_raises():
    m = SWIRAdaIRSR(scales=(1,))
    with pytest.raises(ValueError):
        m(torch.randn(1, 1, 64, 64), out_scale=2)


def test_base_gate_is_identity_at_init():
    """g=1, b=0 at initialisation (spec SS8)."""
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1)
    feat = torch.randn(2, m.head_in_channels, 8, 8)
    g, b = m.base_gate(feat)
    assert torch.allclose(g, torch.ones_like(g))
    assert torch.allclose(b, torch.zeros_like(b))
