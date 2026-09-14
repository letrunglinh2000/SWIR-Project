"""Padding wrapper handles non-multiple-of-16 inputs (spec SS8, M1)."""

import pytest
import torch

from net.swir_adair_sr import SWIRAdaIRSR


@pytest.mark.parametrize('h,w', [(129, 131), (100, 70), (255, 319)])
@pytest.mark.parametrize('out_scale', [1, 2])
def test_odd_sizes_round_trip(h, w, out_scale):
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1)
    y = m(torch.randn(1, 1, h, w), out_scale=out_scale)
    assert y.shape == (1, 1, out_scale * h, out_scale * w)


def test_check_image_size_pads_to_multiple_of_16():
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1)
    padded, h, w = m.check_image_size(torch.randn(1, 1, 129, 131))
    assert (h, w) == (129, 131)
    assert padded.shape[-2] % 16 == 0 and padded.shape[-1] % 16 == 0
