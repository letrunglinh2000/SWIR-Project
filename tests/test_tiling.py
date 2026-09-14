"""Tiled inference is seam-checked and reported, not asserted equal (SS8, SS17.17)."""

import torch

from net.swir_adair_sr import SWIRAdaIRSR
from utils.swir_metrics import psnr


def test_tiled_matches_whole_frame_shape_and_is_finite():
    torch.manual_seed(0)
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1).eval()
    x = torch.rand(1, 1, 192, 192)

    with torch.no_grad():
        whole = m(x, out_scale=2)
        tiled = m.forward_tiled(x, out_scale=2, tile=96, overlap=16)

    assert tiled.shape == whole.shape
    assert torch.isfinite(tiled).all()


def test_tile_whole_delta_is_recorded():
    """Report delta-PSNR(tile vs whole); the FFT branch makes equality impossible."""
    torch.manual_seed(0)
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1).eval()
    x = torch.rand(1, 1, 192, 192)

    with torch.no_grad():
        whole = m(x, out_scale=2)
        tiled = m.forward_tiled(x, out_scale=2, tile=96, overlap=16)

    delta = psnr(tiled.clamp(0, 1), whole.clamp(0, 1))
    print('tiled-vs-whole PSNR: {:.2f} dB'.format(delta))
    assert delta > 20.0, 'tiling diverges implausibly far from whole-frame'


def test_tile_larger_than_image_falls_back_to_whole_frame():
    torch.manual_seed(0)
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1).eval()
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        a = m(x, out_scale=1)
        b = m.forward_tiled(x, out_scale=1, tile=256, overlap=16)
    assert torch.allclose(a, b)


def test_larger_overlap_does_not_hurt():
    torch.manual_seed(0)
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1).eval()
    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        whole = m(x, out_scale=1)
        lo = m.forward_tiled(x, out_scale=1, tile=96, overlap=8)
        hi = m.forward_tiled(x, out_scale=1, tile=96, overlap=32)
    print('overlap 8: {:.2f} dB | overlap 32: {:.2f} dB'.format(
        psnr(lo.clamp(0, 1), whole.clamp(0, 1)), psnr(hi.clamp(0, 1), whole.clamp(0, 1))))
