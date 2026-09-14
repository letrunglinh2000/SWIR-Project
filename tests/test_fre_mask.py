"""D-2: frequency mask degeneracy and Soft-FMiM differentiability (spec SS5)."""

import torch

from net.model import FreModule as UpstreamFreModule
from net.swir_adair_sr import SoftFreModule

# (dim, h, w) for fre1/fre2/fre3 at a 128px patch and at full frame
CASES = [('fre1@128', 384, 16, 16), ('fre2@128', 192, 32, 32),
         ('fre3@128', 96, 64, 64), ('fre3@512x640', 96, 256, 320)]


def test_A_released_behaviour_is_recorded_not_assumed():
    """Record the released integer-mask occupancy; assert the degeneracy found."""
    torch.manual_seed(0)
    report = {}
    for name, dim, h, w in CASES:
        f = UpstreamFreModule(dim, num_heads=4, bias=False, in_dim=1)
        high, low = f.fft(torch.randn(2, 1, h, w))
        report[name] = (low.abs().mean().item(), high.abs().mean().item())

    for name in ('fre1@128', 'fre2@128', 'fre3@128'):
        assert report[name][0] == 0.0, (name, report[name])
        assert report[name][1] > 0.0
    print('released integer-mask low/high means:', report)


def test_B_soft_mask_is_non_degenerate():
    torch.manual_seed(0)
    f = SoftFreModule(96, num_heads=4, bias=False, in_dim=1, mask_mode='soft_relative')
    mask = f.debug_mask(torch.randn(2, 1, 64, 64))
    assert 0.01 < mask.mean().item() < 0.99, mask.mean().item()


def test_C_both_spectral_branches_contain_signal():
    torch.manual_seed(0)
    f = SoftFreModule(96, num_heads=4, bias=False, in_dim=1, mask_mode='soft_relative')
    low, high = f.debug_split(torch.randn(2, 1, 64, 64))
    assert low.abs().mean() > 0
    assert high.abs().mean() > 0


def test_D_cutoff_predictor_is_differentiable():
    """para1 is zero-initialised upstream, so neutralise that gate first."""
    torch.manual_seed(0)
    f = SoftFreModule(96, num_heads=4, bias=False, in_dim=1, mask_mode='soft_relative')
    with torch.no_grad():
        f.para1.fill_(1.0)

    f.zero_grad(set_to_none=True)
    out = f(torch.randn(2, 1, 64, 64), torch.randn(2, 96, 32, 32))
    out.square().mean().backward()

    g = f.rate_conv[0].weight.grad
    assert g is not None
    assert g.abs().sum() > 0


def test_upstream_integer_mode_gets_no_cutoff_gradient():
    """The released mask is non-differentiable -- the contrast that motivates A2."""
    torch.manual_seed(0)
    f = SoftFreModule(96, num_heads=4, bias=False, in_dim=1, mask_mode='upstream_integer')
    with torch.no_grad():
        f.para1.fill_(1.0)

    f.zero_grad(set_to_none=True)
    out = f(torch.randn(2, 1, 64, 64), torch.randn(2, 96, 32, 32))
    out.square().mean().backward()

    g = f.rate_conv[0].weight.grad
    assert g is None or g.abs().sum() == 0


def test_soft_mask_is_resolution_relative():
    """Same cutoff => same mask occupancy at different resolutions."""
    torch.manual_seed(0)
    f = SoftFreModule(96, num_heads=4, bias=False, in_dim=1, mask_mode='soft_relative')
    cut = torch.full((1, 1), 0.5)
    m_small = f._mask_soft(32, 32, cut, torch.device('cpu')).mean().item()
    m_large = f._mask_soft(256, 256, cut, torch.device('cpu')).mean().item()
    assert abs(m_small - m_large) < 0.02, (m_small, m_large)
