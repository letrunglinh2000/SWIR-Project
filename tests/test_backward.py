"""Gradient-coverage audit across both scale branches (spec SS5, SS19.1, M1)."""

import torch

from net.swir_adair_sr import SWIRAdaIRSR


def _coverage(model, sizes=((1, 1, 64, 64), (1, 1, 64, 64))):
    covered = set()
    for scale, shape in zip((1, 2), sizes):
        model.zero_grad(set_to_none=True)
        out = model(torch.randn(*shape), out_scale=scale)
        out.mean().backward()
        covered |= {n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None}
    return covered


def test_no_unintended_dead_parameters():
    """Every trainable parameter gets a gradient from s=1 or s=2 (not both)."""
    torch.manual_seed(0)
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1)
    with torch.no_grad():
        # para1 is zero-initialised, which legitimately blocks gradients inside
        # the frequency branch on the very first step (spec SS5 caveat).
        for f in (m.fre1, m.fre2, m.fre3):
            f.para1.fill_(1.0)

    expected = {n for n, p in m.named_parameters() if p.requires_grad}
    unexpected = expected - _coverage(m)
    assert not unexpected, sorted(unexpected)


def test_single_scale_batch_only_touches_its_own_head():
    """SS19.1: the unselected head receiving no gradient is expected, not a bug."""
    torch.manual_seed(0)
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1)
    m.zero_grad(set_to_none=True)
    m(torch.randn(1, 1, 64, 64), out_scale=1).mean().backward()

    assert all(p.grad is not None for p in m.output_heads['1'].parameters())
    assert all(p.grad is None for p in m.output_heads['2'].parameters())
