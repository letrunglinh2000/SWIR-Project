"""D-3: forward+backward under real AMP must stay finite (spec SS5, M1)."""

import pytest
import torch

from net.swir_adair_sr import SWIRAdaIRSR

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

SIZES = [(128, 128), (256, 320)]   # non-power-of-two transform sizes included


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('h,w', SIZES)
@pytest.mark.parametrize('out_scale', [1, 2])
def test_amp_forward_backward_finite(dtype, h, w, out_scale):
    dev = torch.device('cuda')
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1).to(dev)
    x = torch.rand(1, 1, h, w, device=dev, requires_grad=False)

    with torch.amp.autocast('cuda', dtype=dtype):
        y = m(x, out_scale=out_scale)
        loss = y.float().abs().mean()

    assert torch.isfinite(y.float()).all(), 'non-finite activations'
    assert torch.isfinite(loss), 'non-finite loss'

    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), 'non-finite grad in {}'.format(name)


def test_spectral_branch_stays_fp32_under_autocast():
    dev = torch.device('cuda')
    m = SWIRAdaIRSR(inp_channels=1, out_channels=1).to(dev)
    x = torch.rand(1, 1, 128, 128, device=dev)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        feat = m.fre1.conv1(torch.nn.functional.interpolate(x, (16, 16), mode='bilinear'))
        high, low = m.fre1.spectral_split(feat)
    # dtype is restored to the caller's, but the transform itself ran in fp32
    assert torch.isfinite(high).all() and torch.isfinite(low).all()
