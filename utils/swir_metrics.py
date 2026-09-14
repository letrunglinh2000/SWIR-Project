"""Pinned evaluation metrics (spec SS11.1).

Different libraries give different numbers, so the conventions are fixed here and
reported with the results:

  PSNR  : data_range=1.0, per image on [0,1]-clamped output, then averaged.
          No border shave.
  SSIM  : scikit-image `structural_similarity`, gaussian_weights=False,
          win_size=11, data_range=1.0, single channel.
  Edge  : mean |grad(yhat) - grad(y)| on forward differences (secondary metric,
          relevant to distant target contours).
  LPIPS : evaluation only, never a training loss. Backbone and package version
          are recorded; grayscale is replicated to 3 channels and mapped to
          [-1,1]. Values are NOT comparable with RGB-trained papers.
"""

import inspect

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SSIM_WIN_SIZE = 11
SSIM_CONVENTION = {
    'implementation': 'skimage.metrics.structural_similarity',
    'win_size': SSIM_WIN_SIZE,
    'gaussian_weights': False,
    'data_range': 1.0,
}

# skimage renamed `multichannel` to `channel_axis` in 0.19 and removed the old
# name in 0.23. Older versions silently absorb unknown kwargs, so select by
# signature rather than by catching an exception.
_HAS_CHANNEL_AXIS = 'channel_axis' in inspect.signature(structural_similarity).parameters

_LPIPS_CACHE = {}


class AverageMeter(object):
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def _to_numpy(t):
    """[B,1,H,W] torch -> [B,H,W] float64 numpy, clamped to [0,1]."""
    a = torch.clamp(t.detach(), 0, 1).float().cpu().numpy()
    if a.ndim != 4 or a.shape[1] != 1:
        raise ValueError('expected [B,1,H,W], got {}'.format(tuple(a.shape)))
    return a[:, 0].astype(np.float64)


def psnr(pred, target):
    p, t = _to_numpy(pred), _to_numpy(target)
    return float(np.mean([peak_signal_noise_ratio(t[i], p[i], data_range=1.0)
                          for i in range(p.shape[0])]))


def ssim(pred, target):
    p, t = _to_numpy(pred), _to_numpy(target)
    kw = dict(data_range=1.0, win_size=SSIM_WIN_SIZE, gaussian_weights=False)
    return float(np.mean([structural_similarity(t[i], p[i], **kw) for i in range(p.shape[0])]))


def edge_l1(pred, target):
    """Mean L1 between forward-difference gradients (lower is better)."""
    p = torch.clamp(pred.detach().float(), 0, 1)
    t = torch.clamp(target.detach().float(), 0, 1)
    dpx, dtx = p[..., :, 1:] - p[..., :, :-1], t[..., :, 1:] - t[..., :, :-1]
    dpy, dty = p[..., 1:, :] - p[..., :-1, :], t[..., 1:, :] - t[..., :-1, :]
    return float(((dpx - dtx).abs().mean() + (dpy - dty).abs().mean()).item() / 2.0)


def get_lpips(backbone='alex', device='cpu'):
    key = (backbone, str(device))
    if key not in _LPIPS_CACHE:
        import lpips as lpips_pkg
        net = lpips_pkg.LPIPS(net=backbone).to(device).eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        _LPIPS_CACHE[key] = net
    return _LPIPS_CACHE[key]


def lpips_score(pred, target, backbone='alex'):
    """Evaluation only (spec SS10.2). Grayscale is replicated to 3 channels."""
    device = pred.device
    net = get_lpips(backbone, device)
    p = torch.clamp(pred.detach().float(), 0, 1).repeat(1, 3, 1, 1) * 2 - 1
    t = torch.clamp(target.detach().float(), 0, 1).repeat(1, 3, 1, 1) * 2 - 1
    with torch.no_grad():
        return float(net(p, t).mean().item())


def compute_metrics(pred, target, metrics=('psnr', 'ssim', 'edge'), lpips_backbone=None):
    out = {}
    if 'psnr' in metrics:
        out['psnr'] = psnr(pred, target)
    if 'ssim' in metrics:
        out['ssim'] = ssim(pred, target)
    if 'edge' in metrics:
        out['edge'] = edge_l1(pred, target)
    if lpips_backbone:
        out['lpips'] = lpips_score(pred, target, lpips_backbone)
    return out


def metric_conventions(lpips_backbone=None):
    """Provenance block to embed in every results file (spec SS11.1, SS17.6)."""
    conv = {
        'psnr': {'data_range': 1.0, 'per_image': True, 'border_shave': 0},
        'ssim': dict(SSIM_CONVENTION, channel_axis_api=_HAS_CHANNEL_AXIS),
        'edge': {'definition': 'mean L1 of forward-difference gradients'},
    }
    if lpips_backbone:
        import lpips as lpips_pkg
        conv['lpips'] = {
            'backbone': lpips_backbone,
            'package_version': getattr(lpips_pkg, '__version__', 'unknown'),
            'grayscale_rule': 'replicate to 3 channels, map [0,1] -> [-1,1]',
            'comparable_with_rgb_papers': False,
        }
    return conv
