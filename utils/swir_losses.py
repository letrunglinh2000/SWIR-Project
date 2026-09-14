"""Composite training loss (spec SS10.2).

Baseline is plain L1; every extra term is config-gated and defaults to zero so
the L1 convergence arm is the literal default (spec: "prove convergence first").

`L_freq` is an EXPERIMENTAL SLOT, not a prescribed formulation. A naive complex
L2 spectrum loss is largely redundant with image-space L2 by Parseval, so the
arms are: none | amplitude | complex_l1, each normalised by the number of bins.
The final method may omit it entirely if it adds no reproducible value.

The SSIM here is a differentiable training term. Reported SSIM always comes from
the pinned skimage implementation in utils/swir_metrics.py -- do not mix them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FREQ_MODES = ('none', 'amplitude', 'complex_l1')


def _gaussian_window(win_size, sigma, channels, device, dtype):
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    win = (g.t() @ g).expand(channels, 1, win_size, win_size).contiguous()
    return win


def ssim_map(x, y, win_size=11, sigma=1.5, data_range=1.0):
    c = x.shape[1]
    win = _gaussian_window(win_size, sigma, c, x.device, x.dtype)
    pad = win_size // 2

    mu_x = F.conv2d(x, win, padding=pad, groups=c)
    mu_y = F.conv2d(y, win, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sx = F.conv2d(x * x, win, padding=pad, groups=c) - mu_x2
    sy = F.conv2d(y * y, win, padding=pad, groups=c) - mu_y2
    sxy = F.conv2d(x * y, win, padding=pad, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx + sy + c2))


def ssim_loss(x, y):
    return 1.0 - ssim_map(x, y).mean()


def edge_loss(x, y):
    dx = (x[..., :, 1:] - x[..., :, :-1]) - (y[..., :, 1:] - y[..., :, :-1])
    dy = (x[..., 1:, :] - x[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def frequency_loss(x, y, mode='amplitude'):
    """Normalised spectral loss; see the module docstring for why it is a slot."""
    if mode == 'none':
        return x.new_zeros(())
    if mode not in FREQ_MODES:
        raise ValueError('freq mode must be one of {}, got {!r}'.format(FREQ_MODES, mode))

    # torch.fft has no half kernels; keep this term in fp32 (same rule as D-3).
    with torch.amp.autocast(x.device.type, enabled=False):
        fx = torch.fft.rfft2(x.float(), norm='ortho')
        fy = torch.fft.rfft2(y.float(), norm='ortho')
        if mode == 'amplitude':
            return (fx.abs() - fy.abs()).abs().mean()
        return (fx - fy).abs().mean()


class CompositeLoss(nn.Module):
    def __init__(self, cfg=None):
        super(CompositeLoss, self).__init__()
        cfg = cfg or {}
        self.w_l1 = float(cfg.get('l1', 1.0))
        self.w_ssim = float(cfg.get('ssim', 0.0))
        self.w_edge = float(cfg.get('edge', 0.0))
        freq = cfg.get('freq', {}) or {}
        self.freq_mode = freq.get('mode', 'none')
        self.w_freq = float(freq.get('weight', 0.0))
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        terms = {'l1': self.l1(pred, target)}
        total = self.w_l1 * terms['l1']

        if self.w_ssim > 0:
            terms['ssim'] = ssim_loss(pred.clamp(0, 1), target)
            total = total + self.w_ssim * terms['ssim']
        if self.w_edge > 0:
            terms['edge'] = edge_loss(pred, target)
            total = total + self.w_edge * terms['edge']
        if self.w_freq > 0 and self.freq_mode != 'none':
            terms['freq'] = frequency_loss(pred, target, self.freq_mode)
            total = total + self.w_freq * terms['freq']

        terms['total'] = total
        return total, {k: float(v.detach()) for k, v in terms.items()}

    def describe(self):
        return {'l1': self.w_l1, 'ssim': self.w_ssim, 'edge': self.w_edge,
                'freq_mode': self.freq_mode, 'freq_weight': self.w_freq}
