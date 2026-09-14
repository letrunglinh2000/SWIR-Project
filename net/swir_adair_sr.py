"""SWIRAdaIRSR -- AdaIR backbone for single-band SWIR restoration + x2 SR.

Spec SS8. The backbone blocks are imported from `net.model` (kept byte-identical
to the upstream release, tag `upstream-baseline`) so this file contains only the
project's own additions:

  * `SoftFreModule`  -- D-1..D-3 applied, with both mask modes (SS19.2)
                        `upstream_integer` : the exact released integer box
                        `soft_relative`    : proposed differentiable Soft-FMiM
  * scale heads      -- ModuleDict '1' -> conv, '2' -> PixelShuffle(2), ICNR init
  * gated base       -- identity-preserving at init (g=1, b=0)
  * padding wrapper and tiled inference

INV-2: `forward` never receives a degradation label. `out_scale` is a deployment
setting, not a degradation class (INV-3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model import (Chanel_Cross_Attention, Downsample, FreRefine,
                       OverlapPatchEmbed, TransformerBlock, Upsample)

MASK_MODES = ('upstream_integer', 'soft_relative')


# --------------------------------------------------------------------------- #
# Frequency module
# --------------------------------------------------------------------------- #

class SoftFreModule(nn.Module):
    """Adaptive Frequency Learning Block with a selectable mask mode.

    Upstream defects addressed (spec SS5):
      D-1  `in_dim` now follows `inp_channels` instead of being fixed at 3.
      D-2  the released mask radius `(h // 128) * threshold` truncates to 0 for
           every feature map used here, so the low band was identically zero and
           `rate_conv` received no gradient. `soft_relative` replaces it with a
           resolution-relative differentiable window.
      D-3  autocast is disabled around the spectral operations only, not around
           the whole block.

    Upstream `conv` and `score_gen` are confirmed unused in the released forward
    path and are deliberately not carried over (spec SS5, SS10.1).
    """

    def __init__(self, dim, num_heads, bias, in_dim=1,
                 mask_mode='soft_relative', tau=0.05, anisotropic=False):
        super(SoftFreModule, self).__init__()
        if mask_mode not in MASK_MODES:
            raise ValueError('mask_mode must be one of {}, got {!r}'.format(MASK_MODES, mask_mode))

        self.mask_mode = mask_mode
        self.tau = tau
        self.anisotropic = anisotropic

        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        # One cutoff for the isotropic window, two for the anisotropic/integer
        # variants. Sizing it by mode avoids leaving an unused output channel,
        # which spec SS5 forbids.
        self.n_cutoffs = 1 if (mask_mode == 'soft_relative' and not anisotropic) else 2
        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 8, self.n_cutoffs, 1, bias=False),
        )

    # -- mask construction -------------------------------------------------- #

    def _cutoffs(self, x):
        t = self.rate_conv(F.adaptive_avg_pool2d(x, 1)).sigmoid()
        return t.reshape(t.shape[0], self.n_cutoffs)

    def _mask_soft(self, h, w, cut, device):
        """M(u,v) = sigmoid((r - rho)/tau),  rho = sqrt((2u/h)^2 + (2v/w)^2).

        rho is 0 at DC and 1 at the Nyquist edge, so the window is
        resolution-relative and r is directly a normalised cutoff.
        """
        u = (torch.arange(h, device=device, dtype=torch.float32) - h // 2).view(1, h, 1)
        v = (torch.arange(w, device=device, dtype=torch.float32) - w // 2).view(1, 1, w)
        nu, nv = 2.0 * u / h, 2.0 * v / w

        if self.anisotropic:
            rh = cut[:, 0].view(-1, 1, 1)
            rw = cut[:, 1].view(-1, 1, 1)
            rho_h = torch.sigmoid((rh - nu.abs()) / self.tau)
            rho_w = torch.sigmoid((rw - nv.abs()) / self.tau)
            mask = rho_h * rho_w
        else:
            rho = torch.sqrt(nu ** 2 + nv ** 2)
            r = cut[:, 0].view(-1, 1, 1)
            mask = torch.sigmoid((r - rho) / self.tau)

        return mask.unsqueeze(1)

    def _mask_upstream(self, h, w, cut, device, n=128):
        """Exact released behaviour: integer box of half-width (dim//n)*threshold.

        Kept verbatim (loop included) so the ablation-A2 baseline reproduces the
        published code rather than an approximation of it. Non-differentiable by
        construction -- that is the point of the comparison.
        """
        mask = torch.zeros(cut.shape[0], 1, h, w, device=device)
        for i in range(cut.shape[0]):
            h_ = int((h // n) * cut[i, 0].item())
            w_ = int((w // n) * cut[i, 1].item())
            if h_ > 0 and w_ > 0:
                mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1
        return mask

    def build_mask(self, x):
        h, w = x.shape[-2:]
        cut = self._cutoffs(x)
        if self.mask_mode == 'soft_relative':
            return self._mask_soft(h, w, cut, x.device)
        return self._mask_upstream(h, w, cut, x.device)

    # -- spectral split ------------------------------------------------------ #

    @staticmethod
    def shift(x):
        h, w = x.shape[-2:]
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    @staticmethod
    def unshift(x):
        h, w = x.shape[-2:]
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def spectral_split(self, feat):
        """D-3: fp32 only around the FFT; callers keep their own AMP dtype."""
        orig_dtype = feat.dtype
        with torch.amp.autocast(feat.device.type, enabled=False):
            f32 = feat.float()
            mask = self.build_mask(f32)

            fft = torch.fft.fft2(f32, norm='forward', dim=(-2, -1))
            fft = self.shift(fft)

            high = self.unshift(fft * (1 - mask))
            high = torch.abs(torch.fft.ifft2(high, norm='forward', dim=(-2, -1)))

            low = self.unshift(fft * mask)
            low = torch.abs(torch.fft.ifft2(low, norm='forward', dim=(-2, -1)))

        return high.to(orig_dtype), low.to(orig_dtype)

    # -- test hooks (spec SS5 assertions) ------------------------------------ #

    def debug_mask(self, x):
        return self.build_mask(self.conv1(x))

    def debug_split(self, x):
        high, low = self.spectral_split(self.conv1(x))
        return low, high

    # -- forward ------------------------------------------------------------- #

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)

        high_feature, low_feature = self.spectral_split(self.conv1(x))

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2


# --------------------------------------------------------------------------- #
# Output heads
# --------------------------------------------------------------------------- #

def icnr_(weight, scale=2, initializer=nn.init.kaiming_normal_):
    """ICNR init so PixelShuffle starts as a nearest-neighbour upsample."""
    out_c, in_c, kh, kw = weight.shape
    sub = torch.zeros(out_c // (scale ** 2), in_c, kh, kw)
    initializer(sub)
    sub = sub.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        weight.copy_(sub)


class ScaleHeadX1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ScaleHeadX1, self).__init__()
        self.body = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.body(x)


class ScaleHeadX2(nn.Module):
    """x2 expansion. `in_channels` is always derived, never a literal (INV-4)."""

    def __init__(self, in_channels, out_channels, bias=False):
        super(ScaleHeadX2, self).__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * 4, 3, 1, 1, bias=bias)
        icnr_(self.expand.weight, scale=2)
        if bias:
            nn.init.zeros_(self.expand.bias)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.expand(x))


class BaseGate(nn.Module):
    """g = 1 + tanh(dg), b = b_max * tanh(db); zero-init => g=1, b=0 (spec SS8)."""

    def __init__(self, in_channels, b_max=0.2):
        super(BaseGate, self).__init__()
        self.b_max = b_max
        hidden = max(4, in_channels // 4)
        self.fc1 = nn.Conv2d(in_channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feat):
        h = F.gelu(self.fc1(F.adaptive_avg_pool2d(feat, 1)))
        d = self.fc2(h)
        g = 1.0 + torch.tanh(d[:, 0:1])
        b = self.b_max * torch.tanh(d[:, 1:2])
        return g, b


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class SWIRAdaIRSR(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, dim=48,
                 num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False,
                 layernorm_type='WithBias',
                 scales=(1, 2), gated_base=True,
                 mask_mode='soft_relative', fre_tau=0.05, anisotropic_mask=False,
                 use_frequency_modules=True, pad_base=16):
        super(SWIRAdaIRSR, self).__init__()

        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.scales = tuple(int(s) for s in scales)
        self.gated_base = gated_base
        self.use_frequency_modules = use_frequency_modules
        self.pad_base = pad_base

        def block(d, nh, n):
            return nn.Sequential(*[TransformerBlock(
                dim=int(d), num_heads=nh, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=layernorm_type) for _ in range(n)])

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        if use_frequency_modules:
            fre_kw = dict(bias=bias, in_dim=inp_channels, mask_mode=mask_mode,
                          tau=fre_tau, anisotropic=anisotropic_mask)
            self.fre1 = SoftFreModule(dim * 2 ** 3, num_heads=heads[2], **fre_kw)
            self.fre2 = SoftFreModule(dim * 2 ** 2, num_heads=heads[2], **fre_kw)
            self.fre3 = SoftFreModule(dim * 2 ** 1, num_heads=heads[2], **fre_kw)

        self.encoder_level1 = block(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = block(dim * 2 ** 1, heads[1], num_blocks[1])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = block(dim * 2 ** 2, heads[2], num_blocks[2])
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = block(dim * 2 ** 3, heads[3], num_blocks[3])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), 1, bias=bias)
        self.decoder_level3 = block(dim * 2 ** 2, heads[2], num_blocks[2])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), 1, bias=bias)
        self.decoder_level2 = block(dim * 2 ** 1, heads[1], num_blocks[1])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = block(dim * 2 ** 1, heads[0], num_blocks[0])
        self.refinement = block(dim * 2 ** 1, heads[0], num_refinement_blocks)

        head_in = int(dim * 2 ** 1)          # derived, never a literal (INV-4)
        self.head_in_channels = head_in
        heads_dict = {}
        if 1 in self.scales:
            heads_dict['1'] = ScaleHeadX1(head_in, out_channels, bias=bias)
        if 2 in self.scales:
            heads_dict['2'] = ScaleHeadX2(head_in, out_channels, bias=bias)
        self.output_heads = nn.ModuleDict(heads_dict)

        if gated_base:
            self.base_gate = BaseGate(head_in)

    # -- padding ------------------------------------------------------------- #

    def check_image_size(self, x):
        _, _, h, w = x.size()
        b = self.pad_base
        ph, pw = (b - h % b) % b, (b - w % b) % b
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x, h, w

    # -- core ---------------------------------------------------------------- #

    def _features(self, inp_img):
        enc1 = self.encoder_level1(self.patch_embed(inp_img))
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        latent = self.latent(self.down3_4(enc3))

        if self.use_frequency_modules:
            latent = self.fre1(inp_img, latent)

        dec3 = self.up4_3(latent)
        dec3 = self.reduce_chan_level3(torch.cat([dec3, enc3], 1))
        dec3 = self.decoder_level3(dec3)
        if self.use_frequency_modules:
            dec3 = self.fre2(inp_img, dec3)

        dec2 = self.up3_2(dec3)
        dec2 = self.reduce_chan_level2(torch.cat([dec2, enc2], 1))
        dec2 = self.decoder_level2(dec2)
        if self.use_frequency_modules:
            dec2 = self.fre3(inp_img, dec2)

        dec1 = self.up2_1(dec2)
        dec1 = self.decoder_level1(torch.cat([dec1, enc1], 1))
        return self.refinement(dec1)

    def forward(self, x, out_scale=2):
        """x: [B,1,H,W] float32 in [0,1]; returns [B,1,s*H,s*W]. No clamp (SS3)."""
        out_scale = int(out_scale)
        key = str(out_scale)
        if key not in self.output_heads:
            raise ValueError('out_scale {} not built; model has scales {}'.format(
                out_scale, sorted(self.output_heads.keys())))

        padded, h, w = self.check_image_size(x)
        feat = self._features(padded)
        res = self.output_heads[key](feat)

        if out_scale == 1:
            base = padded
        else:
            base = F.interpolate(padded, scale_factor=out_scale,
                                 mode='bicubic', align_corners=False)

        if self.gated_base:
            g, b = self.base_gate(feat)
            out = res + g * base + b
        else:
            out = res + base

        return out[..., :out_scale * h, :out_scale * w]

    # -- tiled inference (spec SS8) ------------------------------------------ #

    @torch.no_grad()
    def forward_tiled(self, x, out_scale=2, tile=256, overlap=32):
        """Feathered tile blending for memory-constrained inference.

        NOT numerically equal to whole-frame inference: the FFT branch depends on
        the spatial support of its input, so tile-wise spectra differ by
        construction (spec SS8, SS17.17). Whole-frame is the primary benchmark.
        """
        b, c, h, w = x.shape
        s = int(out_scale)
        if tile >= max(h, w):
            return self.forward(x, out_scale=s)

        stride = max(1, tile - overlap)
        out = x.new_zeros((b, self.out_channels, h * s, w * s))
        acc = x.new_zeros((1, 1, h * s, w * s))

        ramp = torch.linspace(0, 1, overlap * s, device=x.device) if overlap > 0 else None

        ys = list(range(0, max(1, h - tile + 1), stride))
        xs = list(range(0, max(1, w - tile + 1), stride))
        if ys[-1] != h - tile and h > tile:
            ys.append(h - tile)
        if xs[-1] != w - tile and w > tile:
            xs.append(w - tile)

        for y0 in ys:
            for x0 in xs:
                patch = x[..., y0:y0 + tile, x0:x0 + tile]
                pred = self.forward(patch, out_scale=s)

                ph, pw = pred.shape[-2:]
                win = pred.new_ones((1, 1, ph, pw))
                if ramp is not None:
                    n = min(ramp.numel(), ph // 2, pw // 2)
                    if n > 0:
                        r = ramp[:n]
                        win[..., :n, :] *= r.view(1, 1, n, 1)
                        win[..., -n:, :] *= r.flip(0).view(1, 1, n, 1)
                        win[..., :, :n] *= r.view(1, 1, 1, n)
                        win[..., :, -n:] *= r.flip(0).view(1, 1, 1, n)

                out[..., y0 * s:y0 * s + ph, x0 * s:x0 * s + pw] += pred * win
                acc[..., y0 * s:y0 * s + ph, x0 * s:x0 * s + pw] += win

        return out / acc.clamp(min=1e-8)


def build_model(cfg):
    """Instantiate from the `model` block of a config dict."""
    m = dict(cfg)
    fm = m.pop('frequency_mask', {}) or {}
    return SWIRAdaIRSR(
        inp_channels=m.get('inp_channels', 1),
        out_channels=m.get('out_channels', 1),
        dim=m.get('dim', 48),
        num_blocks=tuple(m.get('num_blocks', (4, 6, 6, 8))),
        num_refinement_blocks=m.get('num_refinement_blocks', 4),
        heads=tuple(m.get('heads', (1, 2, 4, 8))),
        ffn_expansion_factor=m.get('ffn_expansion_factor', 2.66),
        bias=m.get('bias', False),
        layernorm_type=m.get('layernorm_type', 'WithBias'),
        scales=tuple(m.get('scales', (1, 2))),
        gated_base=m.get('gated_base', True),
        mask_mode=fm.get('mode', 'soft_relative'),
        fre_tau=fm.get('tau', 0.05),
        anisotropic_mask=fm.get('anisotropic', False),
        use_frequency_modules=m.get('use_frequency_modules', True),
    )
