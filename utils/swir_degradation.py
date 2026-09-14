"""Modular SWIR degradation operator D_s (spec SS7).

    D_s = Q o D_noise o S_s o D_optics o D_scatter o D_illum ,
    S_1 = Identity,  S_2 = downsample_2

So D_2 contains *exactly one* x2 spatial reduction and D_1 contains none. Never
compose an external downsample around this operator (spec SS6).

Every component is independently switchable and draws from its own RNG stream,
so a fixed seed reproduces the same phi and the same output byte-for-byte.
`__call__` returns a `meta` dict recording every flag and sampled parameter, for
stratified evaluation and logging only -- it is never fed to the model (INV-2).

Kernel note: spec SS7.4 asks that LSFDNet's vendored Real-ESRGAN kernel utilities
be reused. That repository is not present in this environment, so the isotropic /
anisotropic Gaussian and motion kernels are implemented here directly. Cite
Real-ESRGAN (Wang et al., ICCVW 2021) for the degradation-family design.
[CITATION REQUIRED: exact publisher record before use in the paper.]
"""

import numpy as np

SCALE_VALUES = (1, 2)

# Frozen per-degradation evaluation subsets (spec SS11.2). `composite` leaves the
# flags to the configured probabilities; every other subset pins them.
EVAL_SUBSETS = {
    'sr_only':         {'low_light': False, 'backscatter': False, 'blur': False, 'noise': False},
    'noise':           {'low_light': False, 'backscatter': False, 'blur': False, 'noise': True},
    'low_light':       {'low_light': True,  'backscatter': False, 'blur': False, 'noise': False},
    'backscatter':     {'low_light': False, 'backscatter': True,  'blur': False, 'noise': False},
    'blur':            {'low_light': False, 'backscatter': False, 'blur': True,  'noise': False},
    'low_light_noise': {'low_light': True,  'backscatter': False, 'blur': False, 'noise': True},
    'composite':       None,
}

# Component order is part of the contract; do not reorder without updating SS7.1.
COMPONENT_ORDER = ('illum', 'scatter', 'optics', 'scale', 'noise', 'quant')


class RNGBundle:
    """One independent generator per component, derived from a single seed."""

    def __init__(self, seed):
        streams = np.random.SeedSequence(seed).spawn(len(COMPONENT_ORDER) + 1)
        self.flags = np.random.default_rng(streams[0])
        for name, ss in zip(COMPONENT_ORDER, streams[1:]):
            setattr(self, name, np.random.default_rng(ss))


# --------------------------------------------------------------------------- #
# blur kernels
# --------------------------------------------------------------------------- #

def _gaussian_kernel(ksize, sigma_x, sigma_y, theta):
    ax = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    c, s = np.cos(theta), np.sin(theta)
    xr = c * xx + s * yy
    yr = -s * xx + c * yy
    k = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    return k / k.sum()


def _motion_kernel(ksize, length, theta):
    k = np.zeros((ksize, ksize), dtype=np.float64)
    c = (ksize - 1) / 2.0
    n = max(2, int(round(length)))
    for t in np.linspace(-length / 2.0, length / 2.0, n * 4):
        x = int(round(c + t * np.cos(theta)))
        y = int(round(c + t * np.sin(theta)))
        if 0 <= x < ksize and 0 <= y < ksize:
            k[y, x] += 1.0
    if k.sum() == 0:
        k[int(c), int(c)] = 1.0
    return k / k.sum()


def sample_kernel(rng, cfg):
    """Draw one blur kernel from the configured family."""
    families = cfg.get('kernel_families', ['iso', 'aniso', 'motion'])
    weights = cfg.get('kernel_weights', None)
    family = rng.choice(families, p=weights)

    ksize = int(cfg.get('kernel_size', 21))
    if ksize % 2 == 0:
        ksize += 1
    lo, hi = cfg.get('sigma_range', [0.2, 2.0])

    if family == 'iso':
        sigma = rng.uniform(lo, hi)
        return _gaussian_kernel(ksize, sigma, sigma, 0.0), {
            'family': 'iso', 'sigma_x': float(sigma), 'sigma_y': float(sigma), 'theta': 0.0}
    if family == 'aniso':
        sx, sy = rng.uniform(lo, hi), rng.uniform(lo, hi)
        theta = rng.uniform(0, np.pi)
        return _gaussian_kernel(ksize, sx, sy, theta), {
            'family': 'aniso', 'sigma_x': float(sx), 'sigma_y': float(sy), 'theta': float(theta)}
    if family == 'motion':
        lo_l, hi_l = cfg.get('motion_length_range', [3.0, 9.0])
        length = rng.uniform(lo_l, hi_l)
        theta = rng.uniform(0, np.pi)
        return _motion_kernel(ksize, length, theta), {
            'family': 'motion', 'length': float(length), 'theta': float(theta)}
    raise ValueError('Unknown kernel family {!r}'.format(family))


def _filter2d(img, kernel):
    import cv2
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #

def apply_illumination(x, rng, cfg):
    """x_L = clip(alpha * x**gamma, 0, 1)   (spec SS7.2)"""
    alpha = rng.uniform(*cfg.get('alpha_range', [0.15, 0.8]))
    gamma = rng.uniform(*cfg.get('gamma_range', [1.0, 2.2]))
    out = np.clip(alpha * np.power(np.clip(x, 0, 1), gamma), 0.0, 1.0)
    return out, {'alpha': float(alpha), 'gamma': float(gamma)}


def _transmission_field(shape, rng, cfg):
    """t = exp(-beta*d); NSLSR has no depth map, so d is one of three proxies."""
    h, w = shape
    model = cfg.get('t_field', 'horizon')
    beta = rng.uniform(*cfg.get('beta_range', [0.4, 1.8]))

    if model == 'constant':
        d = np.ones((h, w), dtype=np.float32)
    elif model == 'smooth':
        # low-frequency random field, upsampled from a coarse grid
        import cv2
        coarse = rng.random((4, 4)).astype(np.float32)
        d = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        d = np.clip(d, 0, 1)
    elif model == 'horizon':
        # maritime geometry: path length grows towards the horizon row
        horizon = rng.uniform(*cfg.get('horizon_range', [0.25, 0.6]))
        rows = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        d = np.exp(-((rows - horizon) ** 2) / (2 * cfg.get('horizon_sigma', 0.25) ** 2))
        d = np.repeat(d, w, axis=1)
    else:
        raise ValueError('Unknown t_field {!r}'.format(model))

    t = np.exp(-beta * d).astype(np.float32)
    return t, {'t_field': model, 'beta': float(beta)}


def apply_backscatter(x, rng, cfg, alpha=1.0):
    """I = J*t + A_eff*(1-t), A_eff = alpha**kappa * A_base  (spec SS7.3).

    Physics-inspired and UNCALIBRATED. This is an atmospheric veiling
    approximation, not a validated active-laser backscatter model.
    """
    t, meta = _transmission_field(x.shape[-2:], rng, cfg)
    a_base = rng.uniform(*cfg.get('A_base_range', [0.5, 1.0]))
    kappa = rng.uniform(*cfg.get('exposure_coupling_kappa', [0.0, 1.0]))
    a_eff = (alpha ** kappa) * a_base

    out = np.clip(x * t + a_eff * (1.0 - t), 0.0, 1.0)
    meta.update({'A_base': float(a_base), 'kappa': float(kappa), 'A_eff': float(a_eff),
                 'calibrated': False})
    return out, meta


def apply_optics(x, rng, cfg):
    kernel, meta = sample_kernel(rng, cfg)
    return np.clip(_filter2d(x, kernel.astype(np.float32)), 0.0, 1.0), meta


def apply_scale(x, out_scale, rng, cfg):
    """S_s: identity for s=1, exactly one x2 reduction for s=2 (spec SS7.1)."""
    if out_scale == 1:
        return x, {'resize': 'identity'}
    if out_scale != 2:
        raise ValueError('out_scale must be 1 or 2, got {}'.format(out_scale))

    import cv2
    # Never bicubic-only (spec SS7.4): the decimation filter is randomised too.
    choices = cfg.get('downsample_interp', ['area', 'bilinear', 'bicubic'])
    name = rng.choice(choices)
    interp = {'area': cv2.INTER_AREA, 'bilinear': cv2.INTER_LINEAR,
              'bicubic': cv2.INTER_CUBIC, 'nearest': cv2.INTER_NEAREST}[name]

    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError('s=2 requires even HR dims, got {}x{}'.format(h, w))
    out = cv2.resize(x, (w // 2, h // 2), interpolation=interp)
    return np.clip(out, 0.0, 1.0), {'resize': name}


def apply_noise(x, rng, cfg):
    """v0 Gaussian (debug) | v1 Poisson-Gaussian | v2 + FPN (spec SS7.5)."""
    version = cfg.get('version', 'v1')
    meta = {'version': version}
    out = x.astype(np.float32)

    if version == 'v0':
        sigma = rng.uniform(*cfg.get('sigma_range', [0.005, 0.05]))
        out = out + rng.standard_normal(out.shape).astype(np.float32) * sigma
        meta['sigma'] = float(sigma)
    elif version in ('v1', 'v2'):
        g = rng.uniform(*cfg.get('gain_range', [80.0, 2000.0]))
        sigma_r = rng.uniform(*cfg.get('read_sigma_range', [0.002, 0.02]))
        out = rng.poisson(np.clip(out, 0, None) * g).astype(np.float32) / g
        out = out + rng.standard_normal(out.shape).astype(np.float32) * sigma_r
        meta.update({'gain': float(g), 'read_sigma': float(sigma_r)})

        if version == 'v2':
            h, w = out.shape[-2:]
            col_s = rng.uniform(*cfg.get('column_sigma_range', [0.002, 0.015]))
            row_s = rng.uniform(*cfg.get('row_sigma_range', [0.0005, 0.005]))
            fpn_s = rng.uniform(*cfg.get('fpn_sigma_range', [0.002, 0.012]))

            out = out + rng.standard_normal((1, w)).astype(np.float32) * col_s
            out = out + rng.standard_normal((h, 1)).astype(np.float32) * row_s

            import cv2
            coarse = rng.standard_normal((max(2, h // 32), max(2, w // 32))).astype(np.float32)
            fpn = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
            out = out + fpn * fpn_s
            meta.update({'column_sigma': float(col_s), 'row_sigma': float(row_s),
                         'fpn_sigma': float(fpn_s)})
    else:
        raise ValueError('Unknown noise version {!r}'.format(version))

    return np.clip(out, 0.0, 1.0), meta


def apply_quantization(x, rng, cfg, data_profile=None):
    """Re-quantise to a configured bit depth (spec SS7.1)."""
    mode = cfg.get('mode', 'source')
    if mode == 'off':
        return x, {'mode': 'off', 'bits': None}

    if mode == 'source':
        if data_profile is None:
            raise ValueError('quantization.mode=source needs a data_profile')
        bits = int(data_profile['encoding']['bit_depth'])
    elif mode == 'target_camera':
        bits = cfg.get('bits')
        if bits is None:
            raise ValueError('quantization.mode=target_camera requires quantization.bits')
        bits = int(bits)
    elif mode == 'random':
        bits = int(rng.choice(cfg.get('bits_choices', [8, 10, 12])))
    else:
        raise ValueError('Unknown quantization mode {!r}'.format(mode))

    levels = 2 ** bits - 1
    return np.round(np.clip(x, 0, 1) * levels) / levels, {'mode': mode, 'bits': bits}


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

class DegradationPipeline:
    """Composite operator with per-example flags (spec SS7.6)."""

    def __init__(self, cfg, data_profile=None):
        self.cfg = cfg or {}
        self.data_profile = data_profile
        self.name = self.cfg.get('name', 'sim_A')
        p = self.cfg.get('probabilities', {})
        self.p_blur = p.get('blur_p', 0.70)
        self.p_noise = p.get('noise_p', 0.60)
        self.p_low_light = p.get('low_light_p', 0.45)
        self.p_backscatter = p.get('backscatter_p', 0.35)

    def __call__(self, hr, out_scale, seed, force_flags=None):
        """hr: float32 [H,W] in [0,1]. Returns (lr, meta).

        `force_flags` pins individual components on/off, used to build the frozen
        per-degradation validation subsets of spec SS11.2. The flag RNG is drawn
        either way so that pinning one component does not shift the others.
        """
        if out_scale not in SCALE_VALUES:
            raise ValueError('out_scale must be 1 or 2, got {}'.format(out_scale))

        rng = RNGBundle(seed)
        x = np.ascontiguousarray(hr, dtype=np.float32)

        flags = {
            'low_light': bool(rng.flags.random() < self.p_low_light),
            'backscatter': bool(rng.flags.random() < self.p_backscatter),
            'blur': bool(rng.flags.random() < self.p_blur),
            'noise': bool(rng.flags.random() < self.p_noise),
        }
        if force_flags:
            unknown = set(force_flags) - set(flags)
            if unknown:
                raise ValueError('Unknown degradation flag(s): {}'.format(sorted(unknown)))
            flags.update({k: bool(v) for k, v in force_flags.items()})
        meta = {'sim': self.name, 'scale': int(out_scale), 'seed': int(seed), 'flags': flags}

        alpha = 1.0
        if flags['low_light']:
            x, m = apply_illumination(x, rng.illum, self.cfg.get('illumination', {}))
            alpha = m['alpha']
            meta['illumination'] = m

        if flags['backscatter']:
            x, m = apply_backscatter(x, rng.scatter, self.cfg.get('backscatter', {}), alpha=alpha)
            meta['backscatter'] = m

        if flags['blur']:
            x, m = apply_optics(x, rng.optics, self.cfg.get('optics', {}))
            meta['optics'] = m

        x, m = apply_scale(x, out_scale, rng.scale, self.cfg.get('optics', {}))
        meta['scale_op'] = m

        if flags['noise']:
            x, m = apply_noise(x, rng.noise, self.cfg.get('noise', {}))
            meta['noise'] = m

        x, m = apply_quantization(x, rng.quant, self.cfg.get('quantization', {}), self.data_profile)
        meta['quantization'] = m

        return np.clip(x, 0.0, 1.0).astype(np.float32), meta


def build_pipeline(cfg, data_profile=None):
    return DegradationPipeline(cfg, data_profile=data_profile)
