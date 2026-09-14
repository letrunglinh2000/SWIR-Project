"""NSLSR SWIR dataset and scale-homogeneous batching (spec SS6.1, SS9).

Key contracts:
  * HR is cropped first, then degraded -- LR and HR are never cropped
    independently (SS9.5).
  * Geometric augmentation is applied to the HR crop *before* degradation and is
    never mixed into the degradation module (SS9.6).
  * `out_scale` is supplied by the batch sampler, never sampled per example
    inside a collated batch (SS6.1, SS17.13).
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from utils.swir_degradation import EVAL_SUBSETS, build_pipeline
from utils.swir_io import list_images, read_image


def _augment(hr, rng, ops, allow_vflip):
    applied = []
    if 'hflip' in ops and rng.random() < 0.5:
        hr = np.ascontiguousarray(hr[:, ::-1]); applied.append('hflip')
    if 'vflip' in ops and allow_vflip and rng.random() < 0.5:
        hr = np.ascontiguousarray(hr[::-1, :]); applied.append('vflip')
    if 'rot90' in ops:
        k = int(rng.integers(0, 4))
        if k:
            hr = np.ascontiguousarray(np.rot90(hr, k)); applied.append('rot90x%d' % k)
    return hr, applied


class NSLSRSWIRDataset(Dataset):
    """Returns HR/LR pairs for one requested output scale."""

    def __init__(self, root, split, data_profile, degradation_cfg,
                 modality='SWIR', hr_patch=256, augment=True,
                 augment_ops=('hflip', 'vflip', 'rot90'), allow_vflip=True,
                 cache_images=True, seed=1234, full_frame=False,
                 eval_subset=None, splits_json=None, shared_cache=None):
        super(NSLSRSWIRDataset, self).__init__()

        if splits_json:
            # Leakage-aware manifest from tools/make_splits.py. The shipped
            # folders are NOT an independent split: NSLSR_test is a subset of
            # NSLSR_val and both stride the same capture sequence (spec SS9.3).
            with open(splits_json) as fh:
                manifest = json.load(fh)
            if split not in manifest['files']:
                raise ValueError('split {!r} not in {} (have {})'.format(
                    split, splits_json, sorted(manifest['files'])))
            self.dir = root
            self.paths = [os.path.join(root, rel) for rel in manifest['files'][split]]
            self.split_scheme = manifest.get('scheme', 'unknown')
        else:
            self.dir = os.path.join(root, split, modality)
            self.paths = list_images(self.dir)
            self.split_scheme = 'shipped_folders'
        self.profile = data_profile
        self.pipeline = build_pipeline(degradation_cfg, data_profile)

        self.hr_patch = int(hr_patch)
        self.augment = bool(augment)
        self.augment_ops = tuple(augment_ops)
        self.allow_vflip = bool(allow_vflip)
        self.seed = int(seed)
        self.full_frame = bool(full_frame)
        self.epoch = 0

        if eval_subset is not None and eval_subset not in EVAL_SUBSETS:
            raise ValueError('Unknown eval subset {!r}; choose from {}'.format(
                eval_subset, sorted(EVAL_SUBSETS)))
        self.eval_subset = eval_subset

        # Cache is keyed by path and may be shared between datasets. The frozen
        # per-degradation validation sets are all built over the SAME frames, so
        # giving each its own list would hold ~14 copies of the split in RAM.
        if cache_images:
            self.cache = {} if shared_cache is None else shared_cache
            for path in self.paths:
                if path not in self.cache:
                    self.cache[path] = read_image(path, data_profile)
        else:
            self.cache = None

    # -- helpers ------------------------------------------------------------- #

    def set_epoch(self, epoch):
        """Training only: makes crops/degradations vary across epochs."""
        self.epoch = int(epoch)

    def _hr(self, i):
        path = self.paths[i]
        if self.cache is not None:
            return self.cache[path]
        return read_image(path, self.profile)

    def _crop_seed(self, i):
        """Crop + augmentation stream. Deliberately independent of out_scale, so
        index i yields the SAME HR content at s=1 and s=2 -- otherwise the
        scale-sharing comparison (E6) would be confounded by different content."""
        ss = np.random.SeedSequence([self.seed, self.epoch, int(i), 0])
        return int(ss.generate_state(1, dtype=np.uint32)[0])

    def _degradation_seed(self, i, out_scale):
        """Degradation stream; scale-dependent because D_s differs by scale."""
        ss = np.random.SeedSequence([self.seed, self.epoch, int(i), int(out_scale)])
        return int(ss.generate_state(1, dtype=np.uint32)[0])

    @staticmethod
    def _crop_multiple(img, base):
        h, w = img.shape[:2]
        return img[:h - h % base, :w - w % base]

    # -- item ---------------------------------------------------------------- #

    def get_item(self, i, out_scale):
        out_scale = int(out_scale)
        rng = np.random.default_rng(self._crop_seed(i))
        seed = self._degradation_seed(i, out_scale)

        hr = self._hr(i)

        if self.full_frame:
            # Full-frame evaluation: HR must stay divisible by 16*out_scale so the
            # LR is an exact 1/s of it and both are network-compatible.
            hr = self._crop_multiple(hr, 16 * out_scale)
        else:
            p = self.hr_patch
            if p % (16 * out_scale):
                raise ValueError('hr_patch {} must be a multiple of {} for s={}'.format(
                    p, 16 * out_scale, out_scale))
            H, W = hr.shape[:2]
            if H < p or W < p:
                raise ValueError('hr_patch {} exceeds image {}x{}'.format(p, H, W))
            top = int(rng.integers(0, H - p + 1))
            left = int(rng.integers(0, W - p + 1))
            hr = hr[top:top + p, left:left + p]

            if self.augment:
                hr, _ = _augment(hr, rng, self.augment_ops, self.allow_vflip)

        force = EVAL_SUBSETS.get(self.eval_subset) if self.eval_subset else None
        lr, meta = self.pipeline(hr, out_scale, seed, force_flags=force)

        meta['name'] = os.path.basename(self.paths[i]).rsplit('.', 1)[0]
        if self.eval_subset:
            meta['eval_subset'] = self.eval_subset

        return {
            'lr': torch.from_numpy(lr.copy()).unsqueeze(0),
            'hr': torch.from_numpy(np.ascontiguousarray(hr)).unsqueeze(0),
            'scale': out_scale,
            'name': meta['name'],
            'meta': meta,
        }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        raise RuntimeError(
            'NSLSRSWIRDataset needs an explicit out_scale. Wrap it in FixedScaleView '
            'or iterate with ScaleHomogeneousBatchSampler (spec SS6.1).')


class FixedScaleView(Dataset):
    """A single-scale view, safe for ordinary tensor collation."""

    def __init__(self, dataset, out_scale):
        self.dataset = dataset
        self.out_scale = int(out_scale)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset.get_item(i, self.out_scale)


class ScaleHomogeneousBatchSampler(Sampler):
    """Yields batches of (index, scale) with ONE scale per batch (spec SS6.1).

    Under DDP every rank must choose the SAME scale on the same step: if one rank
    runs a 128x128 s=2 batch while another runs a 256x256 s=1 batch, the backward
    shapes diverge and the gradient all-reduce desynchronises. That is guaranteed
    structurally here rather than by coincidence -- all ranks drive one RNG seeded
    only by (seed, epoch), and the scale is drawn from it BEFORE the global batch
    is sharded, so the draw cannot depend on rank.

    Each rank then takes a disjoint stride of the global batch, so the union of
    ranks covers it exactly once and every rank yields the same number of steps.
    """

    def __init__(self, n_items, batch_size, p_s1=0.5, p_s2=0.5, seed=1234,
                 drop_last=True, scales=(1, 2), rank=0, world_size=1):
        self.n_items = int(n_items)
        self.batch_size = int(batch_size)
        self.scales = tuple(scales)
        probs = {1: p_s1, 2: p_s2}
        total = sum(probs[s] for s in self.scales)
        if total <= 0:
            raise ValueError('scale probabilities sum to zero')
        self.probs = [probs[s] / total for s in self.scales]
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError('rank {} outside world_size {}'.format(rank, world_size))
        self.epoch = 0

    @property
    def global_batch(self):
        """Samples consumed per optimizer step across all ranks."""
        return self.batch_size * self.world_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        g = self.global_batch
        return self.n_items // g if self.drop_last else -(-self.n_items // g)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        order = rng.permutation(self.n_items)
        g = self.global_batch

        for b in range(len(self)):
            # Drawn before sharding, from a rank-independent stream.
            scale = int(rng.choice(self.scales, p=self.probs))

            chunk = order[b * g:(b + 1) * g]
            mine = chunk[self.rank::self.world_size]
            if self.drop_last and len(mine) < self.batch_size:
                break
            yield [(int(i), scale) for i in mine]


class ScaleBatchDataset(Dataset):
    """Adapter so a (index, scale) pair from the sampler reaches `get_item`."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        i, scale = key
        return self.dataset.get_item(i, scale)


def collate(batch):
    scales = {b['scale'] for b in batch}
    if len(scales) != 1:
        raise RuntimeError('Mixed scales in one batch: {} (violates INV-3/SS6.1)'.format(scales))
    return {
        'lr': torch.stack([b['lr'] for b in batch]),
        'hr': torch.stack([b['hr'] for b in batch]),
        'scale': scales.pop(),
        'name': [b['name'] for b in batch],
        'meta': [b['meta'] for b in batch],
    }
