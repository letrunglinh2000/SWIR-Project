"""HR-crop -> aligned-LR contract (spec SS9.5, M2)."""

import copy

import cv2
import numpy as np
import pytest
import torch
import yaml

from utils.swir_io import load_data_profile
from utils.swir_sr_dataset import (NSLSRSWIRDataset, ScaleBatchDataset,
                                   ScaleHomogeneousBatchSampler, collate)

# Read the root from the config rather than hard-coding it: the active tree can be
# switched (e.g. to the single-band PNG re-container) and a pinned path here would
# silently disagree with configs/data_profile.yaml.
ROOT = yaml.safe_load(open('configs/swir_base.yaml'))['data']['root']
SPLIT = 'NSLSR_train'


def _clean_cfg(base):
    """All degradations off and quantisation off: D_s reduces to S_s alone."""
    cfg = copy.deepcopy(base)
    cfg['probabilities'] = {'blur_p': 0.0, 'noise_p': 0.0,
                            'low_light_p': 0.0, 'backscatter_p': 0.0}
    cfg['quantization'] = {'mode': 'off'}
    cfg['optics']['downsample_interp'] = ['area']
    return cfg


@pytest.fixture(scope='module')
def base_cfg():
    return yaml.safe_load(open('configs/swir_base.yaml'))['degradation']


def _dataset(cfg, **kw):
    return NSLSRSWIRDataset(ROOT, SPLIT, load_data_profile(), cfg,
                            hr_patch=256, cache_images=False, seed=1234, **kw)


def test_s1_identity_when_no_degradation(base_cfg):
    ds = _dataset(_clean_cfg(base_cfg), augment=False)
    item = ds.get_item(0, 1)
    assert torch.allclose(item['lr'], item['hr'], atol=1e-6)


def test_s2_lr_is_the_aligned_downsample_of_hr(base_cfg):
    ds = _dataset(_clean_cfg(base_cfg), augment=False)
    item = ds.get_item(0, 2)
    hr = item['hr'][0].numpy()
    expected = cv2.resize(hr, (hr.shape[1] // 2, hr.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
    assert np.allclose(item['lr'][0].numpy(), expected, atol=1e-6)


def test_hr_crop_is_identical_across_scales(base_cfg):
    """The same index must yield the same HR crop whichever scale is asked for."""
    ds = _dataset(_clean_cfg(base_cfg), augment=False)
    assert torch.allclose(ds.get_item(3, 1)['hr'], ds.get_item(3, 2)['hr'])


def test_shapes_follow_the_patch_rule(base_cfg):
    ds = _dataset(base_cfg)
    assert ds.get_item(0, 2)['lr'].shape[-2:] == (128, 128)
    assert ds.get_item(0, 2)['hr'].shape[-2:] == (256, 256)
    assert ds.get_item(0, 1)['lr'].shape[-2:] == (256, 256)


def test_getitem_without_scale_is_refused(base_cfg):
    with pytest.raises(RuntimeError):
        _ = _dataset(base_cfg)[0]


def test_batches_are_scale_homogeneous(base_cfg):
    """SS6.1 / SS17.13: a batch never mixes s=1 and s=2 spatial sizes."""
    ds = _dataset(base_cfg)
    sampler = ScaleHomogeneousBatchSampler(len(ds), batch_size=4, seed=7)
    loader = torch.utils.data.DataLoader(ScaleBatchDataset(ds), batch_sampler=sampler,
                                         collate_fn=collate, num_workers=0)
    seen = set()
    for i, batch in enumerate(loader):
        s = batch['scale']
        seen.add(s)
        assert batch['hr'].shape[-1] == s * batch['lr'].shape[-1]
        assert len({m['scale'] for m in batch['meta']}) == 1
        if i >= 30:
            break
    assert seen == {1, 2}, seen


def test_collate_rejects_mixed_scales(base_cfg):
    ds = _dataset(base_cfg)
    with pytest.raises(RuntimeError):
        collate([ds.get_item(0, 1), ds.get_item(1, 2)])


def test_epoch_changes_the_crop(base_cfg):
    ds = _dataset(base_cfg)
    a = ds.get_item(0, 2)['hr']
    ds.set_epoch(1)
    b = ds.get_item(0, 2)['hr']
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# DDP: ranks must agree on the scale and partition the indices (spec SS6.1)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('world_size', [2, 4])
def test_ranks_agree_on_scale_every_step(world_size):
    samplers = [ScaleHomogeneousBatchSampler(256, 4, seed=7, rank=r, world_size=world_size)
                for r in range(world_size)]
    runs = [list(s) for s in samplers]

    assert len({len(r) for r in runs}) == 1, 'ranks yield different step counts'
    for step, batches in enumerate(zip(*runs)):
        scales = {s for b in batches for _, s in b}
        assert len(scales) == 1, 'step {} mixed scales across ranks: {}'.format(step, scales)


@pytest.mark.parametrize('world_size', [2, 4])
def test_ranks_see_disjoint_indices(world_size):
    samplers = [ScaleHomogeneousBatchSampler(256, 4, seed=7, rank=r, world_size=world_size)
                for r in range(world_size)]
    runs = [list(s) for s in samplers]

    for batches in zip(*runs):
        seen = [ {i for i, _ in b} for b in batches ]
        union = set().union(*seen)
        assert len(union) == sum(len(x) for x in seen), 'ranks share indices in one step'
    for b in runs[0]:
        assert len(b) == 4, 'per-rank batch size must stay the configured value'


def test_single_process_behaviour_unchanged():
    s = ScaleHomogeneousBatchSampler(64, 4, seed=7)
    batches = list(s)
    assert s.world_size == 1 and s.global_batch == 4
    assert len(batches) == 16 and all(len(b) == 4 for b in batches)


def test_rank_outside_world_is_rejected():
    with pytest.raises(ValueError):
        ScaleHomogeneousBatchSampler(64, 4, rank=3, world_size=2)
