"""Train SWIRAdaIRSR on NSLSR (spec SS10).

Config-driven, plain PyTorch. One optimizer batch carries exactly one output
scale (INV-3 / SS6.1). Everything needed to reproduce a number is written into
the run directory (SS10.3, SS17.6).
"""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from net.swir_adair_sr import build_model
from utils.swir_io import load_data_profile
from utils.swir_losses import CompositeLoss
from utils.swir_metrics import AverageMeter, compute_metrics, metric_conventions
from utils.swir_sr_dataset import (FixedScaleView, NSLSRSWIRDataset, ScaleBatchDataset,
                                   ScaleHomogeneousBatchSampler, collate)
from utils.schedulers import LinearWarmupCosineAnnealingLR


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def git_commit():
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
        dirty = subprocess.call(['git', 'diff', '--quiet']) != 0
        return out.decode().strip() + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def file_sha(path):
    import hashlib
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def environment(cfg, profile, model):
    return {
        'commit': git_commit(),
        'seed': cfg['experiment']['seed'],
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'split_manifest': cfg['data'].get('splits_json'),
        'split_manifest_sha256': file_sha(cfg['data'].get('splits_json')),
        'data_profile_sha256': file_sha(cfg['data']['data_profile']),
        'encoding': profile['encoding'],
        'params_M': sum(p.numel() for p in model.parameters()) / 1e6,
        'metric_conventions': metric_conventions(),
    }


def setup_distributed():
    """Initialise DDP when launched by torchrun; otherwise stay single-process."""
    if 'LOCAL_RANK' not in os.environ:
        return False, 0, 1, 0
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def assert_scale_synchronised(scale, device):
    """Every rank must be on the same output scale this step.

    Mixed scales across ranks means mismatched backward shapes and a desynchronised
    gradient all-reduce -- which shows up as a hang or as silently wrong gradients,
    not as a clean error. One tiny all-reduce per step is cheap insurance.
    """
    probe = torch.tensor([scale, -scale], device=device, dtype=torch.int64)
    dist.all_reduce(probe, op=dist.ReduceOp.MAX)
    hi, lo = int(probe[0]), -int(probe[1])
    if hi != lo:
        raise RuntimeError(
            'Ranks disagree on out_scale this step (saw {} and {}). The batch sampler '
            'must draw the scale from a rank-independent stream (spec SS6.1).'.format(lo, hi))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def build_train_dataset(cfg, profile):
    d = cfg['data']
    return NSLSRSWIRDataset(
        root=d['root'], split=d['train_split'], data_profile=profile,
        degradation_cfg=cfg['degradation'], modality=d['modality'],
        hr_patch=d['hr_patch'], augment=d['augment'],
        augment_ops=tuple(d['augment_ops']),
        allow_vflip=d.get('vflip_with_horizon_backscatter', False)
        or cfg['degradation'].get('backscatter', {}).get('t_field') != 'horizon',
        cache_images=d.get('cache_images', True),
        seed=cfg['experiment']['seed'], splits_json=d.get('splits_json'))


def build_val_datasets(cfg, profile, degradation_cfg, subsets, scales):
    """One frozen dataset per (subset, scale) (spec SS11.2).

    `eval.val_full_frame` controls the training-time validation resolution. The
    resolution is recorded in env.json and printed, because INV-7 requires every
    table to state it. Final numbers always come from test_swir.py at full frame.
    """
    d = cfg['data']
    full = cfg['eval'].get('val_full_frame', True)
    crop = cfg['eval'].get('val_crop', d['hr_patch'])
    out = {}
    shared = {}          # every subset/scale view reads the same frames
    for subset in subsets:
        for s in scales:
            out[(subset, s)] = NSLSRSWIRDataset(
                root=d['root'], split=d['val_split'], data_profile=profile,
                degradation_cfg=degradation_cfg, modality=d['modality'],
                hr_patch=crop, augment=False,
                cache_images=d.get('cache_images', True),
                seed=cfg['experiment']['seed'], full_frame=full,
                eval_subset=subset, splits_json=d.get('splits_json'),
                shared_cache=shared)
    return out


@torch.no_grad()
def validate(model, val_sets, device, max_images, metrics):
    model.eval()
    results = {}
    for (subset, scale), ds in val_sets.items():
        n = len(ds) if max_images <= 0 else min(max_images, len(ds))
        loader = DataLoader(FixedScaleView(ds, scale), batch_size=1, shuffle=False,
                            num_workers=0, sampler=list(range(n)), collate_fn=collate)
        meters = {}
        for batch in loader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            pred = model(lr, out_scale=scale).clamp(0, 1)
            for k, v in compute_metrics(pred, hr, metrics).items():
                meters.setdefault(k, AverageMeter()).update(v, lr.size(0))
        results[(subset, scale)] = {k: m.avg for k, m in meters.items()}
    model.train()
    return results


# --------------------------------------------------------------------------- #
# checkpoint / init
# --------------------------------------------------------------------------- #

def load_pretrained_partial(model, path, rgb_to_gray='mean', reinit_rate_conv=True,
                            verbose=True):
    """Spec SS10.1: partial load with a printed key report. strict=False without
    an inspected report is forbidden (SS17.7)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    src = ckpt.get('state_dict', ckpt.get('model', ckpt))
    src = {k.replace('net.', '', 1) if k.startswith('net.') else k: v for k, v in src.items()}

    tgt = model.state_dict()
    loaded, skipped, adapted = [], [], []

    for k, v in src.items():
        if k not in tgt:
            skipped.append((k, 'absent in target'))
            continue
        if tgt[k].shape == v.shape:
            tgt[k] = v
            loaded.append(k)
        elif (v.dim() == 4 and tgt[k].dim() == 4 and v.shape[1] == 3
              and tgt[k].shape[1] == 1 and v.shape[0] == tgt[k].shape[0]):
            # RGB -> single-band input projection: W_gray = mean_c W_rgb
            tgt[k] = v.mean(dim=1, keepdim=True) if rgb_to_gray == 'mean' else v[:, :1]
            adapted.append(k)
        else:
            skipped.append((k, 'shape {} vs {}'.format(tuple(v.shape), tuple(tgt[k].shape))))

    if reinit_rate_conv:
        # Soft-FMiM changes the mask semantics, so the cutoff predictor is not
        # inherited from an upstream integer-mask checkpoint (spec SS10.1).
        fresh_state = model.state_dict()
        for k in [k for k in tgt if '.rate_conv.' in k]:
            tgt[k] = fresh_state[k]
            if k in loaded:
                loaded.remove(k)
            if k in adapted:
                adapted.remove(k)

    missing = [k for k in tgt if k not in loaded and k not in adapted]
    model.load_state_dict(tgt)

    if verbose:
        print('[init] pretrained-partial from {}'.format(path))
        print('[init]   loaded  {} tensors'.format(len(loaded)))
        print('[init]   adapted {} tensors (RGB->gray): {}'.format(len(adapted), adapted))
        print('[init]   skipped {} source keys'.format(len(skipped)))
        for k, why in skipped[:20]:
            print('[init]     - {}: {}'.format(k, why))
        print('[init]   randomly initialised {} target tensors (heads, base gate, '
              'rate_conv): {}'.format(len(missing), missing[:20]))
    return {'loaded': len(loaded), 'adapted': adapted,
            'skipped': [k for k, _ in skipped], 'random_init': missing}


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, cfg):
    torch.save({'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'best': best, 'config': cfg, 'commit': git_commit()}, path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description='Train SWIRAdaIRSR (spec SS10)')
    ap.add_argument('--config', default='configs/swir_base.yaml')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--resume', default='')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--max_steps', type=int, default=0, help='debug: stop after N steps')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs

    distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    out_dir = args.out_dir or cfg['experiment']['out_dir']
    if is_main:
        os.makedirs(out_dir, exist_ok=True)

    # Identical on every rank: the whole data path is index-seeded, so this keeps
    # runs reproducible without making ranks draw the same degradations.
    set_seed(cfg['experiment']['seed'])
    device = (torch.device('cuda', local_rank) if distributed
              else torch.device(args.device))
    profile = load_data_profile(cfg['data']['data_profile'])

    model = build_model(cfg['model']).to(device)

    init_cfg = cfg['train'].get('init', {})
    init_report = None
    if init_cfg.get('mode') == 'pretrained_partial' and init_cfg.get('checkpoint'):
        # Every rank loads the same weights; only rank 0 prints the key report.
        init_report = load_pretrained_partial(
            model, init_cfg['checkpoint'], init_cfg.get('rgb_to_gray', 'mean'),
            init_cfg.get('reinit_rate_conv', True), verbose=is_main)
    elif is_main:
        print('[init] training from scratch (ablation A5 arm)')

    trainset = build_train_dataset(cfg, profile)
    scales = tuple(cfg['model']['scales'])
    sampler = ScaleHomogeneousBatchSampler(
        len(trainset), cfg['train']['batch_size'],
        p_s1=cfg['scale_sampling']['p_s1'], p_s2=cfg['scale_sampling']['p_s2'],
        seed=cfg['experiment']['seed'], scales=scales,
        rank=rank, world_size=world_size)
    loader = DataLoader(ScaleBatchDataset(trainset), batch_sampler=sampler,
                        collate_fn=collate, num_workers=cfg['train']['num_workers'],
                        pin_memory=True, worker_init_fn=worker_init)

    val_sets = build_val_datasets(cfg, profile, cfg['degradation'],
                                 cfg['eval']['fixed_subsets'], scales)

    loss_fn = CompositeLoss(cfg.get('loss'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'],
                                  weight_decay=cfg['train']['weight_decay'])

    core = model
    if distributed:
        # Only the selected output head runs on a given step, so the other head's
        # parameters legitimately receive no gradient (spec SS19.1).
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        find_unused_parameters=True)

    iters = max(1, len(sampler))
    # Step per iteration: LinearWarmupCosineAnnealingLR starts at warmup_start_lr=0,
    # so epoch-level stepping would pin the entire first epoch at lr=0.
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=cfg['train']['warmup_epochs'] * iters,
        max_epochs=cfg['train']['epochs'] * iters)
    amp = bool(cfg['train'].get('amp', False))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    env = environment(cfg, profile, core)
    env['loss'] = loss_fn.describe()
    env['init'] = init_report
    env['train_frames'] = len(trainset)
    env['val_resolution'] = ('full_frame' if cfg['eval'].get('val_full_frame', True)
                             else 'crop_{}'.format(cfg['eval'].get('val_crop', cfg['data']['hr_patch'])))
    env['split_scheme'] = trainset.split_scheme
    env['world_size'] = world_size
    env['per_gpu_batch'] = cfg['train']['batch_size']
    env['effective_batch'] = cfg['train']['batch_size'] * world_size
    env['steps_per_epoch'] = len(sampler)
    if is_main:
        with open(os.path.join(out_dir, 'env.json'), 'w') as fh:
            json.dump(env, fh, indent=2, default=str)
        with open(os.path.join(out_dir, 'config.yaml'), 'w') as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        print('[env]', json.dumps({k: env[k] for k in
                                   ('commit', 'seed', 'torch', 'gpu', 'params_M',
                                    'train_frames', 'split_scheme', 'world_size',
                                    'effective_batch', 'steps_per_epoch')}, default=str))

    start_epoch, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        core.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch, best = ck['epoch'] + 1, ck.get('best', -1.0)
        if is_main:
            print('[resume] from {} at epoch {}'.format(args.resume, start_epoch))

    log_path = os.path.join(out_dir, 'metrics.csv')
    val_keys = [(s, sc) for s in cfg['eval']['fixed_subsets'] for sc in scales]
    if is_main and not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as fh:
            csv.writer(fh).writerow(
                ['epoch', 'lr', 'train_loss'] +
                ['{}_s{}_{}'.format(s, sc, m) for s, sc in val_keys for m in ('psnr', 'ssim')])

    step = 0
    for epoch in range(start_epoch, cfg['train']['epochs']):
        trainset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()

        meter = AverageMeter()
        scale_counts = {s: 0 for s in scales}
        t0 = time.time()

        for batch in loader:
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            scale = batch['scale']
            scale_counts[scale] += 1
            if distributed:
                assert_scale_synchronised(scale, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp, dtype=torch.bfloat16):
                pred = model(lr_img, out_scale=scale)
                loss, terms = loss_fn(pred, hr_img)

            scaler.scale(loss).backward()
            if cfg['train'].get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            meter.update(terms['total'], lr_img.size(0))
            step += 1
            if is_main and step % 50 == 0:
                print('epoch {:3d} step {:6d} | loss {:.4f} | lr {:.2e} | s={}'.format(
                    epoch, step, meter.avg, optimizer.param_groups[0]['lr'], scale), flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        if is_main:
            print('epoch {:3d} done in {:.1f}s | loss {:.4f} | batches per scale {}'.format(
                epoch, time.time() - t0, meter.avg, scale_counts), flush=True)

        results = {}
        do_val = (epoch + 1) % cfg['train']['val_every'] == 0 or args.max_steps
        if do_val and is_main:
            results = validate(core, val_sets, device, cfg['eval']['val_images'],
                               cfg['eval']['metrics'])
            for (s, sc), r in results.items():
                print('  val {:<16} s={} PSNR {:.2f} SSIM {:.4f}'.format(
                    s, sc, r['psnr'], r['ssim']), flush=True)
            sel = float(np.mean([r['psnr'] for r in results.values()]))
            print('  val selection metric (mean PSNR over subsets x scales): {:.3f}'.format(sel))
            if sel > best:
                best = sel
                save_checkpoint(os.path.join(out_dir, 'best.pth'), core, optimizer,
                                scheduler, epoch, best, cfg)
                print('  new best -> best.pth', flush=True)

        if distributed and do_val:
            # Other ranks idle through rank 0's validation; rejoin before the next epoch.
            dist.barrier()

        if is_main:
            # `core`, not `model`: a DDP wrapper would write module.-prefixed keys
            # that test_swir.py could not load.
            save_checkpoint(os.path.join(out_dir, 'latest.pth'), core, optimizer,
                            scheduler, epoch, best, cfg)
            with open(log_path, 'a', newline='') as fh:
                row = [epoch, optimizer.param_groups[0]['lr'], meter.avg]
                for key in val_keys:
                    r = results.get(key, {})
                    row += [r.get('psnr', ''), r.get('ssim', '')]
                csv.writer(fh).writerow(row)

        if args.max_steps and step >= args.max_steps:
            break

    if is_main:
        print('done. best selection metric {:.3f}'.format(best))
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
