"""Evaluate a SWIRAdaIRSR checkpoint (spec SS11).

Always reports the per-degradation x per-scale x per-simulator breakdown;
aggregate-only PSNR is forbidden (SS17.8). sim_B results are SIMULATOR-SHIFT
robustness, never "external validity" (SS17.14).
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.swir_adair_sr import build_model
from utils.swir_io import load_data_profile, write_image
from utils.swir_metrics import AverageMeter, compute_metrics, metric_conventions
from utils.swir_sr_dataset import FixedScaleView, NSLSRSWIRDataset, collate


def load_checkpoint(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if 'config' not in ck:
        raise ValueError('{} has no embedded config; it cannot be reproduced '
                         '(spec SS17.6)'.format(path))
    model = build_model(ck['config']['model']).to(device).eval()
    model.load_state_dict(ck['model'])
    return model, ck


@torch.no_grad()
def evaluate(model, cfg, profile, degradation_cfg, split, subset, scale, device,
             limit=0, metrics=('psnr', 'ssim', 'edge'), lpips_backbone=None,
             out_dir=None, full_frame=True):
    d = cfg['data']
    ds = NSLSRSWIRDataset(root=d['root'], split=split, data_profile=profile,
                          degradation_cfg=degradation_cfg, modality=d['modality'],
                          hr_patch=d['hr_patch'], augment=False, cache_images=False,
                          seed=cfg['experiment']['seed'], full_frame=full_frame,
                          eval_subset=subset, splits_json=d.get('splits_json'))
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    loader = DataLoader(FixedScaleView(ds, scale), batch_size=1, shuffle=False,
                        num_workers=2, sampler=list(range(n)), collate_fn=collate)

    meters = {}
    for batch in tqdm(loader, desc='{}/s{}'.format(subset, scale), leave=False):
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        pred = model(lr, out_scale=scale).clamp(0, 1)

        for k, v in compute_metrics(pred, hr, metrics, lpips_backbone).items():
            meters.setdefault(k, AverageMeter()).update(v, lr.size(0))

        if out_dir:
            name = batch['name'][0]
            write_image(pred[0].cpu().numpy(),
                        os.path.join(out_dir, subset, 's{}'.format(scale), name + '.png'))

    return {k: m.avg for k, m in meters.items()}, n


def main():
    ap = argparse.ArgumentParser(description='Evaluate SWIRAdaIRSR (spec SS11)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default=None, help='default: the config test split')
    ap.add_argument('--sim_b', default='configs/sim_b.yaml',
                    help='held-out simulator; "" to skip the sim_B arm')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--lpips', default='', help='alex|vgg to enable LPIPS')
    ap.add_argument('--output_path', default='')
    ap.add_argument('--patch_eval', action='store_true',
                    help='evaluate on hr_patch crops instead of full frames')
    ap.add_argument('--out', default='')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ck = load_checkpoint(args.ckpt, device)
    cfg = ck['config']
    profile = load_data_profile(cfg['data']['data_profile'])

    split = args.split or cfg['data']['test_split']
    scales = tuple(cfg['model']['scales'])
    subsets = cfg['eval']['fixed_subsets']
    metrics = tuple(cfg['eval']['metrics'])

    sims = [('sim_A', cfg['degradation'])]
    if args.sim_b:
        sims.append(('sim_B', yaml.safe_load(open(args.sim_b))['degradation']))

    resolution = 'hr_patch_{}'.format(cfg['data']['hr_patch']) if args.patch_eval else 'full_frame'
    print('checkpoint : {} (epoch {}, commit {})'.format(
        args.ckpt, ck.get('epoch'), ck.get('commit')))
    print('split      : {}  scheme={}'.format(split, cfg['data'].get('splits_json') or 'shipped'))
    print('resolution : {}   (INV-7: stated with every table)'.format(resolution))

    report = {
        'checkpoint': args.ckpt, 'epoch': ck.get('epoch'), 'commit': ck.get('commit'),
        'split': split, 'resolution': resolution, 'seed': cfg['experiment']['seed'],
        'splits_json': cfg['data'].get('splits_json'),
        'metric_conventions': metric_conventions(args.lpips or None),
        'results': {},
    }

    for sim_name, sim_cfg in sims:
        print('\n=== {} ==={}'.format(
            sim_name, '  (simulator-shift robustness, NOT external validity)'
            if sim_name == 'sim_B' else ''))
        header = '{:<16} {:>5} {:>8} {:>8} {:>8} {:>7}'.format(
            'degradation', 'scale', 'PSNR', 'SSIM', 'edge', 'n')
        print(header)
        print('-' * len(header))

        for subset in subsets:
            for scale in scales:
                res, n = evaluate(model, cfg, profile, sim_cfg, split, subset, scale,
                                  device, args.limit, metrics, args.lpips or None,
                                  args.output_path or None, full_frame=not args.patch_eval)
                report['results']['{}/{}/s{}'.format(sim_name, subset, scale)] = dict(res, n=n)
                print('{:<16} {:>5} {:>8.2f} {:>8.4f} {:>8.4f} {:>7d}'.format(
                    subset, scale, res['psnr'], res['ssim'], res.get('edge', float('nan')), n))

        for scale in scales:
            vals = [v['psnr'] for k, v in report['results'].items()
                    if k.startswith(sim_name) and k.endswith('s{}'.format(scale))]
            print('{:<16} {:>5} {:>8.2f}   <- mean over subsets (never reported alone)'.format(
                'MEAN', scale, float(np.mean(vals))))

    out = args.out or os.path.join(os.path.dirname(args.ckpt) or '.', 'test_results.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
