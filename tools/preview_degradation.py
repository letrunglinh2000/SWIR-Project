"""Dump degraded/clean pairs as images, to look at what the model is fed.

Goes through NSLSRSWIRDataset itself rather than re-applying the degradations, so
what you see is exactly what training sees -- same frozen seeds, same subsets.
Visualisation only; nothing here feeds back into training.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.swir_degradation import EVAL_SUBSETS  # noqa: E402
from utils.swir_io import load_data_profile  # noqa: E402
from utils.swir_sr_dataset import NSLSRSWIRDataset  # noqa: E402

LABEL_H = 22


def to_u8(t):
    """[1,H,W] float tensor in [0,1] -> HxW uint8."""
    return (np.clip(t.numpy()[0], 0, 1) * 255).round().astype(np.uint8)


def label(img, text):
    """Stack a caption bar above an image."""
    bar = np.full((LABEL_H, img.shape[1]), 30, dtype=np.uint8)
    cv2.putText(bar, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 255, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser(description='Save degraded inputs for inspection')
    ap.add_argument('--config', default='configs/swir_base.yaml')
    ap.add_argument('--sim_b', default='', help='use configs/sim_b.yaml to preview the held-out simulator')
    ap.add_argument('--split', default=None, help='default: the config train split')
    ap.add_argument('--subsets', nargs='+', default=None, help='default: all eval subsets')
    ap.add_argument('--scale', type=int, default=2, choices=[1, 2])
    ap.add_argument('--n', type=int, default=4, help='how many frames')
    ap.add_argument('--index', type=int, nargs='*', default=None, help='specific frame indices')
    ap.add_argument('--patch', action='store_true', help='crop patches instead of full frames')
    ap.add_argument('--separate', action='store_true', help='also write each panel as its own file')
    ap.add_argument('--out', default='runs/preview')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    deg = (yaml.safe_load(open(args.sim_b))['degradation'] if args.sim_b
           else cfg['degradation'])
    profile = load_data_profile(cfg['data']['data_profile'])
    split = args.split or cfg['data']['train_split']
    subsets = args.subsets or list(EVAL_SUBSETS)

    os.makedirs(args.out, exist_ok=True)
    d = cfg['data']

    def make(subset):
        return NSLSRSWIRDataset(
            root=d['root'], split=split, data_profile=profile, degradation_cfg=deg,
            modality=d['modality'], hr_patch=d['hr_patch'], augment=False,
            cache_images=False, seed=cfg['experiment']['seed'],
            full_frame=not args.patch, eval_subset=subset,
            splits_json=d.get('splits_json'))

    sets = {s: make(s) for s in subsets}
    n_total = len(next(iter(sets.values())))
    idxs = args.index if args.index else list(range(min(args.n, n_total)))

    print('split {}  |  simulator {}  |  s={}  |  {} frame(s)'.format(
        split, deg.get('name', '?'), args.scale, len(idxs)))

    for i in idxs:
        cols, name = [], None
        for s in subsets:
            item = sets[s].get_item(i, args.scale)
            name = item['name']
            lr, hr = to_u8(item['lr']), to_u8(item['hr'])

            # s=2 inputs are half-size; upscale (nearest, so the loss of detail is
            # visible rather than smoothed over) to sit beside the target.
            if lr.shape != hr.shape:
                lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Report what was ACTUALLY sampled, not the subset's nominal definition:
            # for `composite` the flags are drawn per example and usually several are on.
            active = [k for k, v in item['meta']['flags'].items() if v]
            tag = ' + '.join(active) if active else 'clean'
            cols.append(label(lr, '{}  [{}]'.format(s, tag)))

            if not cols[:-1]:                      # add the target once, at the left
                cols.insert(0, label(hr, 'HR target {}x{}'.format(*hr.shape)))
            if args.separate:
                cv2.imwrite(os.path.join(args.out, '{}_{}_s{}.png'.format(name, s, args.scale)), lr)

        sep = np.full((cols[0].shape[0], 6), 90, dtype=np.uint8)
        panel = cols[0]
        for c in cols[1:]:
            panel = np.hstack([panel, sep, c])

        path = os.path.join(args.out, '{}_s{}_{}.png'.format(
            name, args.scale, deg.get('name', 'sim')))
        cv2.imwrite(path, panel)
        print('  wrote {}   {}x{}'.format(path, panel.shape[1], panel.shape[0]))

    print('\nleftmost panel is the TARGET; the rest are model INPUTS '
          '(nearest-upscaled to match, so s=2 detail loss stays visible).')


if __name__ == '__main__':
    main()
