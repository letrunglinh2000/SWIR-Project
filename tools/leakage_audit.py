"""Spec SS9.3 -- blocking cross-split near-duplicate audit.

NSLSR was captured by a synchronous binocular rig across time periods, so
adjacent frames are plausibly near-duplicates. Any reported number is invalid if
test frames also appear in train. Reports the full distribution either way.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.fft import dctn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.swir_io import list_images, load_data_profile, read_image  # noqa: E402

import cv2  # noqa: E402


def phash(img, hash_size=8, highfreq_factor=4):
    """Classic DCT perceptual hash, returned as a boolean vector."""
    size = hash_size * highfreq_factor
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    coeffs = dctn(small, norm='ortho')[:hash_size, :hash_size]
    med = np.median(coeffs[1:, 1:])          # drop DC from the median
    return (coeffs > med).flatten()


def thumbnails(paths, profile, size):
    hashes, thumbs = [], []
    for p in paths:
        img = read_image(p, profile)
        hashes.append(phash(img))
        thumbs.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    return np.array(hashes), np.array(thumbs, dtype=np.float32).reshape(len(paths), -1)


def main():
    ap = argparse.ArgumentParser(description='Cross-split leakage audit (spec SS9.3)')
    ap.add_argument('--root', default='datasets/NSLSR')
    ap.add_argument('--modality', default='SWIR')
    ap.add_argument('--query_split', default='NSLSR_test')
    ap.add_argument('--ref_splits', nargs='+', default=['NSLSR_train'])
    ap.add_argument('--thumb', type=int, default=64, help='thumbnail side for the MAD metric')
    ap.add_argument('--hamming_threshold', type=int, default=5,
                    help='pHash Hamming distance at or below which a pair is a near-duplicate')
    ap.add_argument('--mad_threshold', type=float, default=0.02,
                    help='mean-absolute-difference at or below which a pair is a near-duplicate')
    ap.add_argument('--out', default='runs/leakage_audit.json')
    ap.add_argument('--profile', default='configs/data_profile.yaml')
    ap.add_argument('--splits_json', default='',
                    help='read file lists from a manifest (configs/splits.json) instead of '
                         'the shipped folders; query/ref names become train|val|test')
    args = ap.parse_args()

    profile = load_data_profile(args.profile)

    if args.splits_json:
        with open(args.splits_json) as fh:
            manifest = json.load(fh)
        q_paths = [os.path.join(args.root, r) for r in manifest['files'][args.query_split]]
        r_paths = [os.path.join(args.root, r)
                   for s in args.ref_splits for r in manifest['files'][s]]
    else:
        q_paths = list_images(os.path.join(args.root, args.query_split, args.modality))
        r_paths = list(itertools.chain.from_iterable(
            list_images(os.path.join(args.root, s, args.modality)) for s in args.ref_splits))
    print('query {}: {} frames | reference {}: {} frames'.format(
        args.query_split, len(q_paths), args.ref_splits, len(r_paths)))

    qh, qt = thumbnails(q_paths, profile, args.thumb)
    rh, rt = thumbnails(r_paths, profile, args.thumb)

    # Hamming distance over pHash bits, and MAD over thumbnails.
    ham = (qh[:, None, :] != rh[None, :, :]).sum(axis=2)
    mad = np.abs(qt[:, None, :] - rt[None, :, :]).mean(axis=2)

    nn_ham, nn_mad = ham.min(axis=1), mad.min(axis=1)
    nn_idx = mad.argmin(axis=1)

    flagged = [
        {'query': os.path.basename(q_paths[i]),
         'nearest_reference': os.path.basename(r_paths[int(nn_idx[i])]),
         'hamming': int(nn_ham[i]), 'mad': float(nn_mad[i])}
        for i in range(len(q_paths))
        if nn_ham[i] <= args.hamming_threshold or nn_mad[i] <= args.mad_threshold
    ]

    report = {
        'query_split': args.query_split,
        'reference_splits': args.ref_splits,
        'n_query': len(q_paths), 'n_reference': len(r_paths),
        'thresholds': {'hamming': args.hamming_threshold, 'mad': args.mad_threshold},
        'nn_hamming': {'min': int(nn_ham.min()), 'p5': float(np.percentile(nn_ham, 5)),
                       'median': float(np.median(nn_ham)), 'max': int(nn_ham.max())},
        'nn_mad': {'min': float(nn_mad.min()), 'p5': float(np.percentile(nn_mad, 5)),
                   'median': float(np.median(nn_mad)), 'max': float(nn_mad.max())},
        'n_flagged': len(flagged),
        'flagged': flagged[:100],
        'passed': len(flagged) == 0,
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2)

    print('nearest-neighbour pHash Hamming: min {} / median {:.1f} / max {}'.format(
        report['nn_hamming']['min'], report['nn_hamming']['median'], report['nn_hamming']['max']))
    print('nearest-neighbour thumbnail MAD: min {:.4f} / median {:.4f} / max {:.4f}'.format(
        report['nn_mad']['min'], report['nn_mad']['median'], report['nn_mad']['max']))
    print('flagged near-duplicate pairs: {}'.format(len(flagged)))
    for f in flagged[:10]:
        print('  {} ~ {}  hamming={} mad={:.4f}'.format(
            f['query'], f['nearest_reference'], f['hamming'], f['mad']))
    print('wrote', args.out)

    if flagged:
        print('\nLEAKAGE AUDIT FAILED (spec SS9.3): resolve before reporting any number.')
        return 1
    print('\nLEAKAGE AUDIT PASSED.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
