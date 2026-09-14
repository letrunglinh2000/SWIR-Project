"""Build split manifests + MANIFEST.sha256 (spec SS9.1, SS9.3).

Why this exists
---------------
The shipped NSLSR folders are NOT usable as an independent train/val/test split
for restoration:

  * `NSLSR_test` is a strict SUBSET of `NSLSR_val` (all 118 ids), so selecting a
    model on val is selecting it on test (violates spec SS17.12);
  * the folders stride one contiguous capture sequence (test = every 10th frame,
    val = every ~4th, train = the rest), so 116/118 test frames have an
    immediately adjacent train frame and `tools/leakage_audit.py` flags 34 of
    them as near-duplicates.

`--scheme segment` therefore partitions the frame-id axis into contiguous
segments and assigns whole segments to train/val/test, so no split boundary ever
falls between adjacent frames. `--scheme original` reproduces the shipped folders
for comparison with published detection numbers; it is expected to fail the
leakage audit and is labelled as such in the manifest.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.swir_io import list_images  # noqa: E402


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def collect_frames(root, modality, splits):
    """Map frame id -> relative path, taking each id once (test duplicates val)."""
    frames = {}
    for split in splits:
        d = os.path.join(root, split, modality)
        if not os.path.isdir(d):
            continue
        for p in list_images(d):
            stem = os.path.basename(p).rsplit('.', 1)[0]
            try:
                fid = int(stem)
            except ValueError:
                continue
            frames.setdefault(fid, os.path.relpath(p, root))
    return frames


def segment_split(ids, segment_len, ratios, seed, guard=0):
    """Assign contiguous runs of frame ids to whole splits.

    `guard` drops that many frames from each end of every segment, so even the
    boundary frames of a held-out segment are not temporally adjacent to a
    training frame.
    """
    ids = sorted(ids)
    segments = [ids[i:i + segment_len] for i in range(0, len(ids), segment_len)]
    if guard > 0:
        segments = [seg[guard:len(seg) - guard] for seg in segments]
        segments = [seg for seg in segments if seg]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(segments))
    n_val = max(1, int(round(ratios[1] * len(segments))))
    n_test = max(1, int(round(ratios[2] * len(segments))))

    assign = {'val': order[:n_val], 'test': order[n_val:n_val + n_test],
              'train': order[n_val + n_test:]}
    out = {}
    for name, idxs in assign.items():
        out[name] = sorted(i for k in idxs for i in segments[k])
    return out, len(segments), segment_len


def main():
    ap = argparse.ArgumentParser(description='Build leakage-aware split manifests')
    ap.add_argument('--root', default='datasets/NSLSR')
    ap.add_argument('--modality', default='SWIR')
    ap.add_argument('--source_splits', nargs='+',
                    default=['NSLSR_train', 'NSLSR_val', 'NSLSR_test'])
    ap.add_argument('--scheme', choices=['segment', 'original'], default='segment')
    ap.add_argument('--segment_len', type=int, default=40,
                    help='frames per contiguous segment (segment scheme)')
    ap.add_argument('--ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    metavar=('TRAIN', 'VAL', 'TEST'))
    ap.add_argument('--guard', type=int, default=2,
                    help='frames dropped at each segment edge (segment scheme)')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default='configs/splits.json')
    ap.add_argument('--hash_manifest', default='datasets/MANIFEST.sha256')
    ap.add_argument('--skip_hashes', action='store_true')
    args = ap.parse_args()

    frames = collect_frames(args.root, args.modality, args.source_splits)
    ids = sorted(frames)
    print('distinct frame ids: {} (range {}..{})'.format(len(ids), ids[0], ids[-1]))

    if args.scheme == 'segment':
        assign, n_seg, seg_len = segment_split(ids, args.segment_len, args.ratios,
                                               args.seed, guard=args.guard)
        note = ('Contiguous segments of {} frames assigned whole to a split, with a {}-frame '
                'guard band dropped at each segment edge, so no held-out frame is temporally '
                'adjacent to a training frame.'.format(seg_len, args.guard))
        leak = 'expected_to_pass'
    else:
        assign = {}
        for split, name in zip(args.source_splits, ('train', 'val', 'test')):
            d = os.path.join(args.root, split, args.modality)
            assign[name] = sorted(int(os.path.basename(p).rsplit('.', 1)[0])
                                  for p in list_images(d))
        n_seg, seg_len = None, None
        note = ('Shipped folder split reproduced verbatim. test is a SUBSET of val and the '
                'folders stride one capture sequence -- expected to FAIL the leakage audit. '
                'Use only for comparison with published detection numbers.')
        leak = 'expected_to_fail'

    overlaps = {
        'train_val': len(set(assign['train']) & set(assign['val'])),
        'train_test': len(set(assign['train']) & set(assign['test'])),
        'val_test': len(set(assign['val']) & set(assign['test'])),
    }
    adjacency = {}
    tr = set(assign['train'])
    for name in ('val', 'test'):
        adjacency[name] = sum(1 for i in assign[name] if (i - 1) in tr or (i + 1) in tr)

    manifest = {
        'root': args.root, 'modality': args.modality, 'scheme': args.scheme,
        'seed': args.seed, 'segment_len': seg_len, 'n_segments': n_seg,
        'note': note, 'leakage_audit': leak,
        'counts': {k: len(v) for k, v in assign.items()},
        'id_overlaps': overlaps,
        'frames_adjacent_to_train': adjacency,
        'files': {k: [frames[i] for i in v] for k, v in assign.items()},
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print('counts:', manifest['counts'])
    print('id overlaps:', overlaps)
    print('frames adjacent to a train frame:', adjacency)
    print('wrote', args.out)

    if not args.skip_hashes:
        lines = []
        for split in ('train', 'val', 'test'):
            for rel in manifest['files'][split]:
                lines.append('{}  {}'.format(sha256(os.path.join(args.root, rel)), rel))
        os.makedirs(os.path.dirname(args.hash_manifest) or '.', exist_ok=True)
        with open(args.hash_manifest, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('wrote {} ({} files hashed)'.format(args.hash_manifest, len(lines)))


if __name__ == '__main__':
    main()
