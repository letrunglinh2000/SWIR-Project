"""Spec SS9.2 -- inspect the stored encoding before any loader is written.

Records, for >=N files per split: extension, PIL mode, numpy dtype, shape,
channel count, whether the channels are identical, and intensity percentiles.
Writes configs/data_profile.yaml, which utils/swir_io.py then reads, so the
normalisation assumption is explicit and versioned (spec SS9.2, SS17.5).
"""

import argparse
import collections
import datetime
import os
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.swir_io import list_images, read_raw  # noqa: E402

# Deciding whether a 3-channel file is genuine colour or a grayscale source that
# was JPEG-encoded cannot be done from the MAXIMUM channel deviation: chroma
# subsampling puts a few isolated pixels several levels apart at sharp edges.
# The distribution separates the two cases cleanly -- a real multi-band image
# differs by tens of levels on average, a replicated grayscale one by << 1.
GRAY_MEAN_DEV_MAX = 1.0    # mean |channel - channel0|, in levels
GRAY_P999_DEV_MAX = 8.0    # 99.9th percentile of the same, in levels


def inspect_file(path):
    raw = read_raw(path)
    with Image.open(path) as im:
        pil_mode = im.mode

    channels = 1 if raw.ndim == 2 else raw.shape[-1]
    identical = None
    mean_dev = p999_dev = max_dev = 0.0
    if channels > 1:
        ref = raw[..., :1].astype(np.int32)
        dev = np.abs(raw.astype(np.int32) - ref).max(axis=-1)
        mean_dev = float(dev.mean())
        p999_dev = float(np.percentile(dev, 99.9))
        max_dev = float(dev.max())
        identical = bool(max_dev == 0)

    band = raw if raw.ndim == 2 else raw[..., 0]
    pct = np.percentile(band, [0, 1, 50, 99, 100])
    return {
        'ext': os.path.splitext(path)[1].lower(),
        'pil_mode': pil_mode,
        'dtype': str(raw.dtype),
        'shape': tuple(int(v) for v in raw.shape),
        'channels': int(channels),
        'channels_identical': identical,
        'mean_channel_deviation': mean_dev,
        'p999_channel_deviation': p999_dev,
        'max_channel_deviation': max_dev,
        'min': float(pct[0]), 'p1': float(pct[1]), 'median': float(pct[2]),
        'p99': float(pct[3]), 'max': float(pct[4]),
    }


def summarise(records):
    """Collapse per-file records into the fields the loader needs."""
    def uniq(key):
        return sorted({r[key] for r in records if r[key] is not None})

    dtypes, channels = uniq('dtype'), uniq('channels')
    if len(dtypes) != 1:
        raise SystemExit('Mixed dtypes across files: {} -- resolve before training.'.format(dtypes))
    if len(channels) != 1:
        raise SystemExit('Mixed channel counts: {} -- resolve before training.'.format(channels))

    dtype, nch = dtypes[0], channels[0]
    if dtype == 'uint8':
        bit_depth, max_value = 8, 255
    elif dtype == 'uint16':
        # Spec SS9.2: do NOT assume 65535 for 16-bit data.
        observed = max(r['max'] for r in records)
        raise SystemExit(
            'Files are uint16 (observed max {:.0f}). The encoding must be verified against the '
            'sensor/ADC range before use; edit configs/data_profile.yaml by hand and rerun with '
            '--accept-manual-encoding.'.format(observed))
    else:
        raise SystemExit('Unhandled dtype {}'.format(dtype))

    all_identical = all(r['channels_identical'] for r in records if r['channels_identical'] is not None)
    worst_mean = max((r['mean_channel_deviation'] for r in records), default=0.0)
    worst_p999 = max((r['p999_channel_deviation'] for r in records), default=0.0)
    worst_max = max((r['max_channel_deviation'] for r in records), default=0.0)

    if nch == 1:
        rule, note = 'single', 'Stored as a single band.'
    elif all_identical:
        rule, note = 'luma', ('Channels are bit-identical, so luma recovers the band exactly.')
    elif worst_mean <= GRAY_MEAN_DEV_MAX and worst_p999 <= GRAY_P999_DEV_MAX:
        rule = 'luma'
        note = ('Channels are not bit-identical (worst-file mean deviation {:.3f} levels, '
                'p99.9 {:.1f}, max {:.0f}) but the deviation is far below what genuine '
                'multi-band data shows; this is JPEG chroma subsampling of a grayscale '
                'source. The band is recovered as BT.601 luma rather than by picking one '
                'channel: every RGB channel is luma plus a chroma-quantisation error, and '
                'cv2 decodes BGR so channel 0 is blue, the worst of the three.'.format(
                    worst_mean, worst_p999, worst_max))
    else:
        raise SystemExit(
            'Channel deviation is too large to treat as a replicated grayscale source '
            '(worst-file mean {:.3f}, p99.9 {:.1f}, max {:.0f} levels). This may be genuine '
            'multi-band data -- resolve the channel rule manually before training.'.format(
                worst_mean, worst_p999, worst_max))

    return {
        'dtype': dtype,
        'bit_depth': bit_depth,
        'max_value': max_value,
        'channels': int(nch),
        'channels_identical': bool(all_identical),
        'worst_mean_channel_deviation': worst_mean,
        'worst_p999_channel_deviation': worst_p999,
        'worst_max_channel_deviation': worst_max,
        'channel_rule': rule,
        'normalization': 'divide_by_{}'.format(max_value),
        'note': note,
    }


def main():
    ap = argparse.ArgumentParser(description='Inspect SWIR encoding (spec SS9.2)')
    ap.add_argument('--root', default='datasets/NSLSR')
    ap.add_argument('--splits', nargs='+',
                    default=['NSLSR_train', 'NSLSR_val', 'NSLSR_test', 'NSLSR320_val'])
    ap.add_argument('--modality', default='SWIR')
    ap.add_argument('--per_split', type=int, default=20, help='files to inspect per split (spec: >=20)')
    ap.add_argument('--out', default='configs/data_profile.yaml')
    args = ap.parse_args()

    all_records, per_split, missing = [], {}, []
    for split in args.splits:
        root = os.path.join(args.root, split, args.modality)
        if not os.path.isdir(root):
            # NSLSR320_val is a detection-only subset (spec SS9.1) and may be absent.
            missing.append(split)
            print('{:<16} ABSENT ({})'.format(split, root))
            continue
        paths = list_images(root)
        sample = paths[:args.per_split]
        records = [inspect_file(p) for p in sample]
        all_records += records

        sizes = collections.Counter(r['shape'] for r in records)
        per_split[split] = {
            'files_total': len(paths),
            'files_inspected': len(records),
            'shapes': {str(k): v for k, v in sizes.items()},
            'extensions': sorted({r['ext'] for r in records}),
            'pil_modes': sorted({r['pil_mode'] for r in records}),
            'intensity': {
                'min': min(r['min'] for r in records),
                'median_of_medians': float(np.median([r['median'] for r in records])),
                'max': max(r['max'] for r in records),
            },
        }
        print('{:<16} {:4d} files, inspected {:2d}, shapes {}, modes {}'.format(
            split, len(paths), len(records), dict(sizes), per_split[split]['pil_modes']))

    encoding = summarise(all_records)

    total = sum(v['files_total'] for k, v in per_split.items() if k != 'NSLSR320_val')
    profile = {
        'generated_by': 'tools/inspect_dataset.py (spec SS9.2)',
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'root': args.root,
        'modality': args.modality,
        'encoding': encoding,
        'splits': per_split,
        'splits_absent': missing,
        'frames_total_excluding_320val': total,
        # Spec SS9.1 quotes 1,205 pairs from the LSFDNet README. Record what is
        # actually on disk rather than asserting the quoted number.
        'spec_quoted_total': 1205,
        'total_matches_spec': bool(total == 1205),
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        yaml.safe_dump(profile, fh, sort_keys=False, default_flow_style=False)

    print('\nencoding:', encoding)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
