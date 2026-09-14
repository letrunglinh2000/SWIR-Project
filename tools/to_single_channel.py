"""Re-container NSLSR frames as true single-band images (spec SS9.2).

NSLSR ships its SWIR frames as 4:2:0 subsampled 3-component JPEGs: the SOF header
declares Y/Cb/Cr, so PIL reports mode 'RGB'. The content is luma only -- Cb and Cr
sit at 128 +/- 0.3 -- but the container forces every reader to decide how to
collapse three channels, and the chroma quantisation leaks a few levels of false
colour near edges.

This writes a parallel tree of genuine 1-channel PNGs. PNG because it is lossless:
every output decodes to EXACTLY the channel-0 values the pipeline already reads,
so no previously measured number changes. Re-encoding to JPEG instead would stack
a second lossy generation onto an already-lossy image, and the model would learn
to undo those artefacts as if they were part of the degradation.

The source tree is never modified -- `tools/make_splits.py --scheme original` and
the detection reproduction still need it.
"""

import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.swir_io import list_images, read_raw, reduce_channels  # noqa: E402


def convert_file(src, dst, rule):
    # Read and collapse through exactly the same code path the loader uses, so the
    # converted file cannot drift from what training would have seen. Reading with
    # PIL here instead would silently take RED (PIL is RGB) where the loader takes
    # BLUE (cv2 is BGR) -- a 7-level difference on 1 frame in 3.
    arr = read_raw(src)
    band = reduce_channels(arr, rule)

    if band.dtype != np.uint8:
        raise ValueError('{}: expected uint8, got {}. A 16-bit source must have its '
                         'encoding verified first (spec SS9.2).'.format(src, band.dtype))

    Image.fromarray(band, mode='L').save(dst, 'PNG', optimize=True)

    # Lossless is a promise, so check it rather than trust it.
    back = np.array(Image.open(dst))
    if back.ndim != 2 or not np.array_equal(back, band):
        raise AssertionError('{}: round-trip is not identical'.format(dst))
    return band.nbytes, os.path.getsize(src), os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser(description='Convert NSLSR to single-band PNG')
    ap.add_argument('--src', default='datasets/NSLSR')
    ap.add_argument('--dst', default='datasets/NSLSR_gray')
    ap.add_argument('--splits', nargs='+',
                    default=['NSLSR_train', 'NSLSR_val', 'NSLSR_test', 'NSLSR320_val'])
    ap.add_argument('--modalities', nargs='+', default=['SWIR', 'LWIR'])
    ap.add_argument('--rule', default='luma', choices=['luma', 'take_channel_0', 'mean'],
                    help='how to collapse the source channels (see utils.swir_io)')
    ap.add_argument('--copy_labels', action='store_true', default=True)
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit('refusing to convert in place: --dst must differ from --src')

    total = src_bytes = dst_bytes = 0
    for split in args.splits:
        sdir = os.path.join(args.src, split)
        if not os.path.isdir(sdir):
            print('{:<16} ABSENT, skipped'.format(split))
            continue

        for mod in args.modalities:
            smod = os.path.join(sdir, mod)
            if not os.path.isdir(smod):
                continue
            dmod = os.path.join(args.dst, split, mod)
            os.makedirs(dmod, exist_ok=True)

            paths = list_images(smod)
            for p in paths:
                stem = os.path.basename(p).rsplit('.', 1)[0]
                _, sb, db = convert_file(p, os.path.join(dmod, stem + '.png'), args.rule)
                src_bytes += sb
                dst_bytes += db
            total += len(paths)
            print('{:<16} {:<5} {:4d} frames -> {}'.format(split, mod, len(paths), dmod))

        slab = os.path.join(sdir, 'labels')
        if args.copy_labels and os.path.isdir(slab):
            dlab = os.path.join(args.dst, split, 'labels')
            if os.path.isdir(dlab):
                shutil.rmtree(dlab)
            shutil.copytree(slab, dlab)
            print('{:<16} labels copied ({} files)'.format('', len(os.listdir(dlab))))

    print('\n{} frames converted, all verified byte-identical to the source band'.format(total))
    print('size: {:.1f} MB -> {:.1f} MB ({:.2f}x)'.format(
        src_bytes / 1e6, dst_bytes / 1e6, dst_bytes / max(src_bytes, 1)))
    print('\nNext: re-run tools/inspect_dataset.py --root {} and tools/make_splits.py '
          '--root {}'.format(args.dst, args.dst))


if __name__ == '__main__':
    main()
