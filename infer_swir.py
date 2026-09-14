"""Run a trained SWIRAdaIRSR checkpoint over real images (spec SS11.3).

Intended for the real-degraded arm: naturally dark or hazy NSLSR frames for
which no reference exists. No PSNR/SSIM is printed here because there is nothing
to compare against -- that arm carries its claim through no-reference metrics,
qualitative panels and detection mAP, not fidelity numbers.
"""

import argparse
import os

import torch

from net.swir_adair_sr import build_model
from utils.swir_io import IMG_EXTS, list_images, load_data_profile, read_image, write_image


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = build_model(ck['config']['model']).to(device).eval()
    model.load_state_dict(ck['model'])
    return model, ck


def main():
    ap = argparse.ArgumentParser(description='Restore real SWIR frames')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input', required=True, help='image file or directory')
    ap.add_argument('--output', required=True)
    ap.add_argument('--out_scale', type=int, default=2, choices=[1, 2])
    ap.add_argument('--tile', type=int, default=0, help='tile size; 0 = whole frame')
    ap.add_argument('--tile_overlap', type=int, default=32)
    ap.add_argument('--bits', type=int, default=8, help='output bit depth')
    ap.add_argument('--profile', default='configs/data_profile.yaml')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ck = load_model(args.ckpt, device)
    profile = load_data_profile(args.profile)

    if os.path.isdir(args.input):
        paths = list_images(args.input)
    elif args.input.lower().endswith(IMG_EXTS):
        paths = [args.input]
    else:
        raise SystemExit('{} is not an image or a directory'.format(args.input))

    print('checkpoint {} (epoch {})  scale x{}  {} frame(s)'.format(
        args.ckpt, ck.get('epoch'), args.out_scale, len(paths)))
    if args.tile:
        print('tiled inference: tile={} overlap={} -- NOT numerically equal to '
              'whole-frame (spec SS17.17)'.format(args.tile, args.tile_overlap))

    os.makedirs(args.output, exist_ok=True)
    for p in paths:
        img = read_image(p, profile)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            if args.tile:
                y = model.forward_tiled(x, out_scale=args.out_scale,
                                        tile=args.tile, overlap=args.tile_overlap)
            else:
                y = model(x, out_scale=args.out_scale)
        y = y.clamp(0, 1)[0].cpu().numpy()

        name = os.path.basename(p).rsplit('.', 1)[0] + '.png'
        write_image(y, os.path.join(args.output, name), bits=args.bits)
        print('  {} {} -> {}'.format(os.path.basename(p), tuple(img.shape), tuple(y.shape[-2:])))

    print('wrote', args.output)


if __name__ == '__main__':
    main()
