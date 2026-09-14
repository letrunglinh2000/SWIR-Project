"""Detection evaluation arms A0-A4 (spec SS12). MILESTONE 3.

Spec SS17.11 forbids any detector in the loop before the M2 gate passes, so this
script never trains anything and never lets detector gradients reach the
restoration model. It builds the five required arms with an identical frozen
detector, identical letterbox size and identical confidence/NMS thresholds:

  A0 native reference SWIR      (reference arm, NOT assumed to be an upper bound)
  A1 degraded input, native size                                    (lower bound)
  A2 bicubic xs of the degraded input                       (resolution control)
  A3 restored output, s=2                                              (the claim)
  A4 restored output, s=1              (isolates restoration from upscaling)

Without A2 and A4 an mAP gain in A3 is confounded by pixel count (SS12).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from net.swir_adair_sr import build_model
from utils.swir_degradation import EVAL_SUBSETS, build_pipeline
from utils.swir_io import load_data_profile, read_image

ARMS = ('A0', 'A1', 'A2', 'A3', 'A4')


def read_yolo_labels(path):
    """Normalised YOLO txt: `class cx cy w h`. Verified for NSLSR (5 fields)."""
    if not os.path.isfile(path):
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError('{}: expected 5 fields (class cx cy w h), got {}'.format(
                    path, len(parts)))
            rows.append([float(v) for v in parts])
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


def label_path_for(image_path, root):
    """labels/ sits beside SWIR/ inside each shipped split folder."""
    split_dir = os.path.dirname(os.path.dirname(image_path))
    stem = os.path.basename(image_path).rsplit('.', 1)[0]
    return os.path.join(split_dir, 'labels', stem + '.txt')


def iou_matrix(a, b):
    """a: [N,4], b: [M,4] in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def average_precision(records, n_gt, iou_thr):
    """records: list of (score, is_tp_at_thr). Standard all-point interpolated AP."""
    if n_gt == 0:
        return float('nan')
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records], dtype=np.float64)
    fp = np.cumsum([not r[1] for r in records], dtype=np.float64)
    recall = tp / n_gt
    precision = tp / np.clip(tp + fp, 1e-9, None)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match(preds, gts, iou_thr):
    """Greedy highest-score-first matching; returns (score, is_tp) records."""
    order = np.argsort(-preds[:, 4]) if len(preds) else np.array([], dtype=int)
    used, out = set(), []
    ious = iou_matrix(preds[order, :4], gts) if len(preds) and len(gts) else None
    for rank, pi in enumerate(order):
        best, best_j = 0.0, -1
        if ious is not None:
            for j in range(len(gts)):
                if j in used:
                    continue
                if ious[rank, j] > best:
                    best, best_j = ious[rank, j], j
        if best >= iou_thr and best_j >= 0:
            used.add(best_j)
            out.append((float(preds[pi, 4]), True))
        else:
            out.append((float(preds[pi, 4]), False))
    return out


class FrozenDetector:
    """Thin wrapper around a user-supplied, FROZEN detector."""

    def __init__(self, weights, imgsz, conf, iou, device):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit(
                'detect_eval needs a detector. Install ultralytics and pass --weights, '
                'or adapt FrozenDetector to your detector (spec SS18 Q7: which detector, '
                'trained on what, and is it frozen?).')
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device

    def __call__(self, img):
        """img: float32 [H,W] in [0,1] -> predictions [N,5] xyxy+score, normalised."""
        rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        r = self.model.predict(rgb, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = r.boxes.xyxy.cpu().numpy()
        score = r.boxes.conf.cpu().numpy()[:, None]
        h, w = rgb.shape[:2]
        xyxy = xyxy / np.array([w, h, w, h], dtype=np.float32)
        return np.concatenate([xyxy, score], axis=1).astype(np.float32)


def yolo_to_xyxy(labels):
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def build_arms(img, model, pipeline, out_scale, device, seed):
    """Return {arm: image}. Labels are normalised, so rescaling needs no transform.

    `seed` varies per frame: a constant seed would give every frame in the split
    the same degradation parameters, which is not the distribution the arms are
    supposed to be compared over. The same seed drives A1/A2 and A3/A4 so the
    arms see the same degraded frame.
    """
    lr, _ = pipeline(img, out_scale, seed=seed, force_flags=EVAL_SUBSETS['composite'])
    x = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

    arms = {'A0': img, 'A1': lr}
    arms['A2'] = F.interpolate(x, scale_factor=out_scale, mode='bicubic',
                               align_corners=False).clamp(0, 1)[0, 0].cpu().numpy()
    with torch.no_grad():
        arms['A3'] = model(x, out_scale=2).clamp(0, 1)[0, 0].cpu().numpy()
        x1, _ = pipeline(img, 1, seed=seed, force_flags=EVAL_SUBSETS['composite'])
        arms['A4'] = model(torch.from_numpy(x1).unsqueeze(0).unsqueeze(0).to(device),
                           out_scale=1).clamp(0, 1)[0, 0].cpu().numpy()
    return arms


def main():
    ap = argparse.ArgumentParser(description='Detection arms A0-A4 (spec SS12, M3)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--weights', required=True, help='FROZEN detector weights')
    ap.add_argument('--split', default='NSLSR_val',
                    help='shipped folder name (labels/ live beside SWIR/)')
    ap.add_argument('--root', default='datasets/NSLSR')
    ap.add_argument('--modality', default='SWIR')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.7, help='NMS IoU')
    ap.add_argument('--map_iou', type=float, nargs='+', default=[0.5])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='runs/detect_eval.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = build_model(cfg['model']).to(device).eval()
    model.load_state_dict(ck['model'])

    profile = load_data_profile(cfg['data']['data_profile'])
    pipeline = build_pipeline(cfg['degradation'], profile)
    detector = FrozenDetector(args.weights, args.imgsz, args.conf, args.iou, args.device)

    from utils.swir_io import list_images
    paths = list_images(os.path.join(args.root, args.split, args.modality))
    if args.limit:
        paths = paths[:args.limit]

    print('arms A0-A4 | frozen detector {} | imgsz {} conf {} nms-iou {} | {} frames'.format(
        args.weights, args.imgsz, args.conf, args.iou, len(paths)))

    records = {a: {t: [] for t in args.map_iou} for a in ARMS}
    n_gt = 0
    base_seed = cfg['experiment']['seed']
    for idx, p in enumerate(paths):
        img = read_image(p, profile)
        gts = yolo_to_xyxy(read_yolo_labels(label_path_for(p, args.root)))
        n_gt += len(gts)
        arms = build_arms(img, model, pipeline, 2, device, seed=base_seed + idx)
        for arm, im in arms.items():
            preds = detector(im)
            for t in args.map_iou:
                records[arm][t] += match(preds, gts, t)

    report = {'checkpoint': args.ckpt, 'detector': args.weights, 'frozen': True,
              'split': args.split, 'n_frames': len(paths), 'n_gt_boxes': n_gt,
              'imgsz': args.imgsz, 'conf': args.conf, 'nms_iou': args.iou,
              'arm_definitions': {
                  'A0': 'native reference SWIR (reference arm, not an assumed upper bound)',
                  'A1': 'degraded input, native size (lower bound)',
                  'A2': 'bicubic xs of the degraded input (resolution control)',
                  'A3': 'restored output, s=2 (the claim)',
                  'A4': 'restored output, s=1 (isolates restoration from upscaling)'},
              'ap': {}}

    print('\n{:<4} {}'.format('arm', '  '.join('AP@{:.2f}'.format(t) for t in args.map_iou)))
    print('-' * 30)
    for arm in ARMS:
        aps = [average_precision(records[arm][t], n_gt, t) for t in args.map_iou]
        report['ap'][arm] = {str(t): a for t, a in zip(args.map_iou, aps)}
        print('{:<4} {}'.format(arm, '  '.join('{:.4f}'.format(a) for a in aps)))

    print('\nReport restoration metrics and mAP jointly. A PSNR gain with an mAP loss is '
          'a publishable finding, not something to suppress (spec SS12).')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
