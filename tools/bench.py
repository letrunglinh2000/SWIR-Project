"""Params / MACs / latency / peak memory (spec SS11.1).

MACs are reported at BOTH 256x320 and 512x640; never compare MACs across papers
without stating the input size. Latency is the mean over >=100 timed runs after
warm-up. Nothing here licenses the word "lightweight" or "real-time" -- SS17.10
requires a stated budget that the model actually meets.
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.swir_adair_sr import build_model  # noqa: E402

SIZES = [(256, 320), (512, 640)]


def gpu_contention():
    """Record whether other processes were on the GPU. Latency measured on a
    shared GPU is not a valid deployment number (spec SS11.1, SS17.10)."""
    import subprocess
    try:
        util = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader'], stderr=subprocess.DEVNULL).decode().strip().splitlines()
        procs = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        procs = [p for p in procs if p.strip()]
        return {'gpu_status': util, 'n_compute_processes': len(procs),
                'exclusive': len(procs) <= 1,
                'warning': None if len(procs) <= 1 else
                           'GPU SHARED with {} other process(es): ms/frame is NOT a valid '
                           'deployment latency. Re-run on an idle GPU.'.format(len(procs) - 1)}
    except Exception as e:
        return {'gpu_status': 'unavailable ({})'.format(e), 'exclusive': None}


def macs(model, shape, out_scale, device):
    try:
        from thop import profile as thop_profile
    except ImportError:
        return None
    x = torch.randn(1, model.inp_channels, *shape, device=device)
    m, _ = thop_profile(model, inputs=(x, out_scale), verbose=False)
    return float(m)


def latency(model, shape, out_scale, device, runs, warmup):
    x = torch.randn(1, model.inp_channels, *shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x, out_scale=out_scale)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x, out_scale=out_scale)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else None
    return dt * 1000.0, peak


def main():
    ap = argparse.ArgumentParser(description='Benchmark SWIRAdaIRSR (spec SS11.1)')
    ap.add_argument('--config', default='configs/swir_base.yaml')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--sizes', type=int, nargs='*', default=None,
                    metavar='H W', help='override sizes, as flat H W pairs')
    ap.add_argument('--out', default='runs/bench.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        cfg = ck['config']
    else:
        cfg = yaml.safe_load(open(args.config))

    device = torch.device(args.device)
    model = build_model(cfg['model']).to(device).eval()

    sizes = SIZES
    if args.sizes:
        sizes = [tuple(args.sizes[i:i + 2]) for i in range(0, len(args.sizes), 2)]

    report = {
        'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        'torch': torch.__version__,
        'params_M': sum(p.numel() for p in model.parameters()) / 1e6,
        'trainable_M': sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        'runs': args.runs, 'warmup': args.warmup,
        'contention': gpu_contention(),
        'note': 'MACs are meaningless without the input size; always quote both.',
        'measurements': [],
    }
    print('params: {:.2f} M'.format(report['params_M']))
    warn = report['contention'].get('warning')
    if warn:
        print('WARNING: {}'.format(warn))
    print('{:<12} {:>5} {:>12} {:>10} {:>12}'.format('input HxW', 'scale', 'GMACs', 'ms/frame', 'peak MiB'))
    print('-' * 56)

    for (h, w) in sizes:
        for s in cfg['model']['scales']:
            try:
                g = macs(model, (h, w), s, device)
                ms, peak = latency(model, (h, w), s, device, args.runs, args.warmup)
            except torch.cuda.OutOfMemoryError:
                print('{:<12} {:>5} {:>12} {:>10} {:>12}'.format(
                    '{}x{}'.format(h, w), s, 'OOM', 'OOM', 'OOM'))
                report['measurements'].append({'h': h, 'w': w, 'scale': s, 'oom': True})
                torch.cuda.empty_cache()
                continue
            report['measurements'].append({
                'h': h, 'w': w, 'scale': s,
                'gmacs': None if g is None else g / 1e9,
                'ms_per_frame': ms, 'peak_mib': peak})
            print('{:<12} {:>5} {:>12} {:>10.2f} {:>12}'.format(
                '{}x{}'.format(h, w), s,
                '-' if g is None else '{:.1f}'.format(g / 1e9), ms,
                '-' if peak is None else '{:.0f}'.format(peak)))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
