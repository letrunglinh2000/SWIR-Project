"""The cross-rank scale guard must actually fire (spec SS6.1, INV-3).

Runs real gloo process groups on CPU, so the guard is exercised end to end rather
than trusted. If ranks ever disagree on out_scale the backward shapes diverge and
the gradient all-reduce desynchronises -- which surfaces as a hang or as silently
wrong gradients, so a passing assertion here is the only cheap protection.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _worker(rank, world, port, scales, q):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('gloo', rank=rank, world_size=world)
    try:
        from train_swir import assert_scale_synchronised
        try:
            assert_scale_synchronised(scales[rank], torch.device('cpu'))
            q.put((rank, 'ok'))
        except RuntimeError:
            q.put((rank, 'raised'))
    finally:
        dist.destroy_process_group()


def _run(scales):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, len(scales), port, scales, q))
             for r in range(len(scales))]
    for p in procs:
        p.start()
    results = dict(q.get(timeout=120) for _ in scales)
    for p in procs:
        p.join(timeout=120)
    return results


@pytest.mark.skipif(not dist.is_gloo_available(), reason='needs gloo')
def test_agreeing_ranks_pass():
    assert set(_run([2, 2]).values()) == {'ok'}


@pytest.mark.skipif(not dist.is_gloo_available(), reason='needs gloo')
def test_disagreeing_ranks_are_caught():
    """Rank 0 on s=1 while rank 1 is on s=2 must raise on BOTH ranks."""
    assert set(_run([1, 2]).values()) == {'raised'}


@pytest.mark.skipif(not dist.is_gloo_available(), reason='needs gloo')
def test_one_dissenter_among_four_is_caught():
    assert set(_run([2, 2, 1, 2]).values()) == {'raised'}
