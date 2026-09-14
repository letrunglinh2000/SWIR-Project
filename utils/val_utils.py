
import inspect
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# skvideo is only needed for the no-reference NIQE metric and is often absent;
# importing it eagerly used to break every module that touches val_utils.
try:
    from skvideo.measure import niqe
except ImportError:  # pragma: no cover
    niqe = None


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


# skimage renamed `multichannel` to `channel_axis` in 0.19 and dropped the old
# name in 0.23. Older versions absorb unknown kwargs into **kwargs and then fail
# with a confusing "win_size exceeds image extent", so pick the name by
# inspecting the signature rather than by catching an exception.
if 'channel_axis' in inspect.signature(structural_similarity).parameters:
    _SSIM_MULTICHANNEL = {'channel_axis': -1}
else:
    _SSIM_MULTICHANNEL = {'multichannel': True}


def _ssim(clean, recoverd):
    """SSIM for one HxWxC image, across skimage versions and channel counts."""
    if clean.shape[-1] == 1:
        return structural_similarity(clean[..., 0], recoverd[..., 0], data_range=1)
    return structural_similarity(clean, recoverd, data_range=1, **_SSIM_MULTICHANNEL)


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += _ssim(clean[i], recoverd[i])

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


def compute_niqe(image):
    if niqe is None:
        raise ImportError('compute_niqe requires scikit-video (pip install scikit-video)')
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0