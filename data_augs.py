import torch
import torch.nn as nn
import numpy as np
from skimage.util.shape import view_as_windows


# This random shift is the most realistic random crop available for Atari setting. Since the paper insists that their random cropped results are 84x84, and the original input is also 84x84, IT is most likely to just see it as a random shifting setting with duplicate padded edges.
def atari_random_shift(imgs, size=84, pad=4):
    m = torch.nn.ReplicationPad2d(pad)
    padded_imgs = m(imgs)
    n, c, h, w = padded_imgs.shape
    w1 = torch.randint(0, w - size + 1, (n,))
    h1 = torch.randint(0, h - size + 1, (n,))
    cropped_imgs = torch.empty((n, c, size, size), dtype=imgs.dtype, device=imgs.device)
    # Shifting should be applied consistently across stacked frames
    for i, (padded_img, w11, h11) in enumerate(zip(padded_imgs, w1, h1)):
        cropped_imgs[i][:] = padded_img[:, h11:h11 + size, w11:w11 + size]
    return cropped_imgs