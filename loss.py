import torch
from metrics import *
from utils import denormalize
import numpy as np
# pi from torch
from torch import pi
import torch.nn.functional as F
import torch.nn as nn
from vgg_perceptual_loss import VGGPerceptualLoss


# Simple generator loss
class GeneratorLoss(torch.nn.Module):
    def __init__(self, lambda_pixel=100):  # lambda
        super(GeneratorLoss, self).__init__()
        self.lambda_pixel = lambda_pixel
        self.gen_loss = l1()
        self.disc_loss = l2()

    def forward(self, gen_out, gt, disc_out):
        ones = torch.ones_like(disc_out)
        return self.lambda_pixel * self.gen_loss(gen_out, gt) + self.disc_loss(disc_out, ones)

# Simple discriminator loss


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.loss = l2()

    def forward(self, real, fake):
        ones = torch.ones_like(real)
        zeros = torch.zeros_like(real)
        return 0.5 * (self.loss(real, ones) + self.loss(fake, zeros))

# From https://github.com/psyrocloud/MS-SSIM_L1_LOSS/blob/main/MS_SSIM_L1_loss.py
class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        device = get_device()
        self.g_masks = g_masks.to(device)
        self.gen_loss = GeneratorLoss()

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y, disc_out):
        
        x = denormalize(x)
        y = denormalize(y)
        ones = torch.ones_like(disc_out)
                
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return (loss_mix.mean() + self.gen_loss(x, y, disc_out)) / 2


class PerceptualLoss(torch.nn.Module):
    def __init__(self, lambda_pixel=100):
        super(PerceptualLoss, self).__init__()
        self.loss = l2()
        self.vgg_loss = VGGPerceptualLoss()
        self.lambda_pixel = lambda_pixel
        
    def forward(self, gen_out, gt, disc_out):
        gen_out = denormalize(gen_out)
        gt = denormalize(gt)
        ones = torch.ones_like(disc_out)
        
        return self.l2(ones, disc_out) + self.lambda_pixel * self.vgg_loss(gen_out, gt)