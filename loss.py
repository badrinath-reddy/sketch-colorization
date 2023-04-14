import torch
from metrics import *
from utils import denormalize
import numpy as np


# Simple generator loss
class GeneratorLoss(torch.nn.Module):
    def __init__(self, lambda_pixel=100):  # lambda
        super(GeneratorLoss, self).__init__()
        self.lambda_pixel = lambda_pixel
        self.gen_loss = l1()
        self.disc_loss = l2()

    def forward(self, gen_out, disc_out, gt):
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


class GeneratorMSSSIMLoss(torch.nn.Module):
    def __init__(self, lambda_pixel=10, ssim_weight=0.84):  # lambda
        super(GeneratorMSSSIMLoss, self).__init__()
        self.lambda_pixel = lambda_pixel
        self.ssim_weight = ssim_weight
        # self.gen_loss = l1()
        self.ms_ssim_loss = ms_ssim()
        self.desc_loss = l2()

    def forward(self, gen_out, disc_out, gt):
        g_window = gaussian_window(gen_out[0], gen_out[2], 0.5)
        ones = torch.ones_like(disc_out)
        # gen_loss = self.gen_loss(gen_out, gt)
        gen_loss = (torch.sum(torch.abs(gen_out - gt) * g_window)
                    ) / gen_out.shape[0]*gen_out.shape[2]
        return self.lambda_pixel * ((1 - self.ssim_weight) * gen_loss + self.ssim_weight * self.ms_ssim_loss(denormalize(gen_out), denormalize(gt))) + self.desc_loss(disc_out, ones)


# from https://github.com/Tandon-A/CycleGAN_ssim/blob/master/cycleGAN_loss.py
def gaussian_window(self, size, channels, sigma):
    gaussian = np.arange(-(size/2), size/2)
    gaussian = np.exp(-1.*gaussian**2/(2*sigma**2))
    gaussian = np.outer(gaussian, gaussian.reshape((size, 1)))  # extend to 2D
    gaussian = gaussian/np.sum(gaussian)								# normailization
    gaussian = np.reshape(gaussian, (1, size, size, 1)) 	# reshape to 4D
    gaussian = np.tile(gaussian, (1, 1, 1, channels))
    return gaussian
