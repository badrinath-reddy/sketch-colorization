from utils import get_device, denormalize
import piqa
import torch

# SSIM by Wang et al. (2004)
# use 1.0 - ssim() to get the loss
def ssim():
    return piqa.SSIM()

# Multi-scale SSIM by Wang et al. (2018)
# use 1.0 - ms_ssim() to get the loss
def ms_ssim():
    return piqa.MS_SSIM()

# PSNR by Simoncelli and Bovik (2003)
# should be maximized
def psnr():
    return piqa.PSNR()

# FID by Heusel et al. (2017)
# should be minimized
def fid():
    return piqa.FID()

# L1
def l1():
    return torch.nn.L1Loss()

# L2
def l2():
    return torch.nn.MSELoss()

# BCE
def bce():
    return torch.nn.BCELoss()