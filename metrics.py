from utils import get_device, denormalize
import piqa
import torch

device = get_device()

# SSIM by Wang et al. (2004)
# use 1.0 - ssim() to get the loss
def ssim():
    return piqa.SSIM().to(device)

# Multi-scale SSIM by Wang et al. (2018)
# use 1.0 - ms_ssim() to get the loss
def ms_ssim():
    return piqa.MS_SSIM().to(device)

# PSNR by Simoncelli and Bovik (2003)
# should be maximized
def psnr():
    return piqa.PSNR().to(device)

# FID by Heusel et al. (2017)
# should be minimized
def fid():
    return piqa.FID().to(device)

# L1
def l1():
    return torch.nn.L1Loss().to(device)

# L2
def l2():
    return torch.nn.MSELoss().to(device)

# BCE
def bce():
    return torch.nn.BCELoss().to(device)