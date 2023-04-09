import numpy as np
from skimage.metrics import structural_similarity
import cv2
from utils import denormalize


def ssim(img1, img2):
    # denormalize
    img1 = denormalize(img1) * 255
    img2 = denormalize(img2) * 255
    print(img1.min(), img1.max(), img2.min(), img2.max())
    img1 = img1.permute(1, 2, 0).numpy().astype(np.uint8)
    img2 = img2.permute(1, 2, 0).numpy().astype(np.uint8)
    return structural_similarity(img1, img2, multichannel=True, channel_axis=2)


def psnr(img1, img2):
    img1 = denormalize(img1)
    img2 = denormalize(img2)
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    return cv2.PSNR(img1, img2)
