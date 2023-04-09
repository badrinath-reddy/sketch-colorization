import cv2
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import util
from skimage import img_as_float
import matplotlib.pylab as plt
from medpy.filter.smoothing import anisotropic_diffusion as ad
from skimage.filters import gaussian, threshold_otsu


def normalize(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))


def edge_detection(img, t=50):
    original_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, t, t*2)
    return original_img, img


def dodgeV2(img):
    orginal_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = gaussian(util.invert(img), sigma=5)
    output = np.divide(img, util.invert(img_blurred) + 1)
    output = normalize(output)
    thresh = threshold_otsu(output)
    output = output > thresh
    output = output.astype(np.uint8) * 255
    return orginal_img, output


def anisotropic_diffusion(img):
    original_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = ad(img, niter=5, kappa=50, gamma=0.1,
                                option=1, voxelspacing=None)
    img = normalize(img)
    thresh = threshold_otsu(img)
    img = img > thresh
    img = img.astype(np.uint8) * 255
    return original_img, img
