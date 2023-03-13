# opencv findcountours

import cv2
import numpy as np


def get_contours(img):
    original_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    result_img = np.zeros(img.shape, np.uint8)
    cv2.drawContours(result_img, contours, -1, (255, 255, 255), 3)
    return (original_img, result_img)