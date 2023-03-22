import cv2


def edge_detection(img, t=50):
    original_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, t, t*2)
    return original_img, img
