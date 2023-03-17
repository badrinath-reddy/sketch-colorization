from main import get_contours
import cv2

read_img = cv2.imread('test.png')

(original_img, result_img) = get_contours(read_img)

# save the result
cv2.imwrite('original_img.png', original_img)
cv2.imwrite('result_img.png', result_img)
