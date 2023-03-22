from os import listdir
from const import DATA_FOLDER, PROCESSED_FOLDER, IMG_SIZE
from edge_detection import edge_detection
import cv2

folders = listdir(DATA_FOLDER)

i = 0
for folder in folders:
    if folder == '.DS_Store':
        continue
    files = listdir(DATA_FOLDER + '/' + folder)
    for file in files:
        i += 1
        if file == '.DS_Store':
            continue
        img = cv2.imread(DATA_FOLDER + '/' + folder + '/' + file)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        original_img, img = edge_detection(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        final_img = cv2.hconcat([original_img, img])

        cv2.imwrite(DATA_FOLDER + '/' + PROCESSED_FOLDER +
                    '/' + str(i) + '.jpg', final_img)
