from os import listdir
from const import DATA_FOLDER, PROCESSED_FOLDER, IMG_SIZE
from edge_detection import edge_detection
import cv2
import random

folders = listdir(DATA_FOLDER)

i = 0
for folder in folders:
    if folder == '.DS_Store' or folder == PROCESSED_FOLDER:
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

        cv2.imwrite(DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + str(i) + '.jpg', final_img)

all_idx = [str(i) + ".jpg" for i in range(1, i + 1)]

random.shuffle(all_idx)
train_idx = all_idx[:int(0.8 * i)]
test_idx = all_idx[int(0.8 * i):int(0.9 * i)]
val_idx = all_idx[int(0.9 * i):]

with open(DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + 'train.txt', 'w') as f:
    f.write(str(train_idx))

with open(DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + 'test.txt', 'w') as f:
    f.write(str(test_idx))

with open(DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + 'val.txt', 'w') as f:
    f.write(str(val_idx))


print('Data extraction completed successfully')
print('Total images: ', i)
