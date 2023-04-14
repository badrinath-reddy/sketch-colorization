from const import *
import cv2
import numpy as np
from tqdm import tqdm

train_file = DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + 'train.txt'
val_file = DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + 'val.txt'
test_file = DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + 'test.txt'

with open(train_file, 'r') as f:
    train_files = eval(f.read())
    
with open(val_file, 'r') as f:
    val_files = eval(f.read())
    
with open(test_file, 'r') as f:
    test_files = eval(f.read())
    
print("Train: ", len(train_files))
print("Val: ", len(val_files))
print("Test: ", len(test_files))

# calculate the mean and std of all training images per channel

sum_r_inp = 0
sum_g_inp = 0
sum_b_inp = 0

sum_r_out = 0
sum_g_out = 0
sum_b_out = 0

for i in tqdm(range(len(train_files))):
    train_files[i] = DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + train_files[i]
    img = cv2.imread(train_files[i])
    img = img.astype(np.float32) / 255
    inp = img[:, :IMG_SIZE]
    out = img[:, IMG_SIZE:]
    sum_r_inp += np.sum(inp[:, :, 0])
    sum_g_inp += np.sum(inp[:, :, 1])
    sum_b_inp += np.sum(inp[:, :, 2])
    
    sum_r_out += np.sum(out[:, :, 0])
    sum_g_out += np.sum(out[:, :, 1])
    sum_b_out += np.sum(out[:, :, 2])
    
mean_r_inp = sum_r_inp / (len(train_files) * IMG_SIZE * IMG_SIZE)
mean_g_inp = sum_g_inp / (len(train_files) * IMG_SIZE * IMG_SIZE)
mean_b_inp = sum_b_inp / (len(train_files) * IMG_SIZE * IMG_SIZE)


mean_r_out = sum_r_out / (len(train_files) * IMG_SIZE * IMG_SIZE)
mean_g_out = sum_g_out / (len(train_files) * IMG_SIZE * IMG_SIZE)
mean_b_out = sum_b_out / (len(train_files) * IMG_SIZE * IMG_SIZE)

print("Mean of input images(rgb): ", mean_r_inp, mean_g_inp, mean_b_inp)
print("Mean of output images(rgb): ", mean_r_out, mean_g_out, mean_b_out)

sd_r_inp = 0
sd_g_inp = 0
sd_b_inp = 0

sd_r_out = 0
sd_g_out = 0
sd_b_out = 0


for i in tqdm(range(len(train_files))):
    img = cv2.imread(train_files[i])
    img = img.astype(np.float32) / 255
    inp = img[:, :IMG_SIZE]
    out = img[:, IMG_SIZE:]
    sd_r_inp += np.sum((inp[:, :, 0] - mean_r_inp) ** 2)
    sd_g_inp += np.sum((inp[:, :, 1] - mean_g_inp) ** 2)
    sd_b_inp += np.sum((inp[:, :, 2] - mean_b_inp) ** 2)
    
    sd_r_out = np.sum((out[:, :, 0] - mean_r_out) ** 2)
    sd_g_out = np.sum((out[:, :, 1] - mean_g_out) ** 2)
    sd_b_out = np.sum((out[:, :, 2] - mean_b_out) ** 2)
    
sd_r_inp = np.sqrt(sd_r_inp / (len(train_files) * IMG_SIZE * IMG_SIZE))
sd_g_inp = np.sqrt(sd_g_inp / (len(train_files) * IMG_SIZE * IMG_SIZE))
sd_b_inp = np.sqrt(sd_b_inp / (len(train_files) * IMG_SIZE * IMG_SIZE))

sd_r_out = np.sqrt(sd_r_out / (len(train_files) * IMG_SIZE * IMG_SIZE))
sd_g_out = np.sqrt(sd_g_out / (len(train_files) * IMG_SIZE * IMG_SIZE))
sd_b_out = np.sqrt(sd_b_out / (len(train_files) * IMG_SIZE * IMG_SIZE))

print("Standard deviation of input images(rgb): ", sd_r_inp, sd_g_inp, sd_b_inp)
print("Standard deviation of output images(rgb): ", sd_r_out, sd_g_out, sd_b_out)