from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from const import DATA_FOLDER, PROCESSED_FOLDER, IMG_SIZE, TRAIN_FILE, TEST_FILE, VAL_FILE, MEAN_INP, SD_INP, MEAN_OUT, SD_OUT
import cv2
import random
import torchvision.transforms.functional as TF
from utils import get_device

device = get_device()

class Data(Dataset):
    def __init__(self, split="train"):
        self.split = split
        if split == "train":
            with open(DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + TRAIN_FILE, 'r') as f:
                imgs = eval(f.read())
        elif split == "test":
            with open(DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + TEST_FILE, 'r') as f:
                imgs = eval(f.read())
        elif split == "val":
            with open(DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + VAL_FILE, 'r') as f:
                imgs = eval(f.read())
        else:
            raise ("Invalid split")

        self.files = [DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + img for img in imgs]


    def __len__(self):
        return len(self.files)
    
    def transform(self, image, mask):

        # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(IMG_SIZE, IMG_SIZE))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    def __getitem__(self, idx):
        full_image = cv2.imread(self.files[idx])
        img = full_image[:, IMG_SIZE:]
        label = full_image[:, :IMG_SIZE]        

        standard_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # normalize_img = transforms.Compose([
        #     transforms.Normalize(MEAN_INP, SD_INP)
        # ])
        
        # normalize_label = transforms.Compose([
        #     transforms.Normalize(MEAN_OUT, SD_OUT)
        # ])

        img = standard_transform(img)
        label = standard_transform(label)
        
        # img = normalize_img(img)
        # label = normalize_label(label)
        
        if self.transform and self.split == "train":
            img, label = self.transform(img, label)

        return img.to(device), label.to(device)


def get_data_loader(batch_size, shuffle=True, split="train"):
    data = Data(split = split)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader
