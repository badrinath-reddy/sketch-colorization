from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from const import DATA_FOLDER, PROCESSED_FOLDER, IMG_SIZE
from os import listdir
import torch
import cv2


class Data(Dataset):
    def __init__(self, is_train=True, transform=None):
        self.transform = transform
        self.is_train = is_train
        self.files = listdir(DATA_FOLDER + '/' + PROCESSED_FOLDER)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        full_image = cv2.imread(
            DATA_FOLDER + '/' + PROCESSED_FOLDER + '/' + self.files[idx])
        img = full_image[:, :IMG_SIZE]
        label = full_image[:, IMG_SIZE:]

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label


def get_data_loader(batch_size, shuffle=True, is_train=True):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation((0, 360)),
        # transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data = Data(is_train, transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader
