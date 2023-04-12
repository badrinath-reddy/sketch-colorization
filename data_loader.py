from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from const import DATA_FOLDER, PROCESSED_FOLDER, IMG_SIZE, TRAIN_FILE, TEST_FILE, VAL_FILE
import cv2
import random
import torchvision.transforms.functional as TF

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

        mandatory_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img = mandatory_transform(img)
        label = mandatory_transform(label)

        if self.transform and self.split == "train":
            img, label = self.transform(img, label)

        return img, label


def get_data_loader(batch_size, shuffle=True, split="train"):
    data = Data(split = split)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader
