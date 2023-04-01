from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from const import DATA_FOLDER, PROCESSED_FOLDER, IMG_SIZE, TRAIN_FILE, TEST_FILE, VAL_FILE
import cv2


class Data(Dataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
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

        self.files = [DATA_FOLDER + '/' +
                      PROCESSED_FOLDER + '/' + img for img in imgs]

        print(self.files[0])


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        full_image = cv2.imread(self.files[idx])
        img = full_image[:, IMG_SIZE:]
        label = full_image[:, :IMG_SIZE]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img = transform(img)
        label = transform(label)

        if self.transform and self.split == "train":
            img = self.transform(img)
            label = self.transform(label)

        return img, label


def get_data_loader(batch_size, shuffle=True, split="train"):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((0, 360)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    ])
    data = Data(split, transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader
