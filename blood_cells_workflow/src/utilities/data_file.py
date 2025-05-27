from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.classes = sorted(self.dataframe['label'].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]['imgPath']
        label_str = self.dataframe.iloc[idx]['label']
        label = self.class_to_idx[label_str]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((360, 363)),  # Resize all images to 360x363
    transforms.ToTensor()  # Converts to tensor and scales to [0, 1]
])


