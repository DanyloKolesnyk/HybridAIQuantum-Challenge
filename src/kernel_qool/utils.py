import os
import re
import torch
import pandas as pd
from torch.utils.data import Dataset

class MNIST_partial(Dataset):
    def __init__(self, data="./data", transform=None, split="train"):
        """
        A minimal CSV-based MNIST loader.
        CSV must have columns: 'image' (a string "[0.1,0.2,...]") and 'label'.
        """
        self.data_dir = data
        self.transform = transform

        if split == 'train':
            filename = os.path.join(self.data_dir, 'train.csv')
        elif split == 'val':
            filename = os.path.join(self.data_dir, 'val.csv')
        else:
            raise AttributeError("split must be 'train' or 'val'")

        self.df = pd.read_csv(filename)

    def __len__(self):
        return len(self.df['image'])

    def __getitem__(self, idx):
        img_str = self.df['image'].iloc[idx]
        label   = self.df['label'].iloc[idx]
        # Convert from string e.g. "[0.0,0.1,0.2,...]" to list of floats
        img_list = re.split(r',', img_str)
        img_list[0]  = img_list[0].lstrip('[')
        img_list[-1] = img_list[-1].rstrip(']')
        img_float = [float(el) for el in img_list]

        # Convert to 1x28x28
        img_tensor = torch.unflatten(torch.tensor(img_float), 0, (1, 28, 28))

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label
