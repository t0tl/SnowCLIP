from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from PIL import Image
import numpy as np


class GSV10kDataset(Dataset):
    def __init__(self, root, df, transform=None, transform_aug=None):
        self.root = root
        self.df = df
        self.transform = transform
        self.transform_aug = transform_aug
        self.n_aug = 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # gps is of format [Lat, lon]
        gps = self.df.iloc[idx].values
        gps = torch.tensor(gps, dtype=torch.float32)
        img = Image.open(os.path.join(self.root, f'{idx}.png'))
        # if self.transform_aug:
        img_1 = self.transform_aug(img)
        augmented_images = torch.empty((self.n_aug,) + img_1.shape, dtype=torch.float32)
        augmented_images[0] = img_1
        for i in range(1, self.n_aug):
            augmented_images[i] = self.transform_aug(img)
        # print(augmented_images.shape, gps.shape)
        return augmented_images, gps

class AmsterdamData(Dataset):
    def __init__(self, root: str, prefix: str, data_df: pd.DataFrame, transform_aug=None, transform=None):
        self.root = root
        self.transform = transform
        self.transform_aug = transform_aug
        self.data_df = data_df
        self.prefix = prefix
        self.n_aug = 2

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        gps = self.data_df.iloc[idx][["lat", "lon"]].values
        # TODO: check lat lon correct and not lon lat
        gps = gps.astype(np.float64)
        gps = torch.tensor(gps, dtype=torch.float32)
        img_path = self.data_df.iloc[idx]["key"]
        city = self.data_df.iloc[idx]["city"]
        img = Image.open(os.path.join(self.root, city, self.prefix, img_path))
        # if self.transform_aug:
        img_1 = self.transform_aug(img)
        augmented_images = torch.empty((self.n_aug,) + img_1.shape, dtype=torch.float32)
        augmented_images[0] = img_1
        #pdb.set_trace()
        for i in range(1, self.n_aug):
            #print(i)
            augmented_images[i] = self.transform_aug(img)

        return augmented_images, gps
