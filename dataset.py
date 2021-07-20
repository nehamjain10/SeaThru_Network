from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image,ImageOps
import numpy as np

class UnderWater(Dataset):
    """UnderWater Dataset."""

    def __init__(self, csv_file = "data.csv", transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_list = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        uw_image = self.files_list["underwater"].iloc[idx]
        gt_image = self.files_list["gt"].iloc[idx]
        
        uw_image = Image.open(uw_image)
        gt_image = Image.open(gt_image)
        flip = np.random.randint(0,2)
        mirror = np.random.randint(0,2)

        if flip==True:
            uw_image = ImageOps.flip(uw_image)
            gt_image = ImageOps.flip(gt_image)
        if mirror==True:
            uw_image = ImageOps.mirror(uw_image)
            gt_image = ImageOps.mirror(gt_image)
        
        if self.transforms is not None:
          uw_image = self.transforms(uw_image)
          gt_image = self.transforms(gt_image)

        sample = {'uw_image': uw_image, 'gt_image': gt_image}


        return sample