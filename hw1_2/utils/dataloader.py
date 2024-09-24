import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image 
import glob, sys
sys.path.append("../") 
from mean_iou_evaluate import read_masks

class ImageFolderDataset(Dataset):
    def __init__(self, config, train = True):
        self.train = train
        if self.train == True:
            self.root_dir   = config.data_train_dir
        else :
            self.root_dir   = config.data_test_dir
        
        self.batch_size     = config.batch_size
        self.data_normalize = config.data_normalize
        self.sat_image      = sorted(glob.glob(os.path.join(self.root_dir, '*sat*')))
        self.mask_images    = read_masks(self.root_dir)

    def __len__(self):
        return len(self.sat_image)

    def __getitem__(self,idx):
        data = dict()
        img_path            = self.sat_image[idx]
        img                 = Image.open(img_path).convert("RGB")
        mask                = self.mask_images[idx]
        data['img']         = self.data_normalize(img)
        data['mask']        = torch.from_numpy(mask)
        data['name']        = os.path.basename(img_path)
        return data
    
def dataloader(config):
    train_dataset       = ImageFolderDataset(config, train = True)                             
    train_data_loader   = DataLoader(   train_dataset,
                                        batch_size=config.batch_size,
                                        shuffle = True)
    test_dataset        = ImageFolderDataset(config, train = False)
    test_data_loader    = DataLoader(   test_dataset,
                                        batch_size=1,
                                        shuffle = False)
    return train_data_loader, test_data_loader