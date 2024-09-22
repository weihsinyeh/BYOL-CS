import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
def create_dataframe_from_directory(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):  
                file_path = os.path.join(root, file)  
                image_data.append({'filename': file})

    df = pd.DataFrame(image_data)
    df['id'] = df.index 
    df = df[['id', 'filename']] 
    return df

class ImageFolderDataset(Dataset):
    def __init__(self, config, finetune = False, train = True):
        self.finetune = finetune
        if finetune == True and train == True:
            self.data_frame = pd.read_csv(config.train_csv_file)
            self.root_dir   = config.finetune_train_dir
        elif finetune == True and train == False:
            self.data_frame = pd.read_csv(config.test_csv_file)
            self.root_dir   = config.finetune_test_dir 
        if finetune == False :
            self.data_frame = create_dataframe_from_directory(config.data_dir)
            self.root_dir   = config.data_dir

        self.transform      = config.data_transform
        self.batch_size     = config.batch_size
        self.data_preprocess = config.data_preprocess

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,idx):
        data = dict()
        img_path            = os.path.join(self.root_dir, self.data_frame.iloc[idx,1])
        img                 = Image.open(img_path).convert("RGB")
        if  self.finetune == True:
            data['label']   = self.data_frame.iloc[idx,2]

        data['img_name']    = self.data_frame.iloc[idx,1]
        data['img']         = self.data_preprocess(img)

        if  self.finetune == False:
            data['img2']        = self.transform(img)

        return data
    
def finetune_dataloader(config):
    train_dataset       = ImageFolderDataset(config, finetune = True, train = True)                             
    train_data_loader   = DataLoader(   train_dataset,
                                        batch_size=config.batch_size,
                                        shuffle = True)
    test_dataset        = ImageFolderDataset(config, finetune = True, train = False)
    test_data_loader    = DataLoader(   test_dataset,
                                        batch_size=1,
                                        shuffle = False)
    return train_data_loader, test_data_loader

def pretrain_dataloader(config):
    train_dataset       = ImageFolderDataset(config, finetune = False)                             
    train_data_loader   = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=1, drop_last=False, shuffle=True)
    return train_data_loader