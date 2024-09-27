import torch, os, argparse
import torchvision
from torchvision import datasets
import torchvision.models as models
from tqdm import tqdm
import numpy as np
from torch import nn
from dataprocess.transforms import data_argument, MultiViewDataInjector
from dataprocess.normalize import data_normalize
import pandas as pd
# HW 1_1
from hw1_1.models.finetune_model import Finetune_Model
from hw1_1.warmup_scheduler import GradualWarmupScheduler
from hw1_1.utils.dataloader import evaluation_dataloader

def finetune(config):
    config.device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    enable_amp = True if config.device == 'cuda' else False

    backbone = models.resnet50(weights=None)
    backbone = backbone.to(config.device)

    # Load Dataset
    config.data_transform = data_argument()
    config.data_normalize = data_normalize()
    val_loader = evaluation_dataloader(config)

    # Finetune Model (backbone + classifier)
    finetune_model = Finetune_Model(backbone        = backbone,
                                    input_features  = 1000,
                                    num_of_class    = 65).to(config.device)
    new_state_dict = {}
    for name, param in finetune_model.named_parameters():
        if 'backbone.' in name:
            new_name = name.replace('backbone.', '')
        if 'classifier.' in name:
            new_name = name.replace('classifier.', '')
        new_state_dict[new_name] = param

    finetune_model.load_state_dict(new_state_dict, strict=False)
    epoch_dict = dict()
    ###################################################################
    # 讓 file 依照 epoch 排序
    filelist = os.listdir(config.finetune_checkpoint)
    filelist.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    for file in filelist:
        filenname = 'model_epoch' + str(file) + '.pth'
        checkpoint_path = os.path.join(config.finetune_checkpoint, file)
        checkpoint = torch.load(checkpoint_path)
        finetune_model.load_state_dict(checkpoint, strict=True)

        data_frame = pd.read_csv(config.test_csv_file)
        validation_dict = dict()
        for idx, row in data_frame.iterrows():
            validation_dict[row['filename']] = row['label']

        finetune_model.eval()
        output_dict = dict()
        correct_number = 0

        # evaluate
        for data in tqdm(val_loader):
            img     = data['img'].to(config.device)
            label   = data['label'].to(config.device)
            
            output_features = finetune_model(img)    
            predictions = torch.argmax(output_features, dim=1)
            correct_number += (predictions == label).sum().item()
            output_dict[data['img_name'][0]] = predictions.item()

            output_dict = {k: v for k, v in sorted(output_dict.items(), key=lambda item: item[0])}
        
        print(f'Epoch: {file}')
        print(f'Validation Accuracy: {correct_number / len(val_loader.dataset)}')
        epoch_dict[file] = correct_number / len(val_loader.dataset)
    return
    with open(config.output_csv_file, 'w') as f:
        for key, value in output_dict.items():
            f.write(f'{key},{value}\n')
        print('Output finished and saved to', config.output_csv_file)

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_csv_file',      type = str,     default = './hw1_data/p1_data/office/val.csv')
    parser.add_argument('--finetune_test_dir',  type = str,     default = './hw1_data/p1_data/office/val')
    parser.add_argument('--output_csv_file',    type = str,     default = 'PB1_output.csv')
    parser.add_argument('--finetune_checkpoint',type = str,     default = './hw1_1/finetune_checkpoints_SettingC_927')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.manual_seed(0)
    config = parse()
    finetune(config)