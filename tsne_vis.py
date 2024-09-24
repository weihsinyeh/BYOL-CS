import torch
import torchvision
from torch import nn
import argparse, os

import numpy as np
import pandas as pd
import sys
# from tsne import bh_sne
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from hw1_1.utils.dataloader import finetune_dataloader
from hw1_1.models.finetune_model import Finetune_Model
from dataprocess.normalize import data_normalize
from dataprocess.transforms import data_argument
# Fixed random seed for reproducibility
torch.manual_seed(0)

def parse():
    parser = argparse.ArgumentParser()
        # finetune path
    parser.add_argument('--train_csv_file',     type = str,     default = './hw1_data/p1_data/office/train.csv')
    parser.add_argument('--test_csv_file',      type = str,     default = './hw1_data/p1_data/office/val.csv')
    parser.add_argument('--finetune_train_dir', type = str,     default = './hw1_data/p1_data/office/train')
    parser.add_argument('--finetune_test_dir',  type = str,     default = './hw1_data/p1_data/office/val')
    parser.add_argument('--finetune_checkpoint',type = str,     default = './hw1_1/finetune_checkpoints_SettingC')

    parser.add_argument('--save_dir',           type = str,       default = './tsne_results')
    parser.add_argument('--batch_size',         type = int,       default=1)

    args = parser.parse_args()
    return args

def gen_features(model, dataloader, device):
    model.eval()
    targets_list = []
    outputs_list = []
    name_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs = data['img'].to(device)
            targets = data['label'].to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = model(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            name_list.append(data['img_name'])
            
    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)


    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.figure(figsize=(10, 10))
    
    scatter = plt.scatter(
        df['x'], df['y'],
        c=df['targets'], 
        cmap='jet', 
        alpha=0.5,
    )
    # from 0 to 64
    plt.colorbar(scatter, ticks=range(65))

    plt.savefig(os.path.join(save_dir,'tsne_epoch15_924.png'), bbox_inches='tight')
    print('done!')

def main(config):

    # Load Dataset
    config.data_transform = data_argument()
    config.data_normalize = data_normalize()
    train_loader, val_loader = finetune_dataloader(config)

    # Load Model
    backbone = torchvision.models.resnet50(weights=None)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    finetune_model = Finetune_Model(backbone        = backbone,
                                    input_features  = 1000,
                                    num_of_class    = 65).to(config.device)
    checkpoint_path = '/home/weihsin/project/dlcv-fall-2024-hw1-weihsinyeh/hw1_1/finetune_checkpoints_SettingC_924/model_epoch15.pth'
    checkpoint = torch.load(checkpoint_path)
    finetune_model.load_state_dict(checkpoint)
    finetune_model = nn.Sequential(*list(finetune_model.children())[:-1])
    targets, outputs = gen_features(finetune_model, train_loader, config.device)
    outputs = outputs.reshape(outputs.shape[0], -1) 
    tsne_plot(config.save_dir, targets, outputs)

if __name__ == '__main__':
    config = parse()
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    config.device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(config)