import os

import torch

from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet50
from trainer import BYOLTrainer
from torch.utils.data import DataLoader


print(torch.__version__)
torch.manual_seed(0)

def main():
    config = {
        # projection head
        "mlp_hidden_size": 512,
        "projection_size": 128,
        # data transform
        "data_s": 1,
        "input_shape": (128,128,3),
        # trainer
        "batch_size": 64,
        "m": 0.996, # momentum update
        "checkpoint_interval": 5000,
        "max_epochs": 40,
        "num_workers": 4,
        # optimizer
        "lr": 0.03,
        "momentum": 0.9,
        "weight_decay": 0.0004,
        # path
        "data_dir" : './hw1_data',
        "checkpointdir" : './hw1'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(config['input_shape'],s = config['data_s'])

    dataset = datasets.ImageFolder(root = config['data_dir'], transform=MultiViewDataInjector([data_transform, data_transform]))
    train_dataset = DataLoader(dataset, split = 'train+unlabeled', batch_size=32)

    # online network
    online_network = ResNet50(**config).to(device)


    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features, 
                        mlp_hidden_size = config['mlp_hidden_size'],
                        projection_size = config['projection_size']).to(device)

    # target encoder
    target_network = ResNet50(**config).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                lr = config['lr'], momentum = config['momentum'], weight_decay = config['weight_decay'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config)

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()