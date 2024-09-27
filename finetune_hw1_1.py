import torch, os, argparse
import torchvision
from torchvision import datasets
import torchvision.models as models
from tqdm import tqdm
import numpy as np
from torch import nn
from dataprocess.transforms import data_argument, MultiViewDataInjector
from dataprocess.normalize import data_normalize

# HW 1_1
from hw1_1.models.finetune_model import Finetune_Model
from hw1_1.warmup_scheduler import GradualWarmupScheduler
from hw1_1.utils.dataloader import finetune_dataloader

def finetune(config):
    # Fixed random seed for reproducibility
    torch.manual_seed(0)
    print(f'random seed set to 0')
    config.device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    enable_amp = True if config.device == 'cuda' else False
    # Load Pretrained Model - Using no weights:
    # Setting A models load resnet50 directly 
    if config.pretrain == False :
        backbone = models.resnet50(weights=None)
        print('Loaded backbone from resnet50 without weights')
    else :
        # Setting B and D 
        backbone = models.resnet50(weights=None)
        if config.TA_pretrain == True : 
            checkpoint_path = './hw1_data/p1_data/pretrain_model_SL.pt'
        # Setting C and E
        if config.My_pretrain == True : 
            checkpoint_path = './hw1_1/checkpoint_dir/backbone_ckpt/backbone_150.pth'
        checkpoint = torch.load(checkpoint_path)
        backbone.load_state_dict(checkpoint)
        print(f'Loaded backbone from {checkpoint_path}')

    # Remove the last two layers for fully convolutional layer in classifier
    # backbone = nn.Sequential(*list(backbone.children())[:-2])
    backbone = backbone.to(config.device)

    # Setting D and E
    if config.freeze == True:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        print('Freezed backbone')

    # Load Dataset
    config.data_transform = data_argument()
    config.data_normalize = data_normalize()
    train_loader, val_loader = finetune_dataloader(config)

    # Finetune Model (backbone + classifier)
    finetune_model = Finetune_Model(backbone        = backbone,
                                    input_features  = 1000,
                                    num_of_class    = 65).to(config.device)

    optimizer   = torch.optim.Adam(finetune_model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()
    total_epoch = 100
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=total_epoch,
        after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.max_epochs * len(train_loader) - total_epoch
        )
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(config.max_epochs+1):
        loss_list = []
        finetune_model.train()
        for data in tqdm(train_loader):
            img     = data['img'].to(config.device)
            label   = data['label'].to(config.device)
            
            # zero the parameter gradients
            scheduler.step()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enable_amp):
                output_features = finetune_model(img)
                loss = criterion(output_features, label)

            loss_list.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if epoch % 5 == 0:
            checkpoint_name = 'model_epoch' + str(epoch) + '.pth'
            save_path = os.path.join(config.finetune_checkpoint, checkpoint_name)
            torch.save(finetune_model.state_dict(), save_path)  

        print(f'Epoch {epoch} : Training Loss {np.mean(loss_list)}')
        finetune_model.eval()
        validation_loss_list = []
        validation_accuarcy_list = []
        # evaluate
        for data in tqdm(val_loader):
            img     = data['img'].to(config.device)
            label   = data['label'].to(config.device)

            output_features = finetune_model(img)
            loss = torch.nn.functional.cross_entropy( output_features, label)
            validation_loss_list.append(loss.item())

            predictions = torch.argmax(output_features, dim=1)
            validation_accuarcy_list.append(torch.mean((predictions == label).type(torch.float)).item())

        validation_loss = sum(validation_loss_list) / len(validation_loss_list)
        validation_accuarcy = sum(validation_accuarcy_list) / len(validation_accuarcy_list)
        print(f"Validation Loss: {validation_loss}")
        print(f"Validation Accuracy: {validation_accuarcy}")   

        if validation_accuarcy > 0.47 and epoch % 5 != 0:
            checkpoint_name = 'model_epoch' + str(epoch) + '.pth'
            save_path = os.path.join(config.finetune_checkpoint, checkpoint_name)
            torch.save(finetune_model.state_dict(), save_path)

def parse():
    parser = argparse.ArgumentParser()

    # data transform
    parser.add_argument('--input_shape',        type = tuple,   default = (128,128,3))
    # trainer
    parser.add_argument('--batch_size',         type = int,     default = 32)
    parser.add_argument('--max_epochs',         type = int,     default = 400)
    # optimizer
    parser.add_argument('--lr',                 type = float,   default = 0.0005)
    # pretrained model path
    parser.add_argument('--data_dir',           type = str,     default = '/project/g/r13922043/hw1_data/p1_data/mini/train')
    parser.add_argument('--checkpointdir',      type = str,     default = '/project/g/r13922043/hw1_1/pretrain_checkpoints')
    parser.add_argument('--logdir',             type = str,     default = '/project/g/r13922043/hw1_1/logdir')
    # finetune path
    parser.add_argument('--train_csv_file',     type = str,     default = '/project/g/r13922043/hw1_data/p1_data/office/train.csv')
    parser.add_argument('--test_csv_file',      type = str,     default = '/project/g/r13922043/hw1_data/p1_data/office/val.csv')
    parser.add_argument('--finetune_train_dir', type = str,     default = '/project/g/r13922043/hw1_data/p1_data/office/train')
    parser.add_argument('--finetune_test_dir',  type = str,     default = '/project/g/r13922043/hw1_data/p1_data/office/val')
    parser.add_argument('--finetune_checkpoint',type = str,     default = './hw1_1/finetune_checkpoints_SettingC')

    # setting C
    # pretrain model
    parser.add_argument('--pretrain',           type = bool,    default = True)
    parser.add_argument('--TA_pretrain',        type = bool,    default = False)
    parser.add_argument('--My_pretrain',        type = bool,    default = True)
    # finetune model
    parser.add_argument('--freeze',             type = bool,    default = False)
    '''
    Description
    Setting A :
        Pretrain : NULL     
        Finetune : Train full model (backbone + classifier)
    Setting B :
        Pretrain : TAs have provided this backbone
        Finetune : Train full model (backbone + classifier) 
    Setting C :
        Pretrain : Your SSL pre-trained backbone 
        Finetune : Train full model (backbone + classifier)
    Setting D :
        Pretrain : TAs have provided this backbone
        Finetune : Fix the backbone. Train classifier only
    Setting E :
        Pretrain : Your SSL pre-trained backbone 
        Finetune : Fix the backbone. Train classifier only
    '''

    args = parser.parse_args()       
    return args

if __name__ == '__main__':
    config = parse()
    if not os.path.exists(config.finetune_checkpoint):
        os.makedirs(config.finetune_checkpoint)
    finetune(config)
