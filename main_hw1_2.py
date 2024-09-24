import torch, os, argparse
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from mean_iou_evaluate import mean_iou_score_for_training
from dataprocess.normalize import data_normalize_hw1_2

# hw1_2
from hw1_2.models.modelA import modelA
from hw1_2.models.modelB import modelB
from hw1_1.warmup_scheduler import GradualWarmupScheduler
from hw1_2.utils.dataloader import dataloader

print(torch.__version__)
torch.manual_seed(0)
def parse():
    parser = argparse.ArgumentParser()
    # data transform
    parser.add_argument('--input_shape',        type = tuple, default = (128,128,3))
    # trainning parameter
    parser.add_argument('--batch_size',         type = int,   default = 16)
    parser.add_argument('--max_epochs',         type = int,   default = 2000)
    # optimizer
    parser.add_argument('--lr',                 type = float, default = 0.0005)
    # MODEL
    parser.add_argument('--modelA',             type = bool,  default = False)
    parser.add_argument('--modelB',             type = bool,  default = True)
    parser.add_argument('--modelC',             type = bool,  default = False)
    # path
    parser.add_argument('--data_train_dir',     type = str,   default = '/project/g/r13922043/hw1_data/p2_data/train')
    parser.add_argument('--data_test_dir',      type = str,   default = '/project/g/r13922043/hw1_data/p2_data/val')
    parser.add_argument('--logdir',             type = str,   default = '/project/g/r13922043/hw1_2/logdir')
    parser.add_argument('--checkpointdir',      type = str,   default = '/project/g/r13922043/hw1_2/modelB_checkpoints')

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config

def main():
    config = parse()
    enable_amp = True if config.device == 'cuda' else False

    config.data_normalize = data_normalize_hw1_2()
    print(f"Training with: { config.device}")

    # Load Dataset
    train_loader, val_loader = dataloader(config)

    # Load model
    if config.modelA == True:
        model = modelA(config.device)
    elif config.modelB == True:
        model = modelB(config.device)

    # Loss function
    loss_function = nn.CrossEntropyLoss()

    # Optimizer
    optim   = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler  = torch.cuda.amp.GradScaler(enable_amp)
    total_epoch = 1000
    scheduler = GradualWarmupScheduler(
        optim,
        multiplier=1,
        total_epoch=total_epoch,
        after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=config.max_epochs * len(train_loader) - total_epoch)
    )

    # Path to save model
    if not os.path.exists(config.checkpointdir):
        os.makedirs(config.checkpointdir)

    writer = SummaryWriter(config.logdir)

    optim.step()
    log_global_step = 0

    # Training
    for epoch in range(config.max_epochs):
        writer.add_scalar('training/epoch', epoch, global_step=log_global_step)
        mean_iou_list = []
        loss_list = []
        model.train()
        for data in tqdm(train_loader):
            optim.zero_grad()
            # Forward pass
            img     = data['img'].to(config.device)
            mask    = data['mask'].to(config.device)
            # mask change to floatting 
            mask = mask.long()
            with torch.cuda.amp.autocast(enable_amp):
                if config.modelB == True:
                    img = img.float()
                predict_feature = model(img)['out']
                '''
                # An overview of semantic image segmentation.
                Reference : https://www.jeremyjordan.me/semantic-segmentation/
                Because the cross entropy loss evaluates the class predictions for each
                pixel vector individually and then averages over all pixels,
                we're essentially asserting equal learning to each pixel in the image.
                '''
                loss = loss_function(predict_feature, mask) 

            # Back propagation
            scheduler.step()
            # scale loss
            scaler.scale(loss).backward()
            # loss.backward()
            # scale step
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            
            # Log
            loss_list.append(loss.item())
            writer.add_scalar( "training/lr", optim.param_groups[0]['lr'], global_step=log_global_step)
            writer.add_scalar( "training/loss", loss.item(), global_step=log_global_step)
            log_global_step += 1

        # Record Model
        print(f"Epoch {epoch} : Training Loss {np.mean(loss_list)}")
        if epoch % 5 == 0:
            if config.modelA == True:
                save_path = os.path.join(config.checkpointdir,f"modelA_{epoch}.pth")
            if config.modelB == True:
                save_path = os.path.join(config.checkpointdir,f"modelB_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

        model.eval()
        mean_iou = []
        for data in tqdm(val_loader):
            img     = data['img'].to(config.device)
            mask    = data['mask'].to(config.device)

            with torch.cuda.amp.autocast(enable_amp):
                predict_feature = model(img)['out']

            predict = predict_feature.argmax(dim=1)
            mean_iou.append(mean_iou_score_for_training(predict.detach().cpu().numpy(), mask.cpu().numpy()))

        print(f"Epoch {epoch} : mean_iou {np.mean(mean_iou)}")

        writer.add_scalar('train/loss', np.mean(loss_list), epoch)

    save_path = os.path.join(config.checkpointdir,f"last_model.pth")
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()