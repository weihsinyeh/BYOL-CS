import torch, os, argparse
from torchvision import datasets
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from dataprocess.transforms import data_argument, MultiViewDataInjector
from dataprocess.normalize import data_normalize

# hw1_1
from hw1_1.trainer import BYOLTrainer
from hw1_1.warmup_scheduler import GradualWarmupScheduler
from hw1_1.utils.dataloader import pretrain_dataloader

print(torch.__version__)
torch.manual_seed(0)
def parse():
    parser = argparse.ArgumentParser()
    # data transform
    parser.add_argument('--input_shape',        type=tuple, default=(128,128,3))
    # trainer
    parser.add_argument('--batch_size',         type=int,   default=64)
    parser.add_argument('--max_epochs',         type=int,   default=2000)
    # optimizer
    parser.add_argument('--lr',                 type=float, default=0.0005)
    # pretrained model path
    parser.add_argument('--data_dir',           type=str,   default='/project/g/r13922043/hw1_data/p1_data/mini/train')
    parser.add_argument('--checkpointdir',      type=str,   default='/project/g/r13922043/hw1_1/pretrain_checkpoints924')
    parser.add_argument('--logdir',             type=str,   default='/project/g/r13922043/hw1_1/logdir_pretrain924')
    # finetune path
    parser.add_argument('--train_csv_file',     type=str,   default='/project/g/r13922043/hw1_data/p1_data/office/train.csv')
    parser.add_argument('--test_csv_file',      type=str,   default='/project/g/r13922043/hw1_data/p1_data/office/val.csv')
    parser.add_argument('--finetune_train_dir', type=str,   default='/project/g/r13922043/hw1_data/p1_data/office/train')
    parser.add_argument('--finetune_test_dir',  type=str,   default='/project/g/r13922043/hw1_data/p1_data/office/val')
    parser.add_argument('--finetune_checkpoint',type=str,   default='/project/g/r13922043/hw1_1/finetune_checkpoints')

    config = parser.parse_args()
    return config

def main():
    config = parse()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enable_amp = True if config.device == 'cuda' else False
    config.data_transform = data_argument()
    config.data_normalize = data_normalize()
    print(f"Training with: { config.device}")

    # Load Dataset
    train_loader, valid_loader = pretrain_dataloader(config)

    # Load Backbone - Using no weights:
    backbone        = models.resnet50(weights=None)
    input = {   'backbone' : backbone,
                'input_shape' : config.input_shape,
                'device' : config.device,
                'hidden_layer' : -2,
                'output_dim' : 256,
                'hidden_dim' : 4096,
                'moving_average_decay' : 0.99,
                'use_momentum' : True }

    '''
        Bootstrap your own latent: A new approach to self-supervised Learning
        Link : https://arxiv.org/pdf/2006.07733
    '''
    learner         = BYOLTrainer(**input).to(config.device)

    # Optimizer
    optim   = torch.optim.Adam(learner.parameters(), lr=config.lr)
    scaler  = torch.cuda.amp.GradScaler(enable_amp)

    total_epoch = 1000
    scheduler = GradualWarmupScheduler(
        optim,
        multiplier=1,
        total_epoch=total_epoch,
        after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=config.max_epochs * len(train_loader) - total_epoch)
    )

    # Path to save model
    if not os.path.exists(config.logdir):
        os.makedirs(config.logdir)
    if not os.path.exists(config.checkpointdir):
        os.makedirs(config.checkpointdir)
    writer = SummaryWriter(config.logdir)

    optim.step()
    log_global_step = 0
    saved_files = []

    for epoch in range(config.max_epochs):
        writer.add_scalar('training/epoch', epoch, global_step=log_global_step)
        # Training
        loss_list = []
        learner.train()
        for data in tqdm(train_loader):
            optim.zero_grad()
            # Forward pass
            img = data['img'].to(config.device)
            img2 = data['img2'].to(config.device)

            with torch.cuda.amp.autocast(enable_amp):
                loss = learner(img,img2)

            # Back propagation
            scheduler.step()
            # scale loss
            scaler.scale(loss).backward()
            # scale step
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            # Log
            loss_list.append(loss.item())
            writer.add_scalar( "training/lr", optim.param_groups[0]['lr'], global_step=log_global_step)
            writer.add_scalar( "training/loss", loss.item(), global_step=log_global_step)
            log_global_step += 1

        # Record Training Loss
        print(f"Epoch {epoch} : Training Loss {np.mean(loss_list)}")
        print(f"Epoch {epoch} : Learning Rate {optim.param_groups[0]['lr']}")
        if epoch % 5 == 0:
            writer.add_scalar('train/loss', np.mean(loss_list), epoch)
            print(f"Epoch {epoch} : Training Loss {np.mean(loss_list)}")
            save_path = os.path.join(config.checkpointdir,f"backbone_{epoch}.pth")
            torch.save(backbone.state_dict(), save_path)

        # Validation
        validation_loss = []
        learner.eval()
        for data in tqdm(valid_loader):
            optim.zero_grad()
            # Forward pass
            img = data['img'].to(config.device)
            img2 = data['img2'].to(config.device)

            with torch.cuda.amp.autocast(enable_amp):
                loss = learner(img,img2)

            loss = loss.item()
            validation_loss.append(loss)
        
        # Record Validation Loss
        writer.add_scalar("validation/loss", np.mean(validation_loss), epoch)
        print(f"Epoch {epoch} : validation_loss Loss {np.mean(validation_loss)}")

        learner.train()

    save_path = os.path.join(config.checkpointdir,f"last_backbone.pth")
    torch.save(backbone.state_dict(), save_path)
    print(f'Best validation loss: {min(saved_files, key=lambda x: -x[0])}')

if __name__ == '__main__':
    main()