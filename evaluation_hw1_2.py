import torch, os, argparse, sys
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from mean_iou_evaluate import mean_iou_score
from dataprocess.normalize import data_normalize_hw1_2

# hw1_2
from hw1_2.models.modelA import modelA
from hw1_1.warmup_scheduler import GradualWarmupScheduler
from hw1_2.utils.dataloader import dataloader 
from PIL import Image
print(torch.__version__)
torch.manual_seed(0)
def parse():
    parser = argparse.ArgumentParser()
    # data transform
    parser.add_argument('--input_shape',        type=tuple, default=(128,128,3))
    # trainning parameter
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--max_epochs',         type=int,   default=2000)
    # optimizer
    parser.add_argument('--lr',                 type=float, default=0.0005)
    # MODEL
    parser.add_argument('--modelA',             type=bool,  default=True)
    parser.add_argument('--modelB',             type=bool,  default=False)
    parser.add_argument('--modelC',             type=bool,  default=False)
    # path
    parser.add_argument('--data_train_dir',     type=str,   default='/project/g/r13922043/hw1_data/p2_data/train')
    parser.add_argument('--data_test_dir',      type=str,   default='/project/g/r13922043/hw1_data/p2_data/val')
    parser.add_argument('--logdir',             type=str,   default='/project/g/r13922043/hw1_2/logdir')
    parser.add_argument('--checkpoint',         type=str,   default='./hw1_2/modelA_checkpoints/modelA_110.pth')
    parser.add_argument('--output_dir',         type=str,   default='/project/g/r13922043/hw1_2/modelA_evaluation')
    config = parser.parse_args()
    return config

def main():
    config = parse()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enable_amp = True if config.device == 'cuda' else False

    config.data_normalize = data_normalize_hw1_2()
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Load Dataset
    train_loader, val_loader = dataloader(config)

    # Load model
    if config.modelA == True:
        model = modelA(config.device)
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint)

    model.eval()
    ACCs = []
    color_map = {   0:  (0,     255,    255),   # Cyan
                    1:  (255,   255,    0),     # Yellow
                    2:  (255,   0,      255),   # Purple
                    3:  (0,     255,    0),     # Green
                    4:  (0,     0,      255),   # Blue
                    5:  (255,   255,    255),   # White
                    6:  (0,     0,      0)      # Black
                }
    for data in tqdm(val_loader):
        img     = data['img'].to(config.device)
        mask    = data['mask'].to(config.device)
        name    = data['name']
        predict_feature = model(img)
        predict_feature = predict_feature.argmax(dim=1)

        mask = predict_feature[0].cpu().numpy()  

        height, width = mask.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_label, color in color_map.items():
            rgb_mask[mask == class_label] = color

        mask_image_pil = Image.fromarray(rgb_mask)

        path_name = os.path.join(config.output_dir, name[0].replace("sat.jpg", "mask.png"))
        mask_image_pil.save(path_name)

if __name__ == '__main__':
    main()