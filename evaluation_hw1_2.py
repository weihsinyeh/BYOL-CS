import torch, os, argparse, sys
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from mean_iou_evaluate import mean_iou_score
from dataprocess.normalize import data_normalize_hw1_2

# hw1_2
from hw1_2.models.modelA import modelA
from hw1_2.models.modelB import modelB

from hw1_2.utils.dataloader import evaluation_dataloader 
from PIL import Image

def parse():
    parser = argparse.ArgumentParser()

    # trainning parameter
    parser.add_argument('--batch_size',         type=int,   default=16)
    # MODEL
    parser.add_argument('--modelA',             type=bool,  default=False)
    parser.add_argument('--modelB',             type=bool,  default=True)
    # path
    parser.add_argument('--data_train_dir',     type=str,   default='./hw1_data/p2_data/train')
    parser.add_argument('--data_test_dir',      type=str,   default='./hw1_data/p2_data/validation')
    parser.add_argument('--checkpoint',         type=str,   default='./bestmodel_PbB.pth')
    parser.add_argument('--output_dir',         type=str,   default='./Pb2_evaluation')
    config = parser.parse_args()
    return config

def main():
    torch.manual_seed(0)
    print(f'random seed set to 0')
    config = parse()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enable_amp = True if config.device == 'cuda' else False

    config.data_normalize = data_normalize_hw1_2()
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Load Dataset
    val_loader = evaluation_dataloader(config)

    # Load model
    if config.modelA == True:
        model = modelA(config.device)
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint)

    if config.modelB == True:
        model = modelB(config.device)
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint)

    model.eval()
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
        # mask    = data['mask'].to(config.device)
        name    = data['name']
        predict_feature = model(img)
        if config.modelA == True:
            predict_feature = predict_feature.argmax(dim=1)
            mask = predict_feature[0].cpu().numpy()  

        elif config.modelB == True:
            predict_feature = model(img)['out']

            predict = predict_feature.argmax(dim=1)
            mask = predict[0].cpu().numpy()  
            

        height, width = mask.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_label, color in color_map.items():
            rgb_mask[mask == class_label] = color

        mask_image_pil = Image.fromarray(rgb_mask)

        path_name = os.path.join(config.output_dir, name[0].replace("sat.jpg", "mask.png"))
        mask_image_pil.save(path_name)

if __name__ == '__main__':
    main()