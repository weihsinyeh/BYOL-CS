import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead

########## HW1_2 A ##########
class modelA(nn.Module):
    def __init__(self, device) :
        super().__init__()
        '''
            Take pretrained VGG's framework
            (224,224,64) channel = 64   (224,224) --+
            Conv block 1                 | divide 2 |
            (112,112,128) channel = 128 (112,112)   |
            Conv block 2                 | divide 2 | 
            (56,56,256) channel = 256   (56,56)     | TOTAL
            Conv block 3                 | divide 2 | DIVIDE 32                                                                                                                          
            (28,28,512) channel = 512   (28,28)     |
            Conv block 4                 | divide 2 |
            (14,14,512) channel = 512   (14,14)     |
            Conv block 5                 | divide 2-+
            (7,7,512) channel = 512     (7,7)
            fully_connected_net 6
            (1,1,4096) channel = 4096
            fully_connected_net 7
            (1,1,4096) channel = 4096
            fully_connected_net 8
            (1,1,1000) channel = 1000
        '''
    
        '''
            Reference : https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py#L43
            self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # classifier[0]
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),        # classifier[3]
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        '''
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.fully_convolutional_net6 = nn.Sequential(  nn.Conv2d(512, 4096, kernel_size=7, padding=3),
                                        nn.ReLU(),
                                        nn.Dropout2d())
        self.fully_convolutional_net7 = nn.Sequential(  nn.Conv2d(4096, 4096, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Dropout2d())
        # classifier on pixel level
        self.fully_convolutional_net8 = nn.Sequential( nn.Conv2d(4096, 7, kernel_size=1), nn.ReLU(), nn.Dropout2d())
        # Deconvolution
        self.deconv = nn.ConvTranspose2d(7, 7, kernel_size=64,stride = 32)
        self.to(device)
        
    def forward(self, x) -> torch.Tensor:
        image_height, image_width = x.shape[2], x.shape[3]
        x = self.vgg16(x)
        x = self.fully_convolutional_net6(x)
        x = self.fully_convolutional_net7(x)
        x = self.fully_convolutional_net8(x)
        x = self.deconv(x)
        x = x[..., x.shape[2] - image_height:, x.shape[3] - image_width:]
        return x

if __name__ == '__main__':
    VGG16 = vgg16(weights=VGG16_Weights.DEFAULT).features
    print(VGG16.modules)