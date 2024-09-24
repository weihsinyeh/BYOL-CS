import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
 
########## HW1_2 B ##########
class modelB(nn.Module):
    def __init__(self, device) :
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # number of class : 7
        self.model.classifier = DeepLabHead(2048, 7)
        self.model.aux_classifier = FCNHead(1024, 7)
        self.to(device)
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
