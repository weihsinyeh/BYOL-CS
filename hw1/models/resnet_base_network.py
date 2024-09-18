import torchvision.models as models
import torch
from models.mlp_head import MLPHead


class ResNet50(torch.nn.Module):
    def __init__(self, **args):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=False)
        print(args)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, 
                                 mlp_hidden_size = args['mlp_hidden_size'],
                                 projection_size = args['projection_size'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)