import random
from torch import nn
from models.classifier import Classifier
class Finetune_Model(nn.Module):
    def __init__(   self, 
                    backbone, 
                    input_features, 
                    num_of_class, 
                    num_of_layer = 3, 
                    dropout = 0.1, 
                    hidden_dim = 128):
        super().__init__()
        self.backbone   = backbone
        self.classifer  = Classifier(   input_features  = input_features,
                                        num_of_class    = num_of_class,
                                        num_of_layer   = num_of_layer,
                                        dropout         = dropout,
                                        hidden_dim      = hidden_dim )
    def forward(self, x):
        embedding       = self.backbone(x)
        output_features = self.classifer(embedding)
        return output_features
