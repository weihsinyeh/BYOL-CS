import random
from torch import nn
from hw1_1.models.classifier import Classifier, Classifier_old
class Finetune_Model(nn.Module):
    def __init__(   self, 
                    backbone, 
                    input_features, 
                    num_of_class, 
                    dropout = 0.1, 
                    hidden_dim = 512):
        super().__init__()
        self.backbone   = backbone
        self.classifer  = Classifier(   input_features  = input_features,
                                        num_of_class    = num_of_class,
                                        dropout         = dropout,
                                        hidden_dim      = hidden_dim )
    def forward(self, x):
        embedding       = self.backbone(x)
        output_features = self.classifer(embedding)
        return output_features
