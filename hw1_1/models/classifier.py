import random
from torch import nn

class Classifier(nn.Module):
    def __init__(   self, 
                    input_features, 
                    num_of_class, 
                    num_of_layer=3, 
                    dropout=0.1, 
                    hidden_dim = 512) :
        super().__init__()
        layers = []
        layers.extend([
            nn.Linear(input_features, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        ])
        layers.append(nn.Linear(hidden_dim, num_of_class))
        self.clf = nn.Sequential(*layers)
    def forward(self, x):
        return self.clf(x)

