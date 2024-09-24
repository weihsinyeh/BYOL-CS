import random
from torch import nn

class Classifier_old(nn.Module):
    def __init__(   self, 
                    input_features, 
                    num_of_class=65, 
                    num_of_layer=3, 
                    dropout=0.1, 
                    hidden_dim = 512) :
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),  # 假設輸入為單通道圖像
            nn.LeakyReLU(0.2))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),  # 假設輸入為單通道圖像
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=1))


        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 512),  # 根據輸入大小調整
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, num_of_class)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
   
        return x

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
            nn.Linear(1000, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, 65),
            nn.BatchNorm1d(65),
            nn.ReLU(True),
            nn.Dropout(dropout),
        ])
        self.clf = nn.Sequential(*layers)
    def forward(self, x):
        return self.clf(x)
