import torch
import torch.nn as nn
from config import architecture

class Yolov1(nn.Module):
    def __init__(self, architecture=architecture, input_dim=3, B=2, S=7, C=20):
        super(Yolov1, self).__init__()
        self.input_dim = input_dim
        self.B = B
        self.S = S
        self.C = C
        self.architecture = architecture
        
        self.darknet = self._create_dark_net()
        
        self.fc1 = nn.Linear(S * S * 1024, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))

        self.apply(self._init_weights)

    def _create_dark_net(self):
        layers = []
        in_channels = self.input_dim

        for layer in self.architecture:
            if layer[0] == 'conv':
                out_channels, kernel_size, stride, padding = layer[1]
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding
                    )
                )
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(0.1))
                in_channels = out_channels

            elif layer[0] == 'maxpool':
                kernel_size, stride = layer[1]
                layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)

        return self.fc2(self.dropout(self.fc1(x)))
