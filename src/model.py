import torch
import torch.nn as nn
from config.config import architecture

class Yolov1(nn.Module):
    """
    YOLOv1 (You Only Look Once) model for object detection.

    Args:
        architecture (list): Defines the architecture of the Darknet feature extractor.
        input_dim (int): Number of input channels (default is 3 for RGB images).
        B (int): Number of bounding boxes per grid cell.
        S (int): Grid size (SxS grid cells in the output feature map).
        C (int): Number of object classes.
    """
    def __init__(self, architecture=architecture, input_dim=3, B=2, S=7, C=20):
        super(Yolov1, self).__init__()
        self.input_dim = input_dim
        self.B = B  # Number of bounding boxes per grid cell
        self.S = S  # Grid size (SxS)
        self.C = C  # Number of classes
        self.architecture = architecture
        
        # Darknet feature extractor
        self.darknet = self._create_dark_net()
        
        # Fully connected layers for detection head
        self.fc1 = nn.Linear(S * S * 1024, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))
        
        # Initialize weights
        self.apply(self._init_weights)

    def _create_dark_net(self):
        """
        Constructs the Darknet feature extractor using a list-based architecture description.
        
        Returns:
            nn.Sequential: Darknet feature extractor composed of convolutional and max-pooling layers.
        """
        layers = []
        in_channels = self.input_dim

        for layer in self.architecture:
            if layer[0] == 'conv':
                out_channels, kernel_size, stride, padding = layer[1]
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                layers.append(nn.BatchNorm2d(out_channels))  # Batch normalization for stability
                layers.append(nn.LeakyReLU(0.1))  # Activation function
                in_channels = out_channels
            
            elif layer[0] == 'maxpool':
                kernel_size, stride = layer[1]
                layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
        
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        """
        Initializes weights of the model using different strategies for Conv2D and Linear layers.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the YOLOv1 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, S * S * (B * 5 + C)),
                          representing the bounding boxes, confidence scores, and class probabilities.
        """
        x = self.darknet(x)  # Feature extraction
        x = torch.flatten(x, start_dim=1)  # Flatten feature map for fully connected layers
        x = self.fc2(self.dropout(self.fc1(x)))  # Detection head
        return x
