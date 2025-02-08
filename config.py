import torch
import torch.optim as optim
from model import Yolov1
from loss import YoloLoss


architecture = [
    ("conv", [64, 7, 2, 3]),  # Conv1: (filters, kernel_size, stride, padding)
    ("maxpool", [2, 2]),      # MaxPool1: (kernel_size, stride)
    ("conv", [192, 3, 1, 1]),
    ("maxpool", [2, 2]),
    ("conv", [128, 1, 1, 0]),
    ("conv", [256, 3, 1, 1]),
    ("conv", [256, 1, 1, 0]),
    ("conv", [512, 3, 1, 1]),
    ("maxpool", [2, 2]),
    
    ("conv", [256, 1, 1, 0]),
    ("conv", [512, 3, 1, 1]),
    ("conv", [256, 1, 1, 0]),
    ("conv", [512, 3, 1, 1]),
    ("conv", [512, 1, 1, 0]),
    ("conv", [1024, 3, 1, 1]),
    ("maxpool", [2, 2]),
    
    ("conv", [1024, 3, 1, 1]),
    ("conv", [1024, 3, 2, 1]),
    ("conv", [1024, 3, 1, 1]),
    ("conv", [1024, 3, 1, 1]),
]

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-5
BETAS = (0.9, 0.99)
EPOCHS = 50
S = 7
B = 2
C = 20
IMAGE_SIZE = 448

model = Yolov1()
criterion = YoloLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
