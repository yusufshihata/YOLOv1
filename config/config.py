import torch
import torch.optim as optim
from torchvision import transforms as T
from src.model import Yolov1
from src.loss import YoloLoss

architecture = [
    ("conv", [32, 3, 1, 1]),    # Initial stem conv
    ("conv", [64, 3, 2, 1]),    # Downsample (Stride=2)
    ("conv", [64, 1, 1, 0]),
    ("conv", [128, 3, 1, 1]),
    ("maxpool", [2, 2]),        # Reduce spatial dims

    # CSP Block (Inspired by CSPDarknet)
    ("conv", [64, 1, 1, 0]),
    ("conv", [128, 3, 1, 1]),
    ("conv", [64, 1, 1, 0]),
    ("conv", [128, 3, 1, 1]),

    ("maxpool", [2, 2]),

    ("conv", [128, 1, 1, 0]),
    ("conv", [256, 3, 1, 1]),
    ("conv", [128, 1, 1, 0]),
    ("conv", [256, 3, 1, 1]),

    ("maxpool", [2, 2]),

    ("conv", [256, 1, 1, 0]),
    ("conv", [512, 3, 1, 1]),
    ("conv", [256, 1, 1, 0]),
    ("conv", [512, 3, 1, 1]),

    ("maxpool", [2, 2]),

    ("conv", [512, 1, 1, 0]),
    ("conv", [1024, 3, 1, 1]),
    ("conv", [512, 1, 1, 0]),
    ("conv", [1024, 3, 1, 1]),

    # Spatial Pyramid Pooling (SPP) to enhance receptive field
    ("conv", [512, 1, 1, 0]),
    ("conv", [1024, 3, 1, 1]),
    ("conv", [512, 1, 1, 0]),
    ("conv", [1024, 3, 1, 1]),

    # Extra Downsample with stride=2 conv
    ("conv", [1024, 3, 2, 1]),

    ("conv", [1024, 3, 1, 1]),
    ("conv", [1024, 3, 1, 1]),
]

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-5
BETAS = (0.9, 0.99)
EPOCHS = 50
NUM_WORKERS = 1
S = 7
B = 2
C = 20
IMAGE_SIZE = 448

aug = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformation = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = Yolov1()
criterion = YoloLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
