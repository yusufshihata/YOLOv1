import torch
import torch.optim as optim
from model import Yolov1
from loss import YoloLoss

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
