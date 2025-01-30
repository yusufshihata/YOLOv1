import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
import numpy as np
from PIL import Image
from config import S, B, C, IMAGE_SIZE

class VOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', download=True, transform=None):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.transform = transform
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        pass
