import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from config import S, B, C
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, root: str, year: str = '2007', image_set: str = 'train', download: bool = True, transform: T.Compose = None):
        self.transform = transform
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, annotation = self.dataset[idx]

        if self.transform:
            img = self.transform(img)
        
        width = int(annotation['annotation']['size']['width'])
        height = int(annotation['annotation']['size']['height'])

        label = torch.zeros((S, S, B * 5 + C))

        objects = annotation['annotation']['object']
        
        # Convert single object to list
        if isinstance(objects, dict):
            objects = [objects]
        
        cell_tracker = {}  # To track the number of boxes per grid cell

        for obj in objects:
            class_name = obj['name']
            class_idx = self.classes.index(class_name)

            bbox = obj['bndbox']
            xmin = int(bbox['xmin'])
            xmax = int(bbox['xmax'])
            ymin = int(bbox['ymin'])
            ymax = int(bbox['ymax'])

            # Convert bbox to YOLO format
            x_center = (xmin + xmax) / (2.0 * width)
            y_center = (ymin + ymax) / (2.0 * height)
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Find grid cell (i, j)
            i = int(S * y_center)
            j = int(S * x_center)

            # Track bounding boxes per grid cell
            if (i, j) not in cell_tracker:
                cell_tracker[(i, j)] = 0
            
            box_idx = cell_tracker[(i, j)]

            if box_idx < B:
                start_idx = box_idx * 5
                label[i, j, start_idx:start_idx+5] = torch.tensor([x_center, y_center, bbox_width, bbox_height, 1])
                label[i, j, B * 5 + class_idx] = 1
                
                cell_tracker[(i, j)] += 1

        return img, label