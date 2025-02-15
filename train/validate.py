import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import DEVICE
from src.utils import decode_bbox, compute_map

@torch.no_grad()
def validate(model: nn.Module, criterion: nn.Module, validloader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_valid_loss = 0.0
    all_pred_boxes = []
    all_true_boxes = []

    for images, targets in validloader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        pred = model(images)
        loss = criterion(pred, targets)
        total_valid_loss += loss.item()

        # Decode bounding boxes for mAP calculation
        pred_boxes = decode_bbox(pred)
        true_boxes = decode_bbox(targets)

        all_pred_boxes.extend(pred_boxes)
        all_true_boxes.extend(true_boxes)

    avg_valid_loss = total_valid_loss / len(validloader)
    valid_map = compute_map(all_pred_boxes, all_true_boxes)

    print(f"Validation Loss: {avg_valid_loss:.4f} - mAP: {valid_map:.4f}")

    model.train()
    
    return avg_valid_loss, valid_map
