import torch
import torch.nn as nn
import cv2
import PIL
from config.config import DEVICE, S, B, C, IMAGE_SIZE

def IOU(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Shape (..., 4) where last dimension represents (x_center, y_center, w, h).
        boxes2 (torch.Tensor): Shape (..., 4) where last dimension represents (x_center, y_center, w, h).

    Returns:
        torch.Tensor: IoU values with shape (...), same batch shape as input.
    """
    # Convert (x_center, y_center, w, h) -> (x1, y1, x2, y2)
    x1_1 = boxes1[..., 0] - boxes1[..., 2] / 2  # x1 = x_center - w/2
    y1_1 = boxes1[..., 1] - boxes1[..., 3] / 2  # y1 = y_center - h/2
    x2_1 = boxes1[..., 0] + boxes1[..., 2] / 2  # x2 = x_center + w/2
    y2_1 = boxes1[..., 1] + boxes1[..., 3] / 2  # y2 = y_center + h/2

    x1_2 = boxes2[..., 0] - boxes2[..., 2] / 2
    y1_2 = boxes2[..., 1] - boxes2[..., 3] / 2
    x2_2 = boxes2[..., 0] + boxes2[..., 2] / 2
    y2_2 = boxes2[..., 1] + boxes2[..., 3] / 2

    # Compute intersection coordinates
    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)

    # Compute intersection area
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_w * inter_h

    # Compute union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    # Compute IoU (avoid division by zero)
    iou = intersection / torch.clamp(union, min=1e-6)

    return iou


def non_max_suppression(bboxes, iou_threshold=0.5):
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        bboxes (list): List of boxes [x, y, w, h, class, confidence].
        iou_threshold (float): IoU threshold to remove overlapping boxes.

    Returns:
        List of filtered boxes.
    """
    bboxes = sorted(bboxes, key=lambda x: x[-1], reverse=True)  # Sort by confidence
    filtered_boxes = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        filtered_boxes.append(chosen_box)

        bboxes = [box for box in bboxes if IOU(torch.tensor(chosen_box[:4]), torch.tensor(box[:4])) < iou_threshold] # Remove the duplicates

    return filtered_boxes

def decode_bbox(predictions: torch.Tensor, conf_threshold: float = 0.5):
    """
    Converts YOLO predictions into bounding box coordinates.
    Args:
        predictions: Model output tensor
        conf_threshold: Confidence threshold for filtering boxes

    Returns:
        List of bounding boxes [x, y, w, h, class]
    """

    # Ensure predictions are reshaped correctly
    if predictions.dim() == 2:  # If it's flattened (batch_size, num_features)
        batch_size = predictions.shape[0]
        S = 7  # Assuming YOLOv1 (SxS grid = 7x7)
        B = 2  # Number of bounding boxes per cell
        C = 20  # Number of classes
        predictions = predictions.view(batch_size, S, S, B*5 + C)

    batch_size, S, S, _ = predictions.shape  # Now this should work

    B = 2  # Number of bounding boxes per cell
    C = 20  # Number of classes

    bboxes = predictions[..., :B*5].view(batch_size, S, S, B, 5)
    bbclass = predictions[..., B*5:]

    all_boxes = []
    for i in range(S):
        for j in range(S):
            for b in range(B):
                box = bboxes[0, i, j, b]
                x, y, w, h, conf = box.tolist()
                if conf >= conf_threshold:
                    class_idx = torch.argmax(bbclass[0, i, j]).item()
                    all_boxes.append([x, y, w, h, class_idx])

    return all_boxes


def compute_map(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Computes Mean Average Precision (mAP) at given IoU threshold.

    Args:
        pred_boxes (list): List of predicted boxes [x, y, w, h, class, conf].
        true_boxes (list): List of ground truth boxes.
        iou_threshold (float): IoU threshold for correct detection.

    Returns:
        float: mAP score
    """
    correct_detections = 0
    total_predictions = len(pred_boxes)
    total_ground_truths = len(true_boxes)

    for pred in pred_boxes:
        for gt in true_boxes:
            iou = IOU(torch.tensor(pred[:4]), torch.tensor(gt[:4]))
            if iou >= iou_threshold:
                correct_detections += 1
                break

    precision = correct_detections / (total_predictions + 1e-6)
    recall = correct_detections / (total_ground_truths + 1e-6)

    return (precision * recall) / (precision + recall + 1e-6)


def save_model(model: nn.Module, optimizer: torch.optim, epoch: int, loss: nn.Module, path: str = "best_model.pth") -> None:
    """
    Saves the model state if it's the best so far.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The current epoch.
        loss (float): The best loss so far.
        path (str): Path to save the model file.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'lr': optimizer.param_groups[0]['lr'],
        'betas': optimizer.param_groups[0]['betas'],
    }, path)
    print(f"✅ Model saved at epoch {epoch} with mAP {loss:.4f} -> {path}")


def load_checkpoint(model: nn.Module, path: str = "best_model.pth", optimizer: torch.optim = None) -> nn.Module:
    """
    Loads a model state to do inference on it

    Args:
        model: the model we need to do inference on
        path: the path for the trained model state_dict
        optimizer: the optimizer that the model trained on
    """
    checkpoint = torch.load(path, map_location=DEVICE)

    # Extract model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:  # Restore optimizer state if provided
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.to(DEVICE)
    model.eval()

    print(f"Model restored from {path}, Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    return model
