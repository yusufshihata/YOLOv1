import torch

def IOU(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Shape (..., 4) where last dimension represents (x, y, w, h).
        boxes2 (torch.Tensor): Shape (..., 4) where last dimension represents (x, y, w, h).

    Returns:
        torch.Tensor: IoU values with shape (...), same batch shape as input.
    """
    # Convert (x, y, w, h) -> (x1, y1, x2, y2)
    x1_1, y1_1, w1, h1 = boxes1[..., 0], boxes1[..., 1], boxes1[..., 2], boxes1[..., 3]
    x1_2, y1_2, w2, h2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]

    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

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
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    # Compute IoU (avoid division by zero)
    iou = intersection / torch.clamp(union, min=1e-6)

    return iou
