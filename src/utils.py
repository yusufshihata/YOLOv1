import torch

def IOU(box1, box2):
    box1_xyxy = torch.cat([
        box1[..., :2] - box1[..., 2:4] / 2,
        box1[..., :2] + box1[..., 2:4] / 2
    ], dim=-1)

    box2_xyxy = torch.cat([
        box2[..., :2] - box2[..., 2:4] / 2,
        box2[..., :2] + box2[..., 2:4] / 2
    ], dim=-1)

    intersection_min = torch.max(box1_xyxy[..., :2], box2_xyxy[..., :2])
    intersection_max = torch.min(box1_xyxy[..., 2:], box2_xyxy[..., 2:])
    intersection_wh = (intersection_max - intersection_min).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]

    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)
