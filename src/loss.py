import torch
import torch.nn as nn
import config
from utils import IOU

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=10, lambda_noobj=0.25, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Computes the YOLOv1 loss function based on localization, confidence, and classification errors.

        Args:
            pred (torch.tensor): The model's predictions with shape (batch_size, S, S, B*5 + C).
                                Contains bounding box coordinates, confidence scores, and class probabilities.
            gt (torch.tensor): The ground truth labels with shape (batch_size, S, S, B*5 + C).
                            Contains true bounding boxes, object presence, and class labels.

        Returns:
            float: The total loss computed as a sum of localization loss, confidence loss, and classification loss.
        """
        pred = pred.view(-1, self.S, self.S, (self.B * 5 + self.C))
        gt = gt.view(-1, self.S, self.S, (self.B * 5 + self.C))

        # Split predictions
        pred_boxes = pred[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        pred_class = pred[..., self.B * 5:]

        # Get ground truth
        gt_boxes = gt[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        gt_confidence = gt_boxes[..., 4]
        gt_class = gt[..., self.B * 5:]

        # Compute IOU
        ious = IOU(pred_boxes[..., :4], gt_boxes[..., :4])  # Compute IOUs for all boxes at once
        _, best_box = torch.max(ious, dim=-1)

        # Masks
        obj_mask = torch.zeros_like(pred_boxes[..., 0], dtype=torch.bool)
        for b in range(self.B):
            obj_mask[..., b] = (best_box == b) & (gt_confidence > 0)
        noobj_mask = gt_confidence == 0

        # Localization loss
        xy_loss = self.mse(pred_boxes[..., :2][obj_mask], gt_boxes[..., :2][obj_mask])
        wh_loss = self.mse(
            torch.sqrt(torch.clamp(pred_boxes[..., 2:4][obj_mask], min=1e-6)),
            torch.sqrt(torch.clamp(gt_boxes[..., 2:4][obj_mask], min=1e-6))
        )
        loc_loss = self.lambda_coord * (xy_loss + wh_loss)

        # Confidence loss
        obj_conf_loss = self.mse(pred_boxes[..., 4][obj_mask], gt_confidence[obj_mask])
        noobj_conf_loss = self.mse(pred_boxes[..., 4][noobj_mask], torch.zeros_like(pred_boxes[..., 4][noobj_mask]))
        conf_loss = obj_conf_loss + self.lambda_noobj * noobj_conf_loss

        # Classification loss
        class_loss = self.ce(pred_class[gt_confidence > 0], gt_class[gt_confidence > 0].argmax(-1))

        # Total loss
        total_loss = (loc_loss + conf_loss + class_loss) / pred.size(0)
        return total_loss
