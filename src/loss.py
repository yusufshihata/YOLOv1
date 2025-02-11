import torch
import torch.nn as nn
from utils import IOU

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=10, lambda_noobj=0.25, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Computes the YOLOv1 loss function.

        Args:
            pred (torch.Tensor): Model predictions with shape (batch_size, S, S, B*5 + C).
            gt (torch.Tensor): Ground truth labels with shape (batch_size, S, S, B*5 + C).

        Returns:
            torch.Tensor: Total loss (localization + confidence + classification).
        """
        batch_size = pred.shape[0]
        
        pred = pred.view(batch_size, self.S, self.S, (self.B * 5 + self.C))
        gt = gt.view(batch_size, self.S, self.S, (self.B * 5 + self.C))

        pred_boxes = pred[..., :self.B * 5].view(batch_size, self.S, self.S, self.B, 5)
        pred_class = pred[..., self.B * 5:]

        gt_boxes = gt[..., :self.B * 5].view(batch_size, self.S, self.S, self.B, 5)
        gt_confidence = gt[..., 4].view(batch_size, self.S, self.S, self.B) # Shape (batch_size, S, S, 2)
        gt_class = gt[..., self.B * 5:]  # Shape (batch_size, S, S, C)

        # Compute IoU for all B boxes
        ious = IOU(pred_boxes[..., :4], gt_boxes[..., :4])  # (batch_size, S, S, B)
        best_box_idx = torch.argmax(ious, dim=-1, keepdim=True)  # Shape (batch_size, S, S, 1)

        # Mask for responsible bounding box
        obj_mask = torch.zeros_like(pred_boxes[..., 0], dtype=torch.bool)  # (batch_size, S, S, B)
        obj_mask.scatter_(-1, best_box_idx, (gt_confidence > 0))  # Assign True to the best box

        # Mask for cells with no objects
        noobj_mask = gt_confidence == 0  # Shape (batch_size, S, S, 1)

        # Localization loss (only for responsible boxes)
        xy_loss = self.mse(
            pred_boxes[..., :2][obj_mask], 
            gt_boxes[..., :2][obj_mask]
        )
        wh_loss = self.mse(
            torch.sqrt(torch.clamp(pred_boxes[..., 2:4][obj_mask], min=1e-6)),
            torch.sqrt(torch.clamp(gt_boxes[..., 2:4][obj_mask], min=1e-6))
        )
        loc_loss = self.lambda_coord * (xy_loss + wh_loss)

        # Confidence loss
        # Extract confidence scores properly
        pred_confidence = pred_boxes[..., 4]  # Shape: (batch_size, S, S, B)

        # Fix obj_mask indexing
        obj_conf_loss = self.mse(pred_confidence[obj_mask], gt_confidence[obj_mask])

        # Fix noobj_mask indexing
        noobj_conf_loss = self.mse(
            pred_confidence[noobj_mask], 
            torch.zeros_like(pred_confidence[noobj_mask])
        )

        conf_loss = obj_conf_loss + self.lambda_noobj * noobj_conf_loss

        # Classification loss (only for cells with objects)
        class_loss = self.ce(
            pred_class[gt_confidence.squeeze(-1) > 0], 
            gt_class.argmax(-1)[gt_confidence.squeeze(-1) > 0]
        )

        # Compute total loss
        total_loss = (loc_loss + conf_loss + class_loss) / batch_size
        return total_loss
