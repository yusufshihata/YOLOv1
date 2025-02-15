import torch
import torch.nn as nn
from utils import IOU

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss(reduction="mean")
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
        gt_confidence = gt[..., 4:6].view(batch_size, self.S, self.S, self.B)
        gt_class = gt[..., self.B * 5:]

        # Compute IoU for all B boxes
        ious = IOU(pred_boxes[..., :4], gt_boxes[..., :4])
        best_box_idx = torch.argmax(ious, dim=-1, keepdim=True)

        # Mask for responsible bounding box
        obj_mask = torch.zeros_like(pred_boxes[..., 0], dtype=torch.bool)
        obj_mask.scatter_(-1, best_box_idx, (gt_confidence > 0))

        # Mask for cells with no objects
        noobj_mask = ~obj_mask

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
        pred_confidence = pred_boxes[..., 4]
        obj_conf_loss = self.mse(pred_confidence[obj_mask], gt_confidence[obj_mask])
        noobj_conf_loss = self.mse(
            pred_confidence[noobj_mask],
            torch.zeros_like(pred_confidence[noobj_mask])
        )
        conf_loss = obj_conf_loss + self.lambda_noobj * noobj_conf_loss

        # Classification loss (only for cells with objects)
        obj_cell_mask = gt_confidence.sum(dim=-1) > 0
        class_loss = self.ce(
            pred_class[obj_cell_mask],
            gt_class.argmax(-1)[obj_cell_mask]
        )

        # Compute total loss
        total_loss = (loc_loss + conf_loss + class_loss) / batch_size
        return total_loss
