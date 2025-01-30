import torch
import torch.nn as nn
import config
from utils import IOU

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        batch_size = config.BATCH_SIZE

        gt = gt.view(batch_size, self.S, self.S, 5 + self.C)
        pred = pred.view(batch_size, self.S, self.S, (self.B * 5 + self.C))

        obj_mask = gt[..., 4] == 1
        noobj_mask = gt[..., 4] == 0

        pred_boxes = pred[..., :self.B * 5].view(batch_size, self.S, self.S, self.B, 5)
        pred_classes = pred[..., self.B * 5:]

        gt_box = gt[..., :5]
        gt_class = gt[..., 5:]

        iou_box1 = IOU(pred_boxes[..., 0, :], gt_box)
        iou_box2 = IOU(pred_boxes[..., 1, :], gt_box)

        iou_max, best_box = torch.max(torch.stack([iou_box1, iou_box2], dim=0), dim=0)

        best_pred_box = torch.gather(pred_boxes, 3, best_box.unsqueeze(-1).expand(-1, -1, -1, 5)).squeeze(3)

        coord_loss = (
            self.mse(best_pred_box[obj_mask][..., :2], gt_box[obj_mask][..., :2]) +
            self.mse(torch.sqrt(best_pred_box[obj_mask][..., 2:4].clamp(1e-6)), torch.sqrt(gt_box[obj_mask][..., 2:4].clamp(1e-6)))
        )

        empty_confidence_loss = self.mse(
            pred_boxes[noobj_mask][..., 4], torch.zeros_like(pred_boxes[noobj_mask][..., 4])
        )

        non_empty_confidence_loss = self.mse(
            best_pred_box[obj_mask][..., 4], gt_box[obj_mask][..., 4]
        )

        classification_loss = self.ce(
            pred_classes[obj_mask].view(-1, self.C),
            gt_class[obj_mask].argmax(dim=-1)
        )

        total_loss = (
            self.lambda_coord * coord_loss +
            non_empty_confidence_loss +
            self.lambda_noobj * empty_confidence_loss +
            classification_loss
        )

        return total_loss
