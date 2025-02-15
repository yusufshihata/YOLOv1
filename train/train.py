import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from inference.visualize import plot_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from config.config import DEVICE, EPOCHS
from train.validate import validate
from src.utils import save_model, compute_map, decode_bbox

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim,
    trainloader: DataLoader,
    validloader: DataLoader
) -> None:
    model.train()

    all_losses = []
    all_maps = []
    best_map = 0.0

    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        all_pred_boxes = []
        all_true_boxes = []

        for images, targets in trainloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            pred = model(images)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Convert predictions & targets to bounding boxes format
            pred_boxes = decode_bbox(pred)
            true_boxes = decode_bbox(targets)

            all_pred_boxes.extend(pred_boxes)
            all_true_boxes.extend(true_boxes)

        avg_loss = total_loss / len(trainloader)
        all_losses.append(avg_loss)

        scheduler.step()

        # Compute mAP for the epoch
        epoch_map = compute_map(all_pred_boxes, all_true_boxes)
        all_maps.append(epoch_map)

        print(f"EPOCH [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - mAP: {epoch_map:.4f}")

        # Run validation step
        _, valid_map = validate(model, criterion, validloader)

        # Save the best model based on mAP
        if valid_map > best_map:
            best_map = valid_map
            save_model(model, optimizer, epoch+1, best_map)

    plot_metrics(all_losses, all_maps)