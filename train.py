import torch
import torch.nn as nn
import config
from dataset import VOCDataset
from torch.utils.data import DataLoader
from validate import validate
from visualize import plot_metrics
from utils import save_model, compute_map, decode_predictions
from torch.optim.lr_scheduler import StepLR

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

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        all_pred_boxes = []
        all_true_boxes = []

        for images, targets in trainloader:
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
            optimizer.zero_grad()

            pred = model(images)
            loss = criterion(pred, targets)
            loss.backward()
            scheduler.step()

            total_loss += loss.item()

            # Convert predictions & targets to bounding boxes format
            pred_boxes = decode_predictions(pred)
            true_boxes = decode_predictions(targets)
            
            all_pred_boxes.extend(pred_boxes)
            all_true_boxes.extend(true_boxes)

        avg_loss = total_loss / len(trainloader)
        all_losses.append(avg_loss)

        # Compute mAP for the epoch
        epoch_map = compute_map(all_pred_boxes, all_true_boxes)
        all_maps.append(epoch_map)

        print(f"EPOCH [{epoch+1}/{config.EPOCHS}] - Loss: {avg_loss:.4f} - mAP: {epoch_map:.4f}")

        # Run validation step
        _, valid_map = validate(model, criterion, validloader)

        # Save the best model based on mAP
        if valid_map > best_map:
            best_map = valid_map
            save_model(model, optimizer, epoch+1, best_map)

    plot_metrics(all_losses, all_maps)