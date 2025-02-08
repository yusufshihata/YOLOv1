import torch
import torch.nn as nn
import config
from dataset import VOCDataset
from torch.utils.data import DataLoader
from validate import validate
from visualize import plot_metrics
from utils import save_model

def train(model, criterion, optimizer):
    trainloader = DataLoader(VOCDataset(), batch_size=config.BATCH_SIZE, shuffle=True)
    model.train()
    
    all_losses = []
    all_accuracies = []  # Optional: If you track accuracy
    best_loss = float("inf")  # Initialize with a very high value

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for images, targets in trainloader:
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy if applicable
            _, predicted_classes = pred[..., config.B * 5:].max(-1)
            _, true_classes = targets[..., config.B * 5:].max(-1)
            correct_preds += (predicted_classes == true_classes).sum().item()
            total_samples += targets.size(0) * config.S * config.S

        avg_loss = total_loss / len(trainloader)
        accuracy = correct_preds / total_samples

        all_losses.append(avg_loss)
        all_accuracies.append(accuracy)

        print(f"EPOCH [{epoch+1}/{config.EPOCHS}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch+1, best_loss)

    # Call visualization function
    plot_metrics(all_losses, all_accuracies)


# Initialize Dataset and Dataloader
train_dataset = VOCDataset()
trainloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Run Training
train(config.model, config.criterion, config.optimizer, trainloader)
