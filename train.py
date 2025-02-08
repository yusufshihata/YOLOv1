import torch
import torch.nn as nn
import config
from dataset import VOCDataset
from torch.utils.data import DataLoader
from validate import validate

def train(model, criterion, optimizer, train_loader, valid_loader):
    model.train()

    for epoch in range(config.EPOCHS):
        total_train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"EPOCH [{epoch+1}/{config.EPOCHS}] → Train Loss: {avg_train_loss:.4f}")

        # Call validate() after each epoch
        validate(model, criterion, valid_loader)

# Initialize Dataset and Dataloader
train_dataset = VOCDataset()
trainloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Run Training
train(config.model, config.criterion, config.optimizer, trainloader)
