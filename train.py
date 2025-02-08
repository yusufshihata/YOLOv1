import torch
import torch.nn as nn
import config
from dataset import VOCDataset
from torch.utils.data import DataLoader

def train(model, criterion, optimizer, dataloader):
    model.train()
    
    for epoch in range(config.EPOCHS):
        total_loss = 0.0  # Reset loss at start of each epoch
        
        for images, targets in dataloader:
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()

            pred = model(images)

            # Ensure correct shape
            pred = pred.view(-1, config.S, config.S, (config.B * 5 + config.C))
            targets = targets.view(-1, config.S, config.S, (5 + config.C))

            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"EPOCH: [{epoch+1}/{config.EPOCHS}], Loss = {avg_loss:.4f}")

# Initialize Dataset and Dataloader
train_dataset = VOCDataset(split="train")  # Ensure split is provided
trainloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Run Training
train(config.model, config.criterion, config.optimizer, trainloader)