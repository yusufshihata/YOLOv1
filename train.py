import torch
import torch.nn as nn
import config
from dataset import VOCDataset
from torch.utils.data import DataLoader

def train(model, criterion, optimizer):
    trainloader = DataLoader(VOCDataset(), batch_size=config.BATCH_SIZE, shuffle=True)
    model.train()
    total_loss = 0.0
    for epoch in range(config.EPOCHS):
        for images, targets in trainloader:
            images = images.to(config.DEVICE)

            optimizer.zero_grad()

            pred = model(images)

            loss = criterion(pred, targets)

            loss.backward()

            optimizer.step()
            
            total_loss += loss.item()
        
        total_loss /= len(trainloader)
        
        print(f"EPOCH: [{epoch}/{config.EPOCHS}], Loss = {total_loss}")

train(config.model, config.criterion, config.optimizer)
