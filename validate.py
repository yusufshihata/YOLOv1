import torch
import config

def validate(model, criterion, valid_loader):
    model.eval()
    total_valid_loss = 0.0
    
    with torch.no_grad():
        for images, targets in valid_loader:
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
            
            pred = model(images)
            loss = criterion(pred, targets)
            
            total_valid_loss += loss.item()
    
    avg_valid_loss = total_valid_loss / len(valid_loader)
    print(f"Validation Loss: {avg_valid_loss:.4f}")
    
    return avg_valid_loss