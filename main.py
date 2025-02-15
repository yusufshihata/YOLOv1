import argparse
import torch
from config.config import model, criterion, optimizer, DEVICE, BATCH_SIZE, NUM_WORKERS
from src.dataset import VOCDataset
from train.train import train
from train.validate import validate
from inference.inference import predict
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="YOLOv1 Training & Inference")
    parser.add_argument("mode", choices=["train", "validate", "inference"], help="Choose mode: train, validate, or inference")
    parser.add_argument("--image", type=str, help="Path to an image for inference (required in inference mode)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint for inference or resume training")

    args = parser.parse_args()

    if args.checkpoint:
        from src.utils import load_checkpoint
        load_checkpoint(model, args.checkpoint, optimizer)

    if args.mode == "train":
        # Initialize Datasets
        train_dataset = VOCDataset(root="datasets", transform=None)  # Change transform as needed
        valid_dataset = VOCDataset(root="datasets", image_set="val", transform=None)

        # Create Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # Train Model
        train(model, criterion, optimizer, train_loader, valid_loader)

    elif args.mode == "validate":
        valid_dataset = VOCDataset(root="datasets", image_set="val", transform=None)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        validate(model, criterion, valid_loader)

    elif args.mode == "inference":
        if not args.image:
            raise ValueError("Inference mode requires --image argument")
        predict(args.image, model)

if __name__ == "__main__":
    main()
