import argparse
from dataset import VOCDataset
from torch.utils.data import DataLoader
import config
from train import train
from inference import predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['train', 'inference'], help="Choose mode: train or inference")
    parser.add_argument("--image", type=str, help="path to image for inference", required=False)
    args = parser.parse_args()

    if args.mode == "train":
        # Initialize Dataset and Dataloader
        train_dataset = VOCDataset(transform=config.augs)
        trainloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

        # Run Training
        train(config.model, config.criterion, config.optimizer, trainloader)
    elif args.mode == "inference":
        if not args.image:
            raise ValueError("Inference mode requires --image argument")
        predict(args.image, config.model)

if __name__ == "__main__":
    main()
