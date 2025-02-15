import matplotlib.pyplot as plt

def plot_metrics(losses: list[float], maps: list[float]) -> None:
    """
    Plots the training loss and mAP over epochs.

    Args:
        losses (list): List of loss values per epoch.
        maps (list): List of mAP values per epoch.
    """
    epochs = list(range(1, len(losses) + 1))

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)

    # Plot mAP
    plt.subplot(1, 2, 2)
    plt.plot(epochs, maps, marker='o', linestyle='-', color='g', label="mAP")
    plt.xlabel("Epochs")
    plt.ylabel("mAP Score")
    plt.title("Mean Average Precision Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()