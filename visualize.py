import matplotlib.pyplot as plt

def plot_metrics(losses, accuracies=None):
    """
    Plots the training loss and optionally accuracy over epochs.

    Args:
        losses (list): List of loss values per epoch.
        accuracies (list, optional): List of accuracy values per epoch.
    """
    epochs = list(range(1, len(losses) + 1))

    plt.figure(figsize=(10, 5))

    # Plot Loss
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label="Training Loss")

    # Plot Accuracy if provided
    if accuracies:
        plt.plot(epochs, accuracies, marker='s', linestyle='--', color='g', label="Training Accuracy")
        plt.ylabel("Loss / Accuracy")
    else:
        plt.ylabel("Loss")

    plt.xlabel("Epochs")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()
