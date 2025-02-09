import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import DEVICE, IMAGE_SIZE, transformation
from utils import decode_predictions, non_max_suppression, load_checkpoint

def predict(image_path: str, model: nn.Module, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> None:
    """
    Runs inference on an image using the YOLOv1 model.
    
    Args:
        image_path (str): Path to the input image.
        model (nn.Module): The trained YOLOv1 model.
        conf_threshold (float): Confidence threshold for predictions.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression.
    """
    model = load_checkpoint(model)

    model.eval()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess image
    img_tensor = transformation(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)
    
    # Decode and filter predictions
    bboxes = decode_predictions(pred, conf_threshold)
    filtered_bboxes = non_max_suppression(bboxes, iou_threshold)
    
    # Plot image with predictions
    _, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    
    for box in filtered_bboxes:
        x, y, w, h, label = box  # Extract values
        
        # Convert YOLO format (center x, center y, width, height) to (x_min, y_min, width, height)
        x_min = x - w / 2
        y_min = y - h / 2
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), w, h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, label, color="red", fontsize=12, weight="bold")
    
    plt.show()
