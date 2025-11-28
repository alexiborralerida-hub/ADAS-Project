
import torch
import cv2
import numpy as np
import os
import config
from model import UNet
import matplotlib.pyplot as plt

def predict(image_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet(n_channels=3, n_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not load image {image_path}")
        return
        
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image)
        # output shape: [1, num_classes, H, W]
        probs = torch.softmax(output, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

    # Visualization
    # Resize mask back to original size
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask overlay
    # Colors for 6 classes (0=background)
    colors = [
        [0, 0, 0],       # Background
        [255, 0, 0],     # Lane 1 (Blue)
        [0, 255, 0],     # Lane 2 (Green)
        [0, 0, 255],     # Lane 3 (Red)
        [255, 255, 0],   # Lane 4 (Cyan)
        [255, 0, 255]    # Lane 5 (Magenta)
    ]
    
    overlay = np.zeros_like(original_image)
    for i in range(1, config.NUM_CLASSES):
        overlay[pred_mask_resized == i] = colors[i]

    # Blend
    alpha = 0.5
    result = cv2.addWeighted(original_image, 1, overlay, alpha, 0)

    cv2.imwrite(output_path, result)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Example usage
    # Need to point to a valid image
    # test_image = "path/to/test_image.jpg"
    # predict(test_image, config.BEST_MODEL_PATH, "output.jpg")
    pass
