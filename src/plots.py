from pathlib import Path

from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

def plot_images_with_captions(images, captions, title="Generated Captions", save_path=None):
    """
    Plot a grid of images with their captions underneath.
    
    Args:
        images: List of PIL Image objects or image tensors
        captions: List of captions (strings) corresponding to images
        title: Title for the overall plot
        save_path: Optional path to save the figure
    """
    n_images = len(images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    plt.suptitle(title, fontsize=16)
    
    for i, (img, caption) in enumerate(zip(images, captions)):
        if i >= len(axes):
            break
            
        # Convert to PIL Image if it's a tensor or bytes
        if not isinstance(img, Image.Image):
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            else:
                # Assuming it's a tensor, convert to numpy and adjust
                img_np = img.detach().cpu().numpy()
                if img_np.shape[0] in [1, 3]:  # If in [C, H, W] format
                    img_np = np.transpose(img_np, (1, 2, 0))
                # Normalize if needed
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                img = Image.fromarray(img_np)
        
        # Display image
        axes[i].imshow(img)
        axes[i].set_title(f"Image {i+1}", fontsize=12)
        axes[i].text(0.5, -0.15, caption, 
                    horizontalalignment='center',
                    verticalalignment='center', 
                    transform=axes[i].transAxes,
                    fontsize=10, wrap=True)
        axes[i].axis('off')
        
    # Hide any unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()