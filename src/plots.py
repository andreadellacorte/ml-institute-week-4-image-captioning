from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def plot_images_with_captions(images, captions, title="Generated Captions", save_path=None, show=False):
    """
    Plot a grid of images with their captions underneath, preserving aspect ratio and limiting height so all images fit well.
    Args:
        images: List of image bytes (or PIL Images)
        captions: List of captions (strings) corresponding to images
        title: Title for the overall plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    n_images = len(images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    # Determine max image height (in inches) so 2 rows fit in ~6 inches
    max_total_height = 6  # inches for all rows
    max_img_height = max_total_height / n_rows

    # Estimate average aspect ratio
    aspect_ratios = []
    pil_images = []
    for img in images:
        pil_img = Image.open(io.BytesIO(img)) if isinstance(img, (bytes, bytearray)) else img
        pil_images.append(pil_img)
        w, h = pil_img.size
        aspect_ratios.append(w / h)
    avg_aspect = np.mean(aspect_ratios) if aspect_ratios else 1.0

    # Set width per image based on aspect ratio and max_img_height
    width_per_img = max_img_height * avg_aspect
    fig_width = n_cols * width_per_img
    fig_height = n_rows * max_img_height

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    plt.suptitle(title, fontsize=16)

    for i, (img, caption) in enumerate(zip(pil_images, captions)):
        if i >= len(axes):
            break
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

    if show:
        plt.show()