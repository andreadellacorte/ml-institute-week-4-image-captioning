from pathlib import Path
import pickle
import io
import numpy as np
from loguru import logger
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from datasets import load_dataset
import os
from tqdm import tqdm

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    hf_dataset: str = "nlphuji/flickr30k",
    output_path: Path = RAW_DATA_DIR / "flickr30k"
    # ----------------------------------------------
):
    sizes = {
        1: "1",
        50: "50",
        100: "100",
        500: "500",
        1000: "1000",
        5000: "5000",
        31014: "full",
    }

    logger.info(f"Loading dataset from {hf_dataset}...")

    # Make sure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(hf_dataset)

    logger.info(f"Dataset loaded.")
    logger.info(f"Dataset info: {dataset}")
    
    # Check if we're getting the full dataset
    test_sample = dataset["test"][0]
    logger.info(f"First sample image type: {type(test_sample['image'])}")
    logger.info(f"First sample image format: {test_sample['image'].format}")
    logger.info(f"First sample image size: {test_sample['image'].size}")
    logger.info(f"First sample image mode: {test_sample['image'].mode}")

    # Save original image as PNG to check true size of first sample
    test_img_path = output_path / "test_sample_lossless.png"
    test_sample['image'].save(test_img_path, format="PNG", compress_level=0)
    raw_img_size = os.path.getsize(test_img_path)
    logger.info(f"First sample lossless PNG size: {raw_img_size / 1024:.2f} KB")

    # save sizes
    for size, size_str in sizes.items():
        logger.info(f"Saving {size_str} samples...")
        
        # Select the dataset subset
        subset = dataset["test"].select(range(min(size, len(dataset["test"]))))
        
        # Process data to ensure images are properly stored
        processed_data = []
        total_raw_bytes = 0
        total_saved_bytes = 0
        
        # Add tqdm progress bar updated every 100 iterations
        pbar = tqdm(total=len(subset), desc=f"Processing {size_str} samples", 
                   unit="img", miniters=min(100, max(1, len(subset)//100)))
        
        for item in subset:
            # Keep the original PIL image representation too - just serialize properly
            # Try to use original format, but if not available use PNG lossless
            img_format = item["image"].format
            if img_format is None:
                img_format = "PNG"  # Default to PNG for unknown formats as it's lossless
            
            # Store the original image data
            img_bytes = io.BytesIO()
            
            # Use lossless or highest quality settings for each format
            if img_format.upper() == "JPEG":
                # For JPEG, use quality=100 to minimize lossy compression
                # Cannot be truly lossless as JPEG is inherently lossy
                item["image"].save(img_bytes, format=img_format, quality=100, subsampling=0)
            elif img_format.upper() in ["PNG", "GIF", "BMP", "TIFF", "WEBP"]:
                # For other formats that support lossless, use lossless settings
                # PNG compress_level=0 is fastest but least compression (still lossless)
                item["image"].save(img_bytes, format=img_format, compress_level=0)
            else:
                # For other formats, use default settings
                # Convert to PNG to ensure lossless storage
                logger.info(f"Converting unknown format {img_format} to lossless PNG")
                img_format = "PNG"
                item["image"].save(img_bytes, format=img_format, compress_level=0)
            
            # Get raw image size and saved size for comparison
            orig_width, orig_height = item["image"].size
            orig_mode = item["image"].mode
            bytes_per_pixel = len(orig_mode)  # RGB=3, RGBA=4, L=1, etc.
            raw_pixel_count = orig_width * orig_height * bytes_per_pixel
            bytes_size = len(img_bytes.getvalue())
            
            total_raw_bytes += raw_pixel_count
            total_saved_bytes += bytes_size
            
            # Replace PIL image with bytes array and all metadata
            processed_item = dict(item)
            processed_item["image_bytes"] = img_bytes.getvalue()
            processed_item["image_format"] = img_format
            processed_item["image_size"] = item["image"].size
            processed_item["image_mode"] = item["image"].mode
            processed_item["bytes_size"] = bytes_size
            processed_item["original_format"] = item["image"].format
            
            # Remove the PIL image object as it's not needed anymore
            del processed_item["image"]
            
            processed_data.append(processed_item)
            
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Report compression ratio
        logger.info(f"Total raw pixel data: {total_raw_bytes/1024/1024:.2f} MB")
        logger.info(f"Total saved bytes: {total_saved_bytes/1024/1024:.2f} MB")
        logger.info(f"Average bytes per image: {total_saved_bytes/len(processed_data)/1024:.2f} KB")
        
        # Use pickle protocol 4 for better handling of large data
        output_file = output_path / f"{size_str}.pkl"
        logger.info(f"Saving processed data to {output_file}...")
        with open(output_file, "wb") as f:
            pickle.dump(processed_data, f, protocol=4)
        
        # Check file size after pickling
        file_size = output_file.stat().st_size
        logger.info(f"Saved {size_str} samples to {output_file}")
        logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
        logger.info(f"Average per sample: {file_size / len(processed_data) / 1024:.2f} KB")

    # Check the final size of all files in the dataset
    total_size = sum(f.stat().st_size for f in output_path.glob("*.pkl"))
    logger.info(f"Total size of all dataset files: {total_size / 1024 / 1024:.2f} MB")
    
    # Compare with expected size (this will help explain the difference with the 4.4GB on Hugging Face)
    logger.info(f"Original HuggingFace dataset claims to be approximately 4.4 GB")
    logger.info(f"If the total size is significantly less, the difference could be due to:")
    logger.info(f" 1. The original dataset includes extra data not in the images")
    logger.info(f" 2. The dataset's reported size includes caching or implementation overhead")
    logger.info(f" 3. Different storage formats (raw vs. compressed)")

    logger.success("Dataset processing complete.")

if __name__ == "__main__":
    main()