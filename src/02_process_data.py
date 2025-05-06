import pprint
import pickle
import uuid
from loguru import logger
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "flickr30k"

SIZES = {
    1: "1",
    50: "50",
    100: "100",
    500: "500",
    1000: "1000",
    5000: "5000",
    31014: "full",
}

def main():
    # Create output directory if it doesn't exist
    (PROCESSED_DATA_DIR / DATASET_NAME).mkdir(parents=True, exist_ok=True)

    for _, size_str in SIZES.items():
        logger.info(f"Processing {size_str} samples...")
        try:
            with open(RAW_DATA_DIR / DATASET_NAME / f"{size_str}.pkl", "rb") as file:
                dataset = pickle.load(file)

                images, captions = process_data(dataset)

                # print the first 5 images and captions
                logger.info(f"First 5 images and captions for {size_str}:")
                # Only print non-binary data for readability
                first_five_images = dict(list(images.items())[:5])
                for img_id, img_data in first_five_images.items():
                    # Don't print the binary data
                    first_five_images[img_id] = {
                        "image_size": img_data["image_size"],
                        "image_format": img_data["image_format"],
                        "image_mode": img_data["image_mode"],
                        "image_bytes_length": len(img_data["image_bytes"]) if "image_bytes" in img_data else "N/A",
                        "caption_ids": img_data["caption_ids"]
                    }
                pprint.pprint(first_five_images)

                pprint.pprint("--------------------")

                pprint.pprint(dict(list(captions.items())[:5]))

                logger.info(f"Saving processed {size_str} images and captions...")
                with open(PROCESSED_DATA_DIR / DATASET_NAME / f"{size_str}_images.pkl", "wb") as f:
                    pickle.dump(images, f)
                with open(PROCESSED_DATA_DIR / DATASET_NAME / f"{size_str}_captions.pkl", "wb") as f:
                    pickle.dump(captions, f)
                logger.success(f"Processed and saved {size_str} dataset")
        except FileNotFoundError:
            logger.warning(f"File {size_str}.pkl not found, skipping")

def process_data(dataset):
    images = {}
    captions = {}
    caption_id = 0

    # Create a progress bar that updates every 100 iterations
    pbar = tqdm(total=len(dataset), desc="Processing items", 
               unit="item", miniters=min(100, max(1, len(dataset)//100)))
    
    for item in dataset:
        img_id = int(item["img_id"])

        if img_id not in images:
            images[img_id] = {
                "image_bytes": item["image_bytes"],
                "image_format": item["image_format"],
                "image_size": item["image_size"],
                "image_mode": item["image_mode"],
                "caption_ids": []
            }
            
            # Add any extra fields that might be in the item
            for key in item:
                if key not in ["image_bytes", "image_format", "image_size", "image_mode", "caption"] and key.startswith("image_"):
                    images[img_id][key] = item[key]

        for caption in item["caption"]:
            images[img_id]["caption_ids"].append(caption_id)
            captions[caption_id] = {
                "caption": caption,
                "img_id": img_id
            }
            caption_id += 1
            
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    logger.info(f"Processed {len(images)} unique images and {caption_id} captions")
    return images, captions

def generate_id():
    # generate a unique id for each image
    return str(uuid.uuid4())

if __name__ == "__main__":
    main()
