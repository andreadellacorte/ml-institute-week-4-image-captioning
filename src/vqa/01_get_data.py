from pathlib import Path
import pickle
import io
import numpy as np
from loguru import logger
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from datasets import load_dataset
import os
from tqdm import tqdm
import csv
import gc

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    hf_dataset: str = "lmms-lab/VQAv2",
    output_path: Path = PROCESSED_DATA_DIR / "VQAv2"
    # ----------------------------------------------
):
    sizes = {
        1: "1",
        10: "10",
        50: "50",
        100: "100",
        500: "500",
        1000: "1000",
        5000: "5000",
        10000: "10000",
    }

    logger.info(f"Loading dataset from {hf_dataset}...")

    # Make sure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Do not cache all items in memory; process streaming dataset for each size
    for size, size_str in sizes.items():
        logger.info(f"Loading dataset for first {size} items...")
        dataset = load_dataset(hf_dataset, split=f"validation", streaming=True)
        index = []
        images = {}
        count = 0
        for item in dataset:
            # Check for missing fields
            if any(field not in item or item[field] is None for field in ["image", "question", "multiple_choice_answer", "image_id", "question_id"]):
                logger.warning(f"Missing field in item: {item}")
                continue

            if len(item["multiple_choice_answer"].split()) != 1:
                logger.warning(f"Answer must be a single word, got: {item['multiple_choice_answer']}")
                continue

            try:
                img_bytes = io.BytesIO()
                img_format = item["image"].format or "PNG"
                item["image"].save(img_bytes, format=img_format)
                images[item["image_id"]] = {
                    "image_bytes": img_bytes.getvalue(),
                    "image_format": img_format,
                    "image_size": item["image"].size,
                    "image_mode": item["image"].mode
                }
                index.append({
                    "image_id": item["image_id"],
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "multiple_choice_answer": item["multiple_choice_answer"]
                })
                count += 1
                if count >= size:
                    break
            except Exception as e:
                logger.warning(f"Failed to process image for item {item.get('image_id', 'unknown')}: {e}")
                continue
            finally:
                # Explicitly close PIL image if possible
                try:
                    item["image"].close()
                except Exception:
                    pass

        if not index or not images:
            logger.warning(f"No valid items found for size {size_str}, skipping save.")
            continue

        logger.info(f"Saving {len(index)} valid items for size {size_str}...")
        try:
            # Save index
            index_file = output_path / f"{size_str}_index.pkl"
            with open(index_file, "wb") as f:
                pickle.dump(index, f)
            # Save images
            images_file = output_path / f"{size_str}_images.pkl"
            with open(images_file, "wb") as f:
                pickle.dump(images, f)
            logger.success(f"Saved {size_str}_index.pkl and {size_str}_images.pkl")
        except Exception as e:
            logger.error(f"Failed to save files for size {size_str}: {e}")

        # Force garbage collection to free resources
        gc.collect()

    logger.success("Dataset processing complete.")

if __name__ == "__main__":
    main()