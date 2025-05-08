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

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    hf_dataset: str = "lmms-lab/VQAv2",
    output_path: Path = RAW_DATA_DIR / "VQAv2"
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
    
    # Only load up to 15,000 images from the test split using streaming
    dataset = load_dataset(hf_dataset, split="test", streaming=True)

    logger.info(f"Dataset loaded.")
    logger.info(f"Dataset info: {dataset}")
    
    # Prepare all items in memory (streaming, so must iterate once)
    all_items = []
    for item in dataset:
        all_items.append(item)
        if len(all_items) >= max(sizes.keys()):
            break

    for size, size_str in sizes.items():
        logger.info(f"Processing first {size} items...")
        subset = all_items[:size]
        index_rows = []
        images_rows = []
        for item in subset:
            image_id = str(item["image_id"])
            question_id = str(item["question_id"])
            question  = item["question"]
            answer = item["multiple_choice_answer"]
            # Index info
            index_rows.append({
                "image_id": image_id,
                "question_id": question_id,
                "question": question,
                "multiple_choice_answer": answer
            })
            # Image info
            img_bytes = io.BytesIO()
            img_format = item["image"].format or "PNG"
            item["image"].save(img_bytes, format=img_format)
            images_rows.append({
                "image_id": image_id,
                "image_bytes": img_bytes.getvalue(),
                "image_format": img_format,
                "image_size": item["image"].size,
                "image_mode": item["image"].mode
            })
        # Save index
        index_file = output_path / f"{size_str}_index.pkl"
        with open(index_file, "wb") as f:
            pickle.dump(index_rows, f)
        # Save images
        images_file = output_path / f"{size_str}_images.pkl"
        with open(images_file, "wb") as f:
            pickle.dump(images_rows, f)
        logger.success(f"Saved {size_str}_index.pkl and {size_str}_images.pkl")

    logger.success("Dataset processing complete.")

if __name__ == "__main__":
    main()