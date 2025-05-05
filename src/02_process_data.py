import pprint

from pathlib import Path

import pickle

import uuid

from loguru import logger
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

from datasets import load_dataset

DATASET_NAME = "flickr30k"

SIZES = {
    1: "1",
    50: "50",
    100: "100",
    500: "500",
    1000: "1k",
    5000: "5k",
    10000: "10k",
}

@app.command()
def main():

    for _, size_str in SIZES.items():
        with open(RAW_DATA_DIR / DATASET_NAME / f"{size_str}.pkl", "rb") as file:
            dataset = pickle.load(file)

            images, captions = process_data(dataset)

            # print the first 5 images and captions
            logger.info(f"First 5 images and captions for {size_str}:")
            pprint.pprint(dict(list(images.items())[:5]))

            pprint.pprint("--------------------")

            pprint.pprint(dict(list(captions.items())[:5]))

            with open(PROCESSED_DATA_DIR / DATASET_NAME / f"{size_str}_images.pkl", "wb") as f:
                pickle.dump(images, f)
            with open(PROCESSED_DATA_DIR / DATASET_NAME / f"{size_str}_captions.pkl", "wb") as f:
                pickle.dump(captions, f)

def process_data(dataset):
    images = {}
    captions = {}

    caption_id = 0

    for _, item in enumerate(dataset):
        img_id = int(item["img_id"])

        if img_id not in images:
            images[img_id] = {
                "image": item["image"],
                "caption_ids": []  # Fixed string literal instead of variable
            }

        for caption in item["caption"]:
            images[img_id]["caption_ids"].append(caption_id)
            captions[caption_id] = {
                "caption": caption,
                "img_id": img_id
            }
            caption_id += 1

    return images, captions

def generate_id():
    # generate a unique id for each image
    return str(uuid.uuid4())

if __name__ == "__main__":
    app()
