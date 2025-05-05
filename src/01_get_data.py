from pathlib import Path

import pickle

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

from datasets import load_dataset

@app.command()
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
        1000: "1k",
        5000: "5k",
        10000: "10k",
    }

    logger.info(f"Loading dataset from {hf_dataset}...")

    dataset = load_dataset(hf_dataset)

    logger.info(f"Dataset loaded.")

    # save sizes
    for size, size_str in sizes.items():
        logger.info(f"Saving {size_str} samples...")
        # use pickle to save the dataset
        with open(output_path / f"{size_str}.pkl", "wb") as f:
            pickle.dump(dataset["test"].select(range(size)), f)

    logger.success("Dataset processing complete.")

if __name__ == "__main__":
    app()
