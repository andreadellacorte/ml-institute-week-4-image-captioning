from PIL import Image
import io

from datetime import datetime

from src.model import UnifiedAutoregressiveDecoder

from src.dataset import ImageCaptioningDataset

from src.config import PROCESSED_DATA_DIR, FIGURES_DIR

from loguru import logger

import pickle
import random
import torch
import wandb
import os

from tqdm import tqdm

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)

def main():
    # Set wandb mode
    os.environ["WANDB_MODE"] = "disabled"  # Disable wandb for overfit test

    set_seed(42)

    # Always load 10 samples for overfit test
    with open(PROCESSED_DATA_DIR / "flickr30k/100_images.pkl", "rb") as f:
        images = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "flickr30k/100_captions.pkl", "rb") as f:
        captions = pickle.load(f)

    # Select 1 image for trivial overfit
    image_ids = list(images.keys())[:1]
    overfit_images = {id: images[id] for id in image_ids}

    # Use same data for train and test
    train_images = overfit_images
    test_images = overfit_images

    logger.info(f"Overfit mode: using {len(train_images)} images for both train and test")

    # Model config (fixed for overfit, dropout=0)
    model = UnifiedAutoregressiveDecoder(
        clip_model_name="openai/clip-vit-base-patch32",
        max_len=25,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
    )
    # Set all dropout layers to 0 (including attention, embeddings, etc.)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = ImageCaptioningDataset(
        train_images,
        captions,
        model)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Use same for validation
    val_dataset = ImageCaptioningDataset(
        test_images,
        captions,
        model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Reduced LR from 1e-2 to 1e-4
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

    # Print detailed debug info for the only sample
    sample = dataset[0]
    # Get caption_id from the dataset's internal data list
    caption_id_for_sample = dataset.data[0]["caption_id"]
    cleaned_caption = dataset.clean_text(captions[caption_id_for_sample]["caption"])
    label_str = f"{cleaned_caption} {model.tokenizer.eos_token}"
    decoded_label = model.decode_tokens(sample["label_ids"])
    logger.info(f"Cleaned caption: {cleaned_caption}")
    logger.info(f"Label string: {label_str}")
    logger.info(f"Decoded label from label_ids: {decoded_label}")
    logger.info(f"Label token ids: {sample['label_ids']}")
    logger.info(f"Input text: {model.decode_tokens(sample['input_ids'])}")
    logger.info(f"Input token ids: {sample['input_ids']}")

    # Overfit loop
    for epoch in range(20):  # Reduced from 300 to 20
        loss = train_one_epoch(model, dataloader, optimizer, criterion, full_sentence_step)
        logger.info(f"[Overfit] Epoch {epoch+1}/20, Loss: {loss:.4f}")
        # Print generated captions every epoch
        model.eval()
        with torch.no_grad():
            sample = val_dataset[0]
            image = sample["image_tensor"].unsqueeze(0).to(device)
            label_ids = sample["label_ids"].unsqueeze(0).to(device)
            # Force generation to match label length (excluding padding)
            label_len = (label_ids != model.tokenizer.pad_token_id).sum().item()
            generated = model.generate_caption(image, max_new_tokens=label_len)
            if isinstance(generated, list):
                generated_text = generated[0]
            else:
                generated_text = generated
            generated_ids = model.tokenizer(generated_text, return_tensors="pt").input_ids[0]
            logger.info(f"Sample 1:\n  Generated: {generated_text}\n  Ground truth: {model.decode_tokens(label_ids.squeeze(0))}")
            logger.info(f"Generated token ids: {generated_ids}")
            logger.info(f"Ground truth token ids: {label_ids.squeeze(0)}")
        model.train()

def train_one_epoch(model, dataloader, optimizer, criterion, step_function):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    device = next(model.parameters()).device
    for batch in progress_bar:
        images = batch["image_tensor"].to(device)
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Track total loss
        loss = step_function(model, images, input_ids, label_ids, attention_mask, optimizer, criterion)
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

def full_sentence_step(model, images, input_ids, label_ids, attention_mask, optimizer, criterion):
    optimizer.zero_grad()
    outputs = model(images, input_ids, attention_mask=attention_mask)
    loss = criterion(outputs.view(-1, outputs.size(-1)), label_ids.view(-1))
    loss.backward()
    optimizer.step()
    return loss

if __name__ == "__main__":
    main()